import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import model.model_base as model_base

class MVAE(model_base.BaseModel):
    NAME = 'MVAE'
    def __init__(self, config, dataset, device):
        super().__init__(config)
        self.config = config
        self.device = device
        self.config['frame_dim'] =  self.frame_dim = dataset.frame_dim
        
        self.num_condition_frames = self.config["data"]["num_condition_frames"]
        self.num_future_predictions = self.config["data"]["num_future_predictions"]
        self.latent_size = self.config["model_hyperparam"]["latent_size"]

        self.vae_type = self.config["MVAE"]["vae_type"]
        self.kl_weight = self.config["MVAE"]["kl_weight"]
        self.recon_weight = self.config["MVAE"]["recon_weight"]
        self._build_model(config)
        return

    def _build_model(self, config):
        if self.vae_type == 'vae':
            self.model = PoseVAEModel(config)
        elif self.vae_type == 'vaemoe':
            self.model = PoseMixtureVAE(config)
        elif self.vae_type == 'vaevq':
            self.model = PoseVQVAE(config)
        elif self.vae_type == 'vaemos':
            self.model = PoseMixtureSpecialistVAE(config)
        else:
            assert (False), 'vae model type available: vae, vaemoe, vaevq, vaemos'
        self.model.to(self.device)
        return

    def eval_step(self, cur_x, extra_dict):
        z = torch.randn(cur_x.shape[0],self.latent_size, device=cur_x.device)
        next_x = self.model.sample(z, cur_x) 
        return next_x

    def eval_seq(self, start_x, extra_dict, num_steps, num_trials):
        output_xs = torch.zeros((num_trials, num_steps, self.frame_dim))
        start_x = start_x[None,:].expand(num_trials, -1)
        for j in range(num_steps):
            with torch.no_grad():
                start_x = self.eval_step(start_x, extra_dict).detach()
            output_xs[:,j,:] = start_x 
        return output_xs
    
    def compute_loss(self, last_x, next_x, cur_epoch, extra_dict):
        loss, loss_dict = self.feed_vae(last_x, next_x)
        return loss, loss_dict

    def get_model_params(self):
        params = list(self.model.parameters())
        return params

    def feed_vae(self, condition, ground_truth):
        #print(condition.shape, ground_truth.shape)
        condition = condition.flatten(start_dim=1, end_dim=-1)
        flattened_truth = ground_truth.flatten(start_dim=1, end_dim=-1)

        output_shape = (-1, self.num_future_predictions, self.model.frame_size)

        if self.vae_type == 'vaevq':
            vae_output, vq_loss, perplexity = self.model(flattened_truth, condition)
            vae_output = vae_output.view(output_shape)
            vq_loss = torch.mean(vq_loss)
            perplexity = torch.mean(perplexity)
            recon_loss = (vae_output - ground_truth).pow(2).mean(dim=(0, -1))
            recon_loss = torch.mean(recon_loss)
            loss = self.recon_weight * recon_loss + self.kl_weight * vq_loss
            loss_dict = {"recon_loss":recon_loss, "vq_loss":vq_loss, "perplexity": perplexity}
            return loss, loss_dict

        elif self.vae_type == 'vaemos':
            vae_output, mu, logvar, coefficient = self.model(flattened_truth, condition)

            recon_loss = (vae_output - ground_truth).pow(2).mean(dim=2).mul(-0.5).exp()
            recon_loss = (recon_loss * coefficient).sum(dim=1).log().mul(-1).mean()
            recon_loss = torch.mean(recon_loss)
            indices = torch.distributions.Categorical(coefficient).sample()
            vae_output = vae_output[torch.arange(vae_output.size(0)), indices]
            vae_output = vae_output.view(output_shape)

            kl_loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum().clamp(max=0)
            kl_loss /= logvar.numel()
            kl_loss = torch.mean(kl_loss)

            loss = self.recon_weight * recon_loss + self.kl_weight * kl_loss
            loss_dict = {"recon_loss":recon_loss, "kl_loss":kl_loss}

            return loss, loss_dict

        elif self.vae_type in ['vae','vaemoe']:
            # PoseVAE and PoseMixtureVAE
            vae_output, mu, logvar = self.model(flattened_truth, condition)
            vae_output = vae_output.view(output_shape)

            kl_loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum().clamp(max=0)
            kl_loss /= logvar.numel()
            kl_loss = torch.mean(kl_loss)

            recon_loss = (vae_output - ground_truth).pow(2).mean(dim=(0, -1))
            recon_loss = torch.mean(recon_loss)
            loss = self.recon_weight * recon_loss + self.kl_weight * kl_loss
            loss_dict = {"recon_loss":recon_loss, "kl_loss":kl_loss}
            
            return loss, loss_dict
        else:
            assert (False), 'vae model type available: vae, vaemoe, vaevq, vaemos'

class PoseMixtureVAE(nn.Module):
    def __init__(
        self,
        config
    ):
        super().__init__()
        self.frame_size = config['frame_dim'] 
        self.latent_size = config["model_hyperparam"]["latent_size"] 
        self.num_condition_frames = config["data"]["num_condition_frames"]
        self.num_future_predictions = config["data"]["num_future_predictions"]
        self.num_experts = config["MVAE"]["num_expert"]
        self.hidden_size = config["model_hyperparam"]["hidden_size"] 
        args = (
            self.frame_size,
            self.latent_size,
            self.hidden_size,
            self.num_condition_frames,
            self.num_future_predictions,
        )
        self.encoder = Encoder(*args)
        self.decoder = MixedDecoder(*args, self.num_experts)

    def encode(self, x, c):
        _, mu, logvar = self.encoder(x, c)
        return mu, logvar

    def forward(self, x, c):
        z, mu, logvar = self.encoder(x, c)
        return self.decoder(z, c), mu, logvar

    def sample(self, z, c, deterministic=False):
        return self.decoder(z, c)



class PoseVAEModel(nn.Module):
    
    def __init__(
        self,
        config
        ):
        super().__init__()
        
        self.frame_size = config['frame_dim'] 
        self.latent_size = config["model_hyperparam"]["latent_size"] 
        self.num_condition_frames = config["data"]["num_condition_frames"]
        self.num_future_predictions = config["data"]["num_future_predictions"]

        h1 = config["model_hyperparam"]["hidden_size"] 
        # Encoder
        # Takes pose | condition (n * poses) as input
        self.fc1 = nn.Linear(
            self.frame_size * (self.num_future_predictions + self.num_condition_frames), h1
        )
        self.fc2 = nn.Linear(self.frame_size + h1, h1)
        # self.fc3 = nn.Linear(h1, h1)
        self.mu = nn.Linear(self.frame_size + h1, self.latent_size)
        self.logvar = nn.Linear(self.frame_size + h1, self.latent_size)

        # Decoder
        # Takes latent | condition as input
        self.fc4 = nn.Linear(self.latent_size + self.frame_size * self.num_condition_frames, h1)
        self.fc5 = nn.Linear(self.latent_size + h1, h1)
        # self.fc6 = nn.Linear(latent_size + h1, h1)
        self.out = nn.Linear(self.latent_size + h1, self.num_future_predictions * self.frame_size)


    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, c), mu, logvar

    def encode(self, x, c):
        h1 = F.elu(self.fc1(torch.cat((x, c), dim=1)))
        h2 = F.elu(self.fc2(torch.cat((x, h1), dim=1)))
        # h3 = F.elu(self.fc3(h2))
        s = torch.cat((x, h2), dim=1)
        return self.mu(s), self.logvar(s)

    def decode(self, z, c):
        h4 = F.elu(self.fc4(torch.cat((z, c), dim=1)))
        h5 = F.elu(self.fc5(torch.cat((z, h4), dim=1)))
        # h6 = F.elu(self.fc6(torch.cat((z, h5), dim=1)))
        return self.out(torch.cat((z, h5), dim=1))

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def sample(self, z, c, deterministic=False):
        return self.decode(z, c)



class AutoEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.frame_size = config['frame_dim'] 
        self.latent_size = config["model_hyperparam"]["latent_size"] 
       
        h1 = 256
        h2 = 128
        # Encoder
        # Takes pose | condition (n * poses) as input
        self.fc1 = nn.Linear(frame_size, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, latent_size)

        # Decoder
        # Takes latent | condition as input
        self.fc4 = nn.Linear(latent_size, h2)
        self.fc5 = nn.Linear(h2, h1)
        self.fc6 = nn.Linear(h1, frame_size)

    def forward(self, x):
        latent = self.encode(x)
        return self.decode(latent)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        return self.fc3(h2)

    def decode(self, x):
        h4 = F.relu(self.fc4(x))
        h5 = F.relu(self.fc5(h4))
        return self.fc6(h5)


class Encoder(nn.Module):
    def __init__(
        self,
        frame_size,
        latent_size,
        hidden_size,
        num_condition_frames,
        num_future_predictions,
    ):
        super().__init__()
        # Encoder
        # Takes pose | condition (n * poses) as input
        input_size = frame_size * (num_future_predictions + num_condition_frames)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(frame_size + hidden_size, hidden_size)
        self.mu = nn.Linear(frame_size + hidden_size, latent_size)
        self.logvar = nn.Linear(frame_size + hidden_size, latent_size)

    def encode(self, x, c):
        h1 = F.elu(self.fc1(torch.cat((x, c), dim=1)))
        h2 = F.elu(self.fc2(torch.cat((x, h1), dim=1)))
        s = torch.cat((x, h2), dim=1)
        return self.mu(s), self.logvar(s)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar


class Decoder(nn.Module):
    def __init__(
        self,
        frame_size,
        latent_size,
        hidden_size,
        num_condition_frames,
        num_future_predictions,
    ):
        super().__init__()
        # Decoder
        # Takes latent | condition as input
        input_size = latent_size + frame_size * num_condition_frames
        output_size = num_future_predictions * frame_size
        self.fc4 = nn.Linear(input_size, hidden_size)
        self.fc5 = nn.Linear(latent_size + hidden_size, hidden_size)
        self.out = nn.Linear(latent_size + hidden_size, output_size)

    def decode(self, z, c):
        h4 = F.elu(self.fc4(torch.cat((z, c), dim=1)))
        h5 = F.elu(self.fc5(torch.cat((z, h4), dim=1)))
        return self.out(torch.cat((z, h5), dim=1))

    def forward(self, z, c):
        return self.decode(z, c)


class MixedDecoder(nn.Module):
    def __init__(
        self,
        frame_size,
        latent_size,
        hidden_size,
        num_condition_frames,
        num_future_predictions,
        num_experts,
    ):
        super().__init__()

        input_size = latent_size + frame_size * num_condition_frames
        inter_size = latent_size + hidden_size
        output_size = num_future_predictions * frame_size
        self.decoder_layers = [
            (
                nn.Parameter(torch.empty(num_experts, input_size, hidden_size)),
                nn.Parameter(torch.empty(num_experts, hidden_size)),
                F.elu,
            ),
            (
                nn.Parameter(torch.empty(num_experts, inter_size, hidden_size)),
                nn.Parameter(torch.empty(num_experts, hidden_size)),
                F.elu,
            ),
            (
                nn.Parameter(torch.empty(num_experts, inter_size, output_size)),
                nn.Parameter(torch.empty(num_experts, output_size)),
                None,
            ),
        ]

        for index, (weight, bias, _) in enumerate(self.decoder_layers):
            index = str(index)
            torch.nn.init.kaiming_uniform_(weight)
            bias.data.fill_(0.01)
            self.register_parameter("w" + index, weight)
            self.register_parameter("b" + index, bias)

        # Gating network
        gate_hsize = 64
        self.gate = nn.Sequential(
            nn.Linear(input_size, gate_hsize),
            nn.ELU(),
            nn.Linear(gate_hsize, gate_hsize),
            nn.ELU(),
            nn.Linear(gate_hsize, num_experts),
        )

    def forward(self, z, c):
        coefficients = F.softmax(self.gate(torch.cat((z, c), dim=1)), dim=1)
        layer_out = c
        for (weight, bias, activation) in self.decoder_layers:
            flat_weight = weight.flatten(start_dim=1, end_dim=2)
            mixed_weight = torch.matmul(coefficients, flat_weight).view(
                coefficients.shape[0], *weight.shape[1:3]
            )
            input = torch.cat((z, layer_out), dim=1).unsqueeze(1)
            mixed_bias = torch.matmul(coefficients, bias).unsqueeze(1)
            out = torch.baddbmm(mixed_bias, input, mixed_weight).squeeze(1)
            layer_out = activation(out) if activation is not None else out

        return layer_out




class PoseMixtureSpecialistVAE(nn.Module):
    def __init__(
        self,
        config
    ):
        super().__init__()
        self.frame_size = config['frame_dim'] 
        self.latent_size = config["model_hyperparam"]["latent_size"] 
        self.num_condition_frames = self.config["dataset"]["num_condition_frames"]
        self.num_future_predictions = self.config["dataset"]["num_future_predictions"]


        self.hidden_size = config["model_hyperparam"]["hidden_size"] 
        args = (
            self.frame_size,
            self.latent_size,
            self.hidden_size,
            self.num_condition_frames,
            self.num_future_predictions,
        )

        self.encoder = Encoder(*args)

        self.decoders = []
        for i in range(num_experts):
            decoder = Decoder(*args)
            self.decoders.append(decoder)
            self.add_module("d" + str(i), decoder)

        # Gating network
        gate_hsize = 128
        input_size = self.latent_size + self.frame_size * self.num_condition_frames
        self.g_fc1 = nn.Linear(input_size, gate_hsize)
        self.g_fc2 = nn.Linear(self.latent_size + gate_hsize, gate_hsize)
        self.g_fc3 = nn.Linear(self.latent_size + gate_hsize, self.num_experts)

    def gate(self, z, c):
        h1 = F.elu(self.g_fc1(torch.cat((z, c), dim=1)))
        h2 = F.elu(self.g_fc2(torch.cat((z, h1), dim=1)))
        return self.g_fc3(torch.cat((z, h2), dim=1))

    def forward(self, x, c):
        z, mu, logvar = self.encoder(x, c)
        coefficients = F.softmax(self.gate(z, c), dim=1)
        predictions = torch.stack([decoder(z, c) for decoder in self.decoders], dim=1)
        return predictions, mu, logvar, coefficients

    def sample(self, z, c, deterministic=False):
        coefficients = F.softmax(self.gate(z, c), dim=1)
        predictions = torch.stack([decoder(z, c) for decoder in self.decoders], dim=1)

        if not deterministic:
            dist = torch.distributions.Categorical(coefficients)
            indices = dist.sample()
        else:
            indices = coefficients.argmax(dim=1)

        return predictions[torch.arange(predictions.size(0)), indices]

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, latent_size):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.latent_size = latent_size

        # self.embedding = nn.Embedding(self.num_embeddings, self.latent_size)
        # self.embedding.weight.data.normal_()

        embed = torch.randn(latent_size, num_embeddings)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(num_embeddings))
        self.register_buffer("embed_avg", embed.clone())

        self.commitment_cost = 0.25
        self.decay = 0.99
        self.epsilon = 1e-5

    def forward(self, inputs):
        # Calculate distances
        dist = (
            inputs.pow(2).sum(1, keepdim=True)
            - 2 * inputs @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )

        _, embed_ind = (-dist).max(1)
        embed_onehot = F.one_hot(embed_ind, self.num_embeddings).type(inputs.dtype)
        embed_ind = embed_ind.view(*inputs.shape[:-1])
        quantize = F.embedding(embed_ind, self.embed.transpose(0, 1))

        # Use EMA to update the embedding vectors
        if self.training:
            self.cluster_size.data.mul_(self.decay).add_(
                1 - self.decay, embed_onehot.sum(0)
            )

            embed_sum = inputs.transpose(0, 1) @ embed_onehot
            self.embed_avg.data.mul_(self.decay).add_(1 - self.decay, embed_sum)
            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.epsilon)
                / (n + self.num_embeddings * self.epsilon)
                * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

        loss = (quantize.detach() - inputs).pow(2).mean()
        quantize = inputs + (quantize - inputs).detach()

        avg_probs = embed_onehot.mean(dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * (avg_probs + 1e-10).log()))

        return quantize, loss, perplexity, embed_ind


class PoseVQVAE(nn.Module):
    def __init__(
        self,
        config
    ):
        super().__init__()
        
        self.frame_size = config['frame_dim'] 
        self.latent_size = config["model_hyperparam"]["latent_size"] 
        self.num_condition_frames = config["data"]["num_condition_frames"]
        self.num_future_predictions = config["data"]["num_future_predictions"]
        self.num_embeddings = config["MVAE"]["num_embeddings"]
        h1 = config["model_hyperparam"]["hidden_size"]# 512 
        # Encoder
        # Takes pose | condition (n * poses) as input
        self.fc1 = nn.Linear(
            self.frame_size * (self.num_future_predictions + self.num_condition_frames), h1
        )
        self.fc2 = nn.Linear(h1, h1)
        self.fc3 = nn.Linear(h1, h1)
        self.mu = nn.Linear(h1, self.latent_size)

        # Decoder
        # Takes latent | condition as input
        self.fc4 = nn.Linear(self.latent_size + self.frame_size * self.num_condition_frames, h1)
        self.fc5 = nn.Linear(h1, h1)
        self.fc6 = nn.Linear(h1, h1)
        self.out = nn.Linear(h1, self.num_future_predictions * self.frame_size)

        self.quantizer = VectorQuantizer(self.num_embeddings, self.latent_size)

    def forward(self, x, c):
        mu = self.encode(x, c)
        quantized, loss, perplexity, _ = self.quantizer(mu)
        recon = self.decode(quantized, c)
        return recon, loss, perplexity

    def encode(self, x, c):
        s = torch.cat((x, c), dim=1)
        h1 = F.relu(self.fc1(s))
        h2 = F.relu(self.fc2(h1))
        h3 = F.relu(self.fc3(h2))
        return self.mu(h3)

    def decode(self, z, c):
        s = torch.cat((z, c), dim=1)
        h4 = F.relu(self.fc4(s))
        h5 = F.relu(self.fc5(h4))
        h6 = F.relu(self.fc6(h5))
        return self.out(h6)

    def sample(self, z, c, deterministic=False):
        if not deterministic:
            dist = torch.distributions.Categorical(z.softmax(dim=1))
            indices = dist.sample()
        else:
            indices = z.argmax(dim=1)
        z = F.embedding(indices, self.quantizer.embed.transpose(0, 1))
        s = torch.cat((z, c), dim=1)
        h4 = F.relu(self.fc4(s))
        h5 = F.relu(self.fc5(h4))
        h6 = F.relu(self.fc6(h5))
        return self.out(h6)

