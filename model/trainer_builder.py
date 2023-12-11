import yaml

import model.amdm_model as amdm_model
import model.amdm_cond_model as amdm_cond_model
import model.amdm_trainer as amdm_trainer
import model.amdm_cond_trainer as amdm_cond_trainer

import model.humor_model as humor_model
import model.humor_trainer as humor_trainer

import model.mvae_model as mvae_model
import model.mvae_trainer as mvae_trainer

import dataset.dataset_builder as dataset_builder

def build_trainer(config_file, device):
    model_config = load_config_file(config_file)
    model_name = model_config["model_name"]
    dataset = dataset_builder.build_dataset(model_config, device=device)

    print("Building {} trainer".format(model_name))
    if (model_name == amdm_model.AMDM.NAME):
        trainer = amdm_trainer.AMDMTrainer(config=model_config, dataset=dataset, device=device)
    elif (model_name == amdm_cond_model.AMDM.NAME):
        trainer = amdm_cond_trainer.AMDMCONDTrainer(config=model_config, dataset=dataset, device=device)
    elif (model_name == humor_model.HUMOR.NAME):
        trainer = humor_trainer.HUMORTrainer(config=model_config, dataset=dataset, device=device)
    elif (model_name == mvae_model.MVAE.NAME):
        trainer = mvae_trainer.MVAETrainer(config=model_config, dataset=dataset, device=device)
    else:
        assert(False), "Unsupported trainer: {}".format(model_name)
    return trainer

def load_config_file(file):
    with open(file, "r") as stream:
        config = yaml.safe_load(stream)
    return config

def get_feature_dim_dict(dataset):
    return {"frame_dim": dataset.frame_dim}