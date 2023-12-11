import yaml
import model.amdm_model as amdm_model
import model.humor_model as humor_model
import model.mvae_model as mvae_model
import model.amdm_cond_model as amdm_cond_model

def build_model(model_config_file, dataset, device):
    model_config = load_model_file(model_config_file)
    model_name = model_config["model_name"]
    print("Building {} model".format(model_name))

    if (model_name == amdm_model.AMDM.NAME):
        model = amdm_model.AMDM(config=model_config, dataset=dataset,device=device)
    elif (model_name == amdm_cond_model.AMDM.NAME):
        model = amdm_cond_model.AMDM(config=model_config, dataset=dataset,device=device)
    elif (model_name == humor_model.HUMOR.NAME):
        model = humor_model.HUMOR(config=model_config, dataset=dataset, device=device)
    elif (model_name == mvae_model.MVAE.NAME):
        model = mvae_model.MVAE(config=model_config, dataset=dataset, device=device)
    else:
        assert(False), "Unsupported model: {}".format(model_name)
        
    return model

def load_model_file(file):
    with open(file, "r") as stream:
        config = yaml.safe_load(stream)
    return config
