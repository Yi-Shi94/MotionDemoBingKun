import dataset.amass_dataset as amass_dataset
import dataset.lafan1_dataset as lafan1_dataset
import yaml

def build_dataset(config_file, device):
    config = load_config_file(config_file)
    dataset_name = config["data"]["dataset_name"]
    dataset_class_name = config["data"].get("dataset_class_name", dataset_name)
    print("Loading {} dataset class".format(dataset_class_name))
    print("Loading {} dataset".format(dataset_name))

    if (dataset_class_name == amass_dataset.AMASS.NAME):
        dataset = amass_dataset.AMASS(config)
    elif (dataset_class_name == lafan1_dataset.LAFAN1.NAME):
        dataset = lafan1_dataset.LAFAN1(config)
    else:
        assert(False), "Unsupported dataset class: {}".format(dataset_class_name)
    return dataset

def load_config_file(file):
    with open(file, "r") as stream:
        config = yaml.safe_load(stream)
    return config