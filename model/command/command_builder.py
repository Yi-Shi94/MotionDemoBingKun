import yaml
import model.command.randomplay_command as randomplay_command
import model.command.randomplay_clip_command as randomplay_clip_command
import model.command.edit_command as edit_command
import model.command.controlnet_command as controlnet_command

def build_command(config_file):
    config = load_yaml_file(config_file)
    command_name = config["test_command_name"]
    print("Building {} command".format(command_name))

    if (command_name == randomplay_command.RandomPlay.NAME):
        command = randomplay_command.RandomPlay(config)
    if (command_name == randomplay_clip_command.RandomPlayCLIP.NAME):
        command = randomplay_clip_command.RandomPlayCLIP(config)
    elif (command_name == edit_command.Edit.NAME):
        command = edit_command.Edit(config)
    elif (command_name == controlnet_command.ControlNet.NAME):
        command = controlnet_command.ControlNet(config)
    else:
        assert(False), "Unsupported command: {}".format(command_name)
        
    return command

def load_yaml_file(file):
    with open(file, "r") as stream:
        config = yaml.safe_load(stream)
    return config
