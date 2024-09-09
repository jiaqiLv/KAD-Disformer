import yaml
import argparse

def dict2namespace(config):
    args = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(args, key, new_value)
    return args

def load_args_from_yaml_all(file_path:str):
    """
    Load config parameters using `yaml` files
    """
    with open(file_path,'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return dict2namespace(config=config)