from utils.utils import load_args_from_yaml_all
from model_trainer import KADTrainer

if __name__ == '__main__':
    args = load_args_from_yaml_all('./config.yml')
    model_trainer = KADTrainer()