from utils.utils import load_args_from_yaml_all
from model_trainer import KADTrainer
from utils.utils import TRAIN_SET,TEST_SET
import pandas as pd
from sklearn.preprocessing import minmax_scale
from torch.utils.data import DataLoader,ConcatDataset
from os.path import join,exists
from models.KAD_Disformer import KAD_Disformer
from utils.dataset import KAD_DisformerTestSet,KAD_DisformerTrainSet,UTSDataset

def prepare_data(data_path,args=None,pattern='pretrain',shuffle=True):
    data_df = pd.read_csv(data_path)
    raw_series = data_df['value'].to_numpy()
    labels = data_df['label'].to_numpy()
    raw_series = minmax_scale(raw_series)
    if pattern == 'pretrain' or pattern == 'fine_tune':
        dataset = KAD_DisformerTrainSet(raw_series,win_len=20,seq_len=100,labels=labels)
    elif pattern == 'test':
        dataset = KAD_DisformerTestSet(raw_series,win_len=20,seq_len=100,labels=labels)
    else:
        raise ValueError('Unexpected Pattern.')
    dataloader = DataLoader(dataset=dataset,batch_size=256,drop_last=False,shuffle=shuffle)
    return dataloader,dataset

def set_dataset(args,pattern='pretrain',shuffle=True):
    """
    pattern: pretrain, fine_tune, test
    """
    data_dir = args.data.data_path
    if pattern == 'pretrain':
        pretrain_data_list = []
        for train_file in TRAIN_SET:
            _dataloader,_dataset = prepare_data(join(data_dir,train_file),args=args,pattern='pretrain')
            pretrain_data_list.append(_dataset)
        dataset = ConcatDataset(pretrain_data_list)
        dataloader = DataLoader(dataset,batch_size=256,drop_last=True,shuffle=True)
    elif pattern == 'fine_tune':
        dataloader,dataset = prepare_data(join(data_dir,TEST_SET[0]),args=args,pattern='fine_tune')
    elif pattern == 'test':
        pass
    else:
        raise ValueError('Unexpected Pattern.')
    return dataloader,dataset

if __name__ == '__main__':
    args = load_args_from_yaml_all('./config.yml')
    model_trainer = KADTrainer()
    model = KAD_Disformer(win_len=20,heads=5)
    model = model.to(args.training.device)
    """Init DataLoader"""
    pretrain_dataloader,_ = set_dataset(args=args,pattern='pretrain')
    """Pretrain"""
    model_trainer.pretrain(model,pretrain_dataloader,args)