from sklearn.preprocessing import minmax_scale
from torch import optim
import torch
import torch.nn as nn
from typing import Union,List
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm
import copy
import numpy as np
from utils.evaluate import f1_score_with_point_adjust,f1_score_point

class KADTrainer():

    def get_model_params(self,model):
        return copy.deepcopy(model.cpu().state_dict())
    
    def set_model_params(self,model,params):
        model.load_state_dict(copy.deepcopy(params))

    def loopy(dl):
        while True:
            for x in iter(dl):
                yield x

    def get_params_to_update(model):
        meta_params = []
        for name,params in model.named_parameters():
            if "Wqm" in name or "Wkm" in name or "Wvm" in name:
                meta_params.append(params)
        return meta_params
    
    # def prtrain_step(self,model,train_data,criterion,optimizer,args):
    #     for epoch in tqdm(range(args.training.epochs)):
    #         for X,y in train_data:
    #             print(X.shape,y.shape)
    #             X = X.to(args.training.device).view(-1,args.training.seq_len,args.model.d_model)
    #             # y = y.to(args.training.device).view(-1,args.training.seq_len,args.model.d_model)
    #             model.zero_grad()
    #             output = model(X)
    #             print(output.shape)
    #             loss = criterion(output,y)
    #             optimizer.zero_grad()
    #             loss.backward()
    #             optimizer.step()

    #         if epoch%10 == 0:
    #             print(f'Epoch: {epoch}, Loss: {loss}')
    #     return model
    
    def prtrain_step(self,model,train_data,criterion,optimizer,args):
        for epoch in tqdm(range(args.training.epochs)):
            for X_context,X_history,X_denoised in train_data:
                # print(X_context.shape,X_history.shape,X_denoised.shape)
                X_context = X_context.to(args.training.device).view(-1,args.training.seq_len,args.model.d_model)
                X_history = X_history.to(args.training.device).view(-1,args.training.seq_len,args.model.d_model)
                model.zero_grad()
                output = model(X_context)
                loss = criterion(output,X_history)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if epoch%10 == 0:
                print(f'Epoch: {epoch}, Loss: {loss}')
        return model

    def pretrain(self,model:nn.Module,train_data:Union[List[DataLoader],DataLoader],args):
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(),lr=args.training.lr)
        if isinstance(train_data,DataLoader):
            model = self.prtrain_step(model,train_data,criterion,optimizer,args)

        elif isinstance(train_data,List[DataLoader]):
            for _train_data in train_data:
                model = self.prtrain_step(model,_train_data,criterion,optimizer,args)
        return model

    def fine_tune_step(self,model,train_data,tuning_data,criterion,optimizer,args):
        for epoch in tqdm(range(args.training.epochs)):
            for X,y in tuning_data:
                X = X.to(args.training.device).view(-1,args.training.seq_len,args.model.d_model)
                y = y.to(args.training.device).view(-1,args.training.seq_len,args.model.d_model)

                train_X, train_y = self.loopy(train_data)
                train_X = train_X.to(args.training.device)
                train_y = train_y.to(args.training.device)

                model.zero_grad()
                optimizer.zero_grad()

                tuning_output = model(X)
                tuning_loss = criterion(tuning_output,y)
                tuning_loss.backward(retain_graph=True)
                optimizer.step()

                train_output = model(train_X)
                train_loss = criterion(train_output,train_y)
                loss = args.training.alpha * tuning_loss + (1-args.training.alpha) * train_loss
                loss.backward()
                optimizer.step()

            if epoch%10 == 0:
                print(f'Epoch: {epoch}, Loss: {loss}')
        return model
                

    def fine_tune(self,model:nn.Module,train_data,tuning_data,args):
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(),lr=args.training.lr)
        model = self.fine_tune_step(model,train_data,tuning_data,criterion,optimizer,args)
        return model

    def test(self,model:nn.Module,test_data:Union[DataLoader,List[DataLoader]],args):
        reconstruct_seq = []
        labels = []
        raw_series = []
        with torch.no_grad():
            for X_context,X_history,X_denoised,y in test_data:
                print(X_context.shape,X_history.shape,X_denoised.shape,y.shape)
                X_context = X_context.to(args.training.device).view(-1,args.training.seq_len,args.model.d_model)
                output = model(X_context)
                raw_series.append(X_context.to('cpu').numpy()[:,-1,-1])
                reconstruct_seq.append(output.to('cpu').numpy()[:,-1,-1])
                labels.append(y.to('cpu').numpy())
        raw_series = np.concatenate(raw_series)
        reconstruct_seq = np.concatenate(reconstruct_seq)
        labels = np.concatenate(labels)
        print(raw_series.shape,reconstruct_seq.shape, labels.shape)
        y_scores = minmax_scale(np.abs(reconstruct_seq-raw_series))
        # BUG: verify functional consistency
        f1_range = f1_score_with_point_adjust(labels, y_scores)
        f1_point = f1_score_point(labels, y_scores)
        print(f1_range)
        print(f1_point)
        return f1_range,f1_point
