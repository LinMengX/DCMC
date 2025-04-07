import math
import os
import re
import sys
from typing import Iterable

import numpy as np
import pandas as pd
import torch

from sklearn.cluster import KMeans
from sklearn.cluster import spectral_clustering
import utils
from model import L2norm
from utils import adjust_learning_config, SmoothedValue, MetricLogger
import evaluate_p

def train_one_epoch(model: torch.nn.Module,
                    data_loader_train: Iterable, data_loader_test: Iterable,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int,
                    state_logger=None,
                    args=None):
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 50
    data_loader = enumerate(metric_logger.log_every(data_loader_train, print_freq, header))
    model.train(True)
    optimizer.zero_grad()

    for data_iter_step, (ids, samples, mask) in data_loader:
        smooth_epoch = epoch + (data_iter_step + 1) / len(data_loader_train)
        lr = adjust_learning_config(optimizer, smooth_epoch, args)
        mmt = args.momentum

        for i in range(args.n_views):
            samples[i] = samples[i].to(device, non_blocking=True)

        with torch.autocast('cuda', enabled=False):
            loss = model(samples, mmt, epoch < args.start_rectify_epoch)

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # if args.print_this_epoch:
        metric_logger.update(lr=lr)
        metric_logger.update(loss=loss_value)

    eval_result = evaluate(model, data_loader_test, device, epoch, args)
    return eval_result


def evaluate(model: torch.nn.Module, data_loader_test: Iterable,
             device: torch.device, epoch: int,
             args=None):
    model.eval()
    extracter = model.extract_feature
    
    with torch.no_grad():
        features_all = torch.zeros(args.n_views, args.n_sample, args.embed_dim).to(device)
        labels_all = torch.zeros(args.n_sample, dtype=torch.long).to(device)
        for indexs, samples, mask,in data_loader_test:
            for i in range(args.n_views):
                samples[i] = samples[i].to(device, non_blocking=True)
                
            features = extracter(samples, mask)
            
            for i in range(args.n_views):
                features_all[i][indexs] = features[i]
               
        features_cat = features_all.permute(1, 0, 2).reshape(args.n_sample, -1)
        features_cat = torch.nn.functional.normalize(features_cat, dim=-1).cpu().numpy()
       
        kmeans_label = KMeans(n_clusters=args.n_classes, random_state=0).fit_predict(features_cat)
        
        y_pred = kmeans_label
        
        survival = pd.read_csv(os.path.join(args.survival_data_path, args.dataset + ".survival"), sep="\t")
        survival = survival.dropna(axis=0)
        
        name_list = list()
      
        survival["PatientID"] = [re.sub("-", ".", x) for x in survival["PatientID"].str.upper()]

        if len(survival["PatientID"][survival.index[0]]) <= len('tcga.16.1060'):
            survival["PatientID"] += ".01"
            
            for token in survival["PatientID"]:
                
                if token[-2] != "0":
                    survival.drop(survival[survival["PatientID"] == token].index, inplace=True)
                    continue
                name_list.append(token)
        survival["label"] = np.array(y_pred)
        df = survival
        res = evaluate_p.log_rank(df)
    clinical_data = evaluate_p.get_clinical(args.clinical_data_path , survival, args.dataset)
    #clinical_data = evaluate_p.get_clinical(args.clinical_data_path , survival, args.dataset)
    cnt = evaluate_p.clinical_enrichement(clinical_data['label'], clinical_data)
    result = {'log10p':res['log10p'],'cnt':cnt ,'y_pred': y_pred}
    return result
