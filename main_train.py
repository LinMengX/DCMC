import argparse
import csv
import datetime
import math
import os
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils import data

from model import DCMC
import utils
from engine_train import train_one_epoch
from dataset_loader import load_dataset, IncompleteDatasetSampler

warnings.filterwarnings("ignore")


def get_args_parser():
    parser = argparse.ArgumentParser(description='Training')

    # config path
    parser.add_argument('--config_file', type=str, default='config/brca.yaml')

    # backbone parameters
    parser.add_argument('--encoder_dim', type=list, nargs='+', default=[])
    parser.add_argument('--embed_dim', type=int, default=0)

    # model parameters
    parser.add_argument('--temperature', type=float, default=0.5)
    parser.add_argument('--start_rectify_epoch', type=int, default=100)
    parser.add_argument('--momentum', type=float, default=0.99)
    parser.add_argument('--drop_rate', type=float, default=0.2)
    parser.add_argument('--n_views', type=int, default=3, help='number of views')
    parser.add_argument('--n_classes', type=int, default=10, help='number of classes')

    # training setting
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch size per GPU')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--warmup_epochs', type=int, default=20, help='epochs to warmup learning rate')
    parser.add_argument('--data_norm', type=str, default='standard', choices=['standard', 'min-max', 'l2-norm'])
    parser.add_argument('--train_time', type=int, default=5)

    # optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='Initial value of the weight decay. (default: 0)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')

    # data loader and logger
    parser.add_argument('--dataset', type=str, default='aml',
                        choices=['breast', 'colon','gbm','kidney','liver','lung','melanoma','ovarian','sarcoma' ])
    parser.add_argument('--missing_rate', type=float, default=0.0)
    parser.add_argument('--data_path', type=str, default='./',
                        help='path to your folder of dataset')
    parser.add_argument('--survival_data_path',type = str,default='./',
                        help='path to your folder of survival data')
    parser.add_argument('--clinical_data_path', type=str, default='./',
                        help='path to your folder of clinical data')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--output_dir', type=str, default='./',
                        help='path where to save, empty for no saving')

    parser.add_argument('--print_freq', default=50)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--seed', default=None, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    return parser


def train_one_time(args, state_logger):
    utils.fix_random_seeds(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    args.cuda = torch.cuda.is_available()
    print("use cuda: {}".format(args.cuda))
    device = torch.device(args.device)

    dataset = load_dataset(args)
    dataset_train, dataset_test = dataset, dataset

    sampler_train = IncompleteDatasetSampler(dataset_train, seed=args.seed)
    sampler_test = torch.utils.data.RandomSampler(dataset_test)

    if args.batch_size > len(sampler_train):
        args.batch_size = len(sampler_train)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, sampler=sampler_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    model = DCMC(n_views=args.n_views,
                   layer_dims=args.encoder_dim,
                   temperature=args.temperature,
                   n_classes=args.n_classes,
                   drop_rate=args.drop_rate, )

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99))

    if args.train_id == 0:
        print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
        state_logger.write('Batch size: {}'.format(args.batch_size))
        state_logger.write('Start time: {}'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M")))
        state_logger.write('Train parameters: {}'.format(args).replace(', ', ',\n'))
        state_logger.write(model.__repr__())
        state_logger.write(optimizer.__repr__())
        print('Data loaded: there are {:} samples.'.format(len(dataset_train)))

    state_logger.write('\n>> Start training {}-th initial, seed: {},'.format(args.train_id, args.seed))

    max_log10p = None
    max_cnt = None
    max_y_pred = None
    for epoch in range(args.start_epoch, args.epochs):

        train_state = train_one_epoch(
            model, data_loader_train, data_loader_test,
            optimizer,
            device, epoch,
            state_logger,
            args
        )

        if args.output_dir and epoch + 1 == args.epochs:
            torch.save(model, args.output_dir + f"checkpoint_{epoch}")


        if max_log10p is None or train_state['log10p'] > max_log10p:
            max_log10p = train_state['log10p']
            max_y_pred = train_state['y_pred']

        if max_cnt is None or train_state['cnt'] > max_cnt:
            max_cnt = train_state['cnt']
        if epoch + 1 == args.epochs:
            train_state['log10p'] = max_log10p
            train_state['cnt'] = max_cnt
            train_state['y_pred'] = max_y_pred
    return train_state

def main(args):
    start_time = time.time()

    result_avr = {'log10p': [], 'cnt': [] ,'y_pred': []}
    max_results = {'max_log10p': None, 'max_cnt': None ,'y_pred' : None}
    batch_scale = args.batch_size / 256
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * batch_scale

    state_logger = utils.FileLogger(os.path.join(args.output_dir, 'log_train.txt'))

    for t in range(args.train_time):
        args.train_id = t
        train_state = train_one_time(args, state_logger)
        args.seed = args.seed + 1
        args.seed = (args.seed + datetime.datetime.now().microsecond) % 999

        for k, v in train_state.items():
            result_avr[k].append(v)
            if k == 'log10p':
                if max_results['max_log10p'] is None or v > max_results['max_log10p']:
                    max_results['max_log10p'] = v
            elif k == 'cnt':
                if max_results['max_cnt'] is None or v > max_results['max_cnt']:
                    max_results['max_cnt'] = v
            elif k == 'y_pred':
                if max_results['max_log10p'] == result_avr['log10p'][-1]:
                    max_results['y_pred'] = v

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    state_logger.write('\nTraining time {}\n'.format(total_time_str))
    state_logger.write('Result: log10p = {:.4f} cnt = {:} y_pred = {:}'
                       .format(max_results['max_log10p'], max_results['max_cnt'], max_results['y_pred']))

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()

    if args.config_file is not None:
        with open(args.config_file) as f:
            if hasattr(yaml, 'FullLoader'):
                configs = yaml.load(f.read(), Loader=yaml.FullLoader)
            else:
                configs = yaml.load(f.read())

        args = vars(args)
        args.update(configs)
        args = argparse.Namespace(**args)

    folder_name = '_'.join(
        [args.dataset, 'msrt', str(args.missing_rate),
         'tau', str(args.temperature), 'bs', str(args.batch_size), 'blr', str(args.blr)])

    args.embed_dim = args.encoder_dim[0][-1]
    args.output_dir = os.path.join(args.output_dir, folder_name)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(args.output_dir, 'visualize')).mkdir(parents=True, exist_ok=True)

    main(args)
