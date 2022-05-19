from avatar3d import *
from train import *
from test import  *
import os
import os.path as osp
import argparse
import numpy as np


def main(**kwargs):
    data_path = kwargs['data_path']
    train_dataset = dict(np.load(osp.join(data_path, 'train.npz')))
    val_dataset = dict(np.load(osp.join(data_path, 'val.npz')))
    data = {'train': train_dataset, 'test': val_dataset}
    image_path = kwargs['image_path']
    if kwargs['mode'] == 'train':
        epoch = kwargs['epoch']
        model = train(data, image_path, epoch=epoch)
    else:
        test(data, image_path, kwargs['model_path'], kwargs['trained_model_path'], './results')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--epoch',type=int, default=1000)
    parser.add_argument('--image_path', type=str)
    parser.add_argument('--mode', choices=['train', 'test'], required=True)
    parser.add_argument('--trained_model_path', type=str)
    parser.add_argument('--save_path', type=str)
    
    args = parser.parse_args()
    #main(args.model_path, args.mode, train_data_path=args.train_data_path, image_path=args.image_path)
    main(**vars(args))
