# Deep autoencoding gaussian mixture model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchinfo import summary
from mpl_toolkits.mplot3d import Axes3D

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

data_root = './data'

batch_size = 32

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5),
    transforms.Lambda(lambda x: x.view(-1)),
])

trainset = datasets.MNIST(
    root        = data_root, 
    train       = True, 
    download    = True,
    transform   = transform
)

testset = datasets.MNIST(
    root        = data_root, 
    train       = False, 
    download    = False,
    transform   = transform
)

train_loader = torch.utils.data.DataLoader(
    dataset     = trainset,
    batch_size  = batch_size,
    shuffle     = True
)

test_loader = torch.utils.data.DataLoader(
    dataset     = testset,
    batch_size  = batch_size,
    shuffle     = False
)

parser = argparse.ArgumentParser(description='parser for argparse test')

parser.add_argument('--input_dim', type=int, default=28*28)
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--num_epoch', type=int, default=10)
parser.add_argument('--enc_hidden_dim', type=str, default='256,128,64,32,3')
parser.add_argument('--dec_hidden_dim', type=str, default='32,64,128,256')
