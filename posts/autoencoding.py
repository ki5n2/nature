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


if 'ipykernel_launcher' in sys.argv[0]:
    sys.argv = [sys.argv[0]]

args = parser.parse_args()

enc_hidden_dim = args.enc_hidden_dim.split(',')
dec_hidden_dim = args.dec_hidden_dim.split(',')
args.enc_hidden_dim_list = []
args.dec_hidden_dim_list = []

args.enc_hidden_dim_list.append(args.input_dim)

for i in enc_hidden_dim:
    args.enc_hidden_dim_list.append(int(i))

args.enc_hidden_dim_list

args.dec_hidden_dim_list.append(args.enc_hidden_dim_list[-1])

for i in dec_hidden_dim:
    args.dec_hidden_dim_list.append(int(i))

args.dec_hidden_dim_list.append(args.input_dim)

args.dec_hidden_dim_list

args

class midlayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(midlayer, self).__init__()
        self.fc_layer = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.LeakyReLU()
    def forward(self, x):
        out = self.fc_layer(x)
        out = self.activation(out)
        return out

class Encoder(nn.Module):
    def __init__(self, hidden_dim_list):
        super(Encoder, self).__init__()
        
        layer_list = []
        for i in range(len(hidden_dim_list)-1):
            layer_list.append(midlayer(hidden_dim_list[i], hidden_dim_list[i+1]))
        
        self.fc_layer = nn.Sequential(*layer_list)
        
    def forward(self, x):
        out = self.fc_layer(x)

        return out

class Decoder(nn.Module):
    def __init__(self, hidden_dim_list):
        super().__init__()
        
        layer_list = []
        for i in range(len(hidden_dim_list)-2):
            layer_list.append(midlayer(hidden_dim_list[i], hidden_dim_list[i+1]))
        
        layer_list.append(nn.Sequential(nn.Linear(hidden_dim_list[i+1], hidden_dim_list[i+2]), nn.Sigmoid()))
        self.fc_layer = nn.Sequential(*layer_list)
    
    def forward(self, x):
        out = self.fc_layer(x)

        return out

class Autoencoder(nn.Module):
    def __init__(self, enc_hidden_dim_list, dec_hidden_dim_list):
        super().__init__()

        self.encoder = Encoder(enc_hidden_dim_list)
        self.decoder = Decoder(dec_hidden_dim_list)

    def forward(self, x):
        out = self.encoder(x)
        out = self.decoder(out)

        return out

autoencoder = Autoencoder(args.enc_hidden_dim_list, args.dec_hidden_dim_list)
autoencoder = autoencoder.to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)

def train_autoencoder(autoencoder, criterion, optimizer, num_epochs):
    train_loss_arr = []
    test_loss_arr = []

    best_test_loss = 99999999
    early_stop, early_stop_max = 0., 3.
    for epoch in range(num_epochs):
        autoencoder.train()
        epoch_loss = 0.
        for data in train_loader:
            inputs, _ = data
            inputs = inputs.to(device)
            optimizer.zero_grad()
            outputs = autoencoder(inputs.view(inputs.size(0), -1))
            train_loss = criterion(outputs, inputs.view(inputs.size(0), -1))
            epoch_loss += train_loss.data
            train_loss.backward()
            optimizer.step()

        train_loss_arr.append(epoch_loss / len(train_loader.dataset))
        
        if epoch != 99:
            autoencoder.eval()

            test_loss = 0.

            for data in test_loader:
                inputs, _ = data
                inputs = inputs.to(device)

                outputs = autoencoder(inputs.view(inputs.size(0), -1))
                batch_loss = criterion(outputs, inputs.view(inputs.size(0), -1))
                test_loss += batch_loss.data

            test_loss = test_loss
            test_loss_arr.append(test_loss)

            if best_test_loss > test_loss:
                best_test_loss = test_loss
                early_stop = 0

                print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss.item():.4f}, Test Loss: {batch_loss.item():.4f} *')
            else:
                early_stop += 1
                print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss.item():.4f}, Test Loss: {batch_loss.item():.4f}') 

        if early_stop >= early_stop_max:
            break

def visualize_images(original, reconstructed, n=10):
    plt.figure(figsize=(20, 4))
    for i in range(n):
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(original[i].reshape(28, 28), cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(reconstructed[i].reshape(28, 28), cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

train_autoencoder(autoencoder, criterion, optimizer, num_epochs=args.num_epoch)

data_iter = iter(test_loader)
images, _ = next(data_iter)
reconstructed = autoencoder(images)
visualize_images(images, reconstructed.detach().numpy())





###

parser = argparse.ArgumentParser(description='parser for argparse test')

parser.add_argument('--input_dim', type=int, default=28*28)
parser.add_argument('--enc_hidden_dim', type=str, default='128,32')
parser.add_argument('--dec_hidden_dim', type=str, default='128')
parser.add_argument('--lr_rate', type=float, default=0.001)
parser.add_argument('--num_epoch', type=int, default=10)

if 'ipykernel_launcher' in sys.argv[0]:
    sys.argv = [sys.argv[0]]  

args = parser.parse_args()

enc_hidden_dim = args.enc_hidden_dim.split(',')
dec_hidden_dim = args.dec_hidden_dim.split(',')

args.enc_hidden_dim_list = []
args.dec_hidden_dim_list = []

args.enc_hidden_dim_list.append(args.input_dim)

for i in enc_hidden_dim:
    args.enc_hidden_dim_list.append(int(i))

args.enc_hidden_dim_list

args.dec_hidden_dim_list.append(args.enc_hidden_dim_list[-1])

for i in dec_hidden_dim:
    args.dec_hidden_dim_list.append(int(i))

args.dec_hidden_dim_list.append(args.input_dim)

args

class midlayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(midlayer, self).__init__()
        self.fc_layer = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.LeakyReLU()
   
    def forward(self, x):
        out = self.fc_layer(x)
        out = self.activation(out)
        return out

class Encoder(nn.Module):
    def __init__(self, hidden_dim_list):
        super(Encoder, self).__init__()
       
        layer_list = []
        for i in range(len(hidden_dim_list)-1):
            layer_list.append(midlayer(hidden_dim_list[i], hidden_dim_list[i+1]))
        
        self.fc_layer = nn.Sequential(*layer_list)
        
    def forward(self, x):
        out = self.fc_layer(x)
        return out

class Decoder(nn.Module):
    def __init__(self, hidden_dim_list):
        super().__init__()
        
        layer_list = []
        for i in range(len(hidden_dim_list)-2):
            layer_list.append(midlayer(hidden_dim_list[i], hidden_dim_list[i+1]))
    
        layer_list.append(nn.Sequential(nn.Linear(hidden_dim_list[i+1], hidden_dim_list[i+2])))
        self.fc_layer = nn.Sequential(*layer_list)
    
    def forward(self, x):
        out = self.fc_layer(x)
        return out

class Autoencoder(nn.Module):
    def __init__(self, enc_hidden_dim_list, dec_hidden_dim_list):
        super().__init__()
        self.encoder = Encoder(enc_hidden_dim_list)
        self.decoder = Decoder(dec_hidden_dim_list)

    def forward(self, x):
        out = self.encoder(x)
        out = self.decoder(out)
        return out

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

autoencoder = Autoencoder(args.enc_hidden_dim_list, args.dec_hidden_dim_list)
autoencoder = autoencoder.to(device)

criterion = nn.MSELoss() 
optimizer = optim.Adam(autoencoder.parameters(), lr=args.lr_rate)

def train_autoencoder(autoencoder, criterion, optimizer, num_epochs):
    train_loss_arr = []
    test_loss_arr = []

    best_test_loss = 99999999
    early_stop, early_stop_max = 0., 3.
    for epoch in range(num_epochs):
        autoencoder.train()
        epoch_loss = 0.
        for data in train_loader:
            inputs, _ = data
            inputs = inputs.to(device)
            optimizer.zero_grad()
            outputs = autoencoder(inputs.view(inputs.size(0), -1))
            train_loss = criterion(outputs, inputs.view(inputs.size(0), -1))
            epoch_loss += train_loss.data
            train_loss.backward()
            optimizer.step()

        train_loss_arr.append(epoch_loss / len(train_loader.dataset))
        
        if epoch != -1:
            autoencoder.eval()

            test_loss = 0.

            for data in test_loader:
                inputs, _ = data
                inputs = inputs.to(device)

                outputs = autoencoder(inputs.view(inputs.size(0), -1))
                batch_loss = criterion(outputs, inputs.view(inputs.size(0), -1))
                test_loss += batch_loss.data

            test_loss = test_loss
            test_loss_arr.append(test_loss)

            if best_test_loss > test_loss:
                best_test_loss = test_loss
                early_stop = 0

                print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss.item():.4f}, Test Loss: {batch_loss.item():.4f} *')
            else:
                early_stop += 1
                print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss.item():.4f}, Test Loss: {batch_loss.item():.4f}') 

        if early_stop >= early_stop_max:
            break

train_autoencoder(autoencoder, criterion, optimizer, num_epochs=args.num_epoch)

data_iter = iter(test_loader)
images, _ = next(data_iter)
reconstructed = autoencoder(images)
visualize_images(images, reconstructed.detach().numpy())



# input_dim도 줄여보자! 

parser = argparse.ArgumentParser(description='parser for argparse test')

parser.add_argument('--enc_hidden_dim', type=str, default='784,128,32')
parser.add_argument('--dec_hidden_dim', type=str, default='128,784')
parser.add_argument('--lr_rate', type=float, default=0.001)
parser.add_argument('--num_epoch', type=int, default=10)

if 'ipykernel_launcher' in sys.argv[0]:
    sys.argv = [sys.argv[0]]  

args = parser.parse_args()

enc_hidden_dim = args.enc_hidden_dim.split(',')
dec_hidden_dim = args.dec_hidden_dim.split(',')

args.enc_hidden_dim_list = []
args.dec_hidden_dim_list = []

for i in enc_hidden_dim:
    args.enc_hidden_dim_list.append(int(i))

args.enc_hidden_dim_list

args.dec_hidden_dim_list.append(args.enc_hidden_dim_list[-1])

for i in dec_hidden_dim:
    args.dec_hidden_dim_list.append(int(i))

args.dec_hidden_dim_list

args

class midlayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(midlayer, self).__init__()
        self.fc_layer = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.LeakyReLU()
   
    def forward(self, x):
        out = self.fc_layer(x)
        out = self.activation(out)
        return out

class Encoder(nn.Module):
    def __init__(self, hidden_dim_list):
        super(Encoder, self).__init__()
       
        layer_list = []
        for i in range(len(hidden_dim_list)-1):
            layer_list.append(midlayer(hidden_dim_list[i], hidden_dim_list[i+1]))
        
        self.fc_layer = nn.Sequential(*layer_list)
        
    def forward(self, x):
        out = self.fc_layer(x)
        return out

class Decoder(nn.Module):
    def __init__(self, hidden_dim_list):
        super().__init__()
        
        layer_list = []
        for i in range(len(hidden_dim_list)-2):
            layer_list.append(midlayer(hidden_dim_list[i], hidden_dim_list[i+1]))
    
        layer_list.append(nn.Sequential(nn.Linear(hidden_dim_list[i+1], hidden_dim_list[i+2])))
        self.fc_layer = nn.Sequential(*layer_list)
    
    def forward(self, x):
        out = self.fc_layer(x)
        return out

class Autoencoder(nn.Module):
    def __init__(self, enc_hidden_dim_list, dec_hidden_dim_list):
        super().__init__()
        self.encoder = Encoder(enc_hidden_dim_list)
        self.decoder = Decoder(dec_hidden_dim_list)

    def forward(self, x):
        out = self.encoder(x)
        out = self.decoder(out)
        return out

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

autoencoder = Autoencoder(args.enc_hidden_dim_list, args.dec_hidden_dim_list)
autoencoder = autoencoder.to(device)

criterion = nn.MSELoss() 
optimizer = optim.Adam(autoencoder.parameters(), lr=args.lr_rate)

def train_autoencoder(autoencoder, criterion, optimizer, num_epochs):
    train_loss_arr = []
    test_loss_arr = []

    best_test_loss = 99999999
    early_stop, early_stop_max = 0., 3.
    for epoch in range(num_epochs):
        autoencoder.train()
        epoch_loss = 0.
        for data in train_loader:
            inputs, _ = data
            inputs = inputs.to(device)
            optimizer.zero_grad()
            outputs = autoencoder(inputs.view(inputs.size(0), -1))
            train_loss = criterion(outputs, inputs.view(inputs.size(0), -1))
            epoch_loss += train_loss.data
            train_loss.backward()
            optimizer.step()

        train_loss_arr.append(epoch_loss / len(train_loader.dataset))
        
        if epoch != -1:
            autoencoder.eval()

            test_loss = 0.

            for data in test_loader:
                inputs, _ = data
                inputs = inputs.to(device)

                outputs = autoencoder(inputs.view(inputs.size(0), -1))
                batch_loss = criterion(outputs, inputs.view(inputs.size(0), -1))
                test_loss += batch_loss.data

            test_loss = test_loss
            test_loss_arr.append(test_loss)

            if best_test_loss > test_loss:
                best_test_loss = test_loss
                early_stop = 0

                print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss.item():.4f}, Test Loss: {batch_loss.item():.4f} *')
            else:
                early_stop += 1
                print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss.item():.4f}, Test Loss: {batch_loss.item():.4f}') 

        if early_stop >= early_stop_max:
            break

train_autoencoder(autoencoder, criterion, optimizer, num_epochs=args.num_epoch)

def visualize_images(original, reconstructed, n=10):
    plt.figure(figsize=(20, 4))
    for i in range(n):
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(original[i].reshape(28, 28), cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(reconstructed[i].reshape(28, 28), cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

data_iter = iter(test_loader)
images, _ = next(data_iter)
reconstructed = autoencoder(images)
visualize_images(images, reconstructed.detach().numpy())



# dec_hidden_dim도 줄여보자! 첫 번째 방법

parser = argparse.ArgumentParser(description='parser for argparse test')

parser.add_argument('--enc_hidden_dim', type=str, default='784,128,32')
parser.add_argument('--lr_rate', type=float, default=0.001)
parser.add_argument('--num_epoch', type=int, default=10)

if 'ipykernel_launcher' in sys.argv[0]:
    sys.argv = [sys.argv[0]]  

args = parser.parse_args()

enc_hidden_dim = args.enc_hidden_dim.split(',') ### key point
dec_hidden_dim = args.enc_hidden_dim.split(',')

args.enc_hidden_dim_list = []
args.dec_hidden_dim_list = []

for i in enc_hidden_dim:
    args.enc_hidden_dim_list.append(int(i))

args.enc_hidden_dim_list

for i in enc_hidden_dim[::-1]:
    args.dec_hidden_dim_list.append(int(i))

args.dec_hidden_dim_list

args

class midlayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(midlayer, self).__init__()
        self.fc_layer = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.LeakyReLU()
   
    def forward(self, x):
        out = self.fc_layer(x)
        out = self.activation(out)
        return out

class Encoder(nn.Module):
    def __init__(self, hidden_dim_list):
        super(Encoder, self).__init__()
       
        layer_list = []
        for i in range(len(hidden_dim_list)-1):
            layer_list.append(midlayer(hidden_dim_list[i], hidden_dim_list[i+1]))
        
        self.fc_layer = nn.Sequential(*layer_list)
        
    def forward(self, x):
        out = self.fc_layer(x)
        return out

class Decoder(nn.Module):
    def __init__(self, hidden_dim_list):
        super().__init__()
        
        layer_list = []
        for i in range(len(hidden_dim_list)-2):
            layer_list.append(midlayer(hidden_dim_list[i], hidden_dim_list[i+1]))
    
        layer_list.append(nn.Sequential(nn.Linear(hidden_dim_list[i+1], hidden_dim_list[i+2])))
        self.fc_layer = nn.Sequential(*layer_list)
    
    def forward(self, x):
        out = self.fc_layer(x)
        return out

class Autoencoder(nn.Module):
    def __init__(self, enc_hidden_dim_list, dec_hidden_dim_list):
        super().__init__()
        self.encoder = Encoder(enc_hidden_dim_list)
        self.decoder = Decoder(dec_hidden_dim_list)

    def forward(self, x):
        out = self.encoder(x)
        out = self.decoder(out)
        return out

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

autoencoder = Autoencoder(args.enc_hidden_dim_list, args.dec_hidden_dim_list)
autoencoder = autoencoder.to(device)

criterion = nn.MSELoss() 
optimizer = optim.Adam(autoencoder.parameters(), lr=args.lr_rate)

def train_autoencoder(autoencoder, criterion, optimizer, num_epochs):
    train_loss_arr = []
    test_loss_arr = []

    best_test_loss = 99999999
    early_stop, early_stop_max = 0., 3.
    for epoch in range(num_epochs):
        autoencoder.train()
        epoch_loss = 0.
        for data in train_loader:
            inputs, _ = data
            inputs = inputs.to(device)
            optimizer.zero_grad()
            outputs = autoencoder(inputs.view(inputs.size(0), -1))
            train_loss = criterion(outputs, inputs.view(inputs.size(0), -1))
            epoch_loss += train_loss.data
            train_loss.backward()
            optimizer.step()

        train_loss_arr.append(epoch_loss / len(train_loader.dataset))
        
        if epoch != -1:
            autoencoder.eval()

            test_loss = 0.

            for data in test_loader:
                inputs, _ = data
                inputs = inputs.to(device)

                outputs = autoencoder(inputs.view(inputs.size(0), -1))
                batch_loss = criterion(outputs, inputs.view(inputs.size(0), -1))
                test_loss += batch_loss.data

            test_loss = test_loss
            test_loss_arr.append(test_loss)

            if best_test_loss > test_loss:
                best_test_loss = test_loss
                early_stop = 0

                print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss.item():.4f}, Test Loss: {batch_loss.item():.4f} *')
            else:
                early_stop += 1
                print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss.item():.4f}, Test Loss: {batch_loss.item():.4f}') 

        if early_stop >= early_stop_max:
            break

train_autoencoder(autoencoder, criterion, optimizer, num_epochs=args.num_epoch)

data_iter = iter(test_loader)
images, _ = next(data_iter)
reconstructed = autoencoder(images)
visualize_images(images, reconstructed.detach().numpy())




# dec_hidden_dim도 줄여보자! 두 번째 방법

parser = argparse.ArgumentParser(description='parser for argparse test')

parser.add_argument('--enc_hidden_dim', type=str, default='784,256,128,64,32')
parser.add_argument('--lr_rate', type=float, default=0.001)
parser.add_argument('--num_epoch', type=int, default=10)

if 'ipykernel_launcher' in sys.argv[0]:
    sys.argv = [sys.argv[0]]  

args = parser.parse_args()

enc_hidden_dim = args.enc_hidden_dim.split(',') ### key point(dec_hidden_dim을 따로 정의하지 않았음)

args.enc_hidden_dim_list = []

for i in enc_hidden_dim:
    args.enc_hidden_dim_list.append(int(i))

args.enc_hidden_dim_list

args.enc_hidden_dim_list[::-1]

args

class midlayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(midlayer, self).__init__()
        self.fc_layer = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.LeakyReLU()
   
    def forward(self, x):
        out = self.fc_layer(x)
        out = self.activation(out)
        return out

class Encoder(nn.Module):
    def __init__(self, hidden_dim_list):
        super(Encoder, self).__init__()
       
        layer_list = []
        for i in range(len(hidden_dim_list)-1):
            layer_list.append(midlayer(hidden_dim_list[i], hidden_dim_list[i+1]))
        
        self.fc_layer = nn.Sequential(*layer_list)
        
    def forward(self, x):
        out = self.fc_layer(x)
        return out

class Decoder(nn.Module):
    def __init__(self, hidden_dim_list):
        super().__init__()
        
        layer_list = []
        for i in range(len(hidden_dim_list)-2):
            layer_list.append(midlayer(hidden_dim_list[i], hidden_dim_list[i+1]))
    
        layer_list.append(nn.Sequential(nn.Linear(hidden_dim_list[i+1], hidden_dim_list[i+2])))
        self.fc_layer = nn.Sequential(*layer_list)
    
    def forward(self, x):
        out = self.fc_layer(x)
        return out

class Autoencoder(nn.Module):
    def __init__(self, enc_hidden_dim_list, dec_hidden_dim_list):
        super().__init__()
        self.encoder = Encoder(enc_hidden_dim_list)
        self.decoder = Decoder(dec_hidden_dim_list)

    def forward(self, x):
        out = self.encoder(x)
        out = self.decoder(out)
        return out

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

autoencoder = Autoencoder(args.enc_hidden_dim_list, args.enc_hidden_dim_list[::-1]) ### key point(dec_hidden_dim_list를 enc_hiffen_dim_list[::-1]을 이용해서 정의함)
autoencoder = autoencoder.to(device)

criterion = nn.MSELoss() 
optimizer = optim.Adam(autoencoder.parameters(), lr=args.lr_rate)

def train_autoencoder(autoencoder, criterion, optimizer, num_epochs):
    train_loss_arr = []
    test_loss_arr = []

    best_test_loss = 99999999
    early_stop, early_stop_max = 0., 3.
    for epoch in range(num_epochs):
        autoencoder.train()
        epoch_loss = 0.
        for data in train_loader:
            inputs, _ = data
            inputs = inputs.to(device)
            optimizer.zero_grad()
            outputs = autoencoder(inputs.view(inputs.size(0), -1))
            train_loss = criterion(outputs, inputs.view(inputs.size(0), -1))
            epoch_loss += train_loss.data
            train_loss.backward()
            optimizer.step()

        train_loss_arr.append(epoch_loss / len(train_loader.dataset))
        
        if epoch != -1:
            autoencoder.eval()

            test_loss = 0.

            for data in test_loader:
                inputs, _ = data
                inputs = inputs.to(device)

                outputs = autoencoder(inputs.view(inputs.size(0), -1))
                batch_loss = criterion(outputs, inputs.view(inputs.size(0), -1))
                test_loss += batch_loss.data

            test_loss = test_loss
            test_loss_arr.append(test_loss)

            if best_test_loss > test_loss:
                best_test_loss = test_loss
                early_stop = 0

                print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss.item():.4f}, Test Loss: {batch_loss.item():.4f} *')
            else:
                early_stop += 1
                print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss.item():.4f}, Test Loss: {batch_loss.item():.4f}') 

        if early_stop >= early_stop_max:
            break

train_autoencoder(autoencoder, criterion, optimizer, num_epochs=args.num_epoch)

data_iter = iter(test_loader)
images, _ = next(data_iter)
reconstructed = autoencoder(images)
visualize_images(images, reconstructed.detach().numpy())