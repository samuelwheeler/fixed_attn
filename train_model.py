import ViT_model
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
import torchvision.transforms as transforms
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import math
import os
import time
from tqdm import tqdm
import pandas as pd
from torch.optim.lr_scheduler import MultiStepLR
from autoaugment import CIFAR10Policy
#import warmup_scheduler


torch.manual_seed(4525)
# set hyperparameters and initial conditions
batch_size = 512
image_size = (32,32)
patch_size = (4,4)
channels = 3
dim = 512
numblocks = 8
hidden_dim = dim
heads = 8
dropout = 0.1
state_path = 'multi_head_q.pth'
epochs = 200
initial_lr = 1e-3
pre_layers = 2
warmup_epoch = 5

load_model = False 
save_model = False



# device 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


model = ViT_model.ViT(image_size = image_size, patch_size = patch_size, num_classes = 10, dim = dim, depth = numblocks, mlp_dim = dim, attention_type = 'vrandom_shuffled', 
            heads = heads, dropout = dropout, emb_dropout = dropout, fixed_size = False, pre_layers = pre_layers)
starting_epoch = 0

model= nn.DataParallel(model)
model = model.to(device)
# optimizer = optim.Adam(model.parameters(), lr = initial_lr, betas=(0.9, 0.99))

optimizer = optim.Adam(model.parameters(), lr = initial_lr, betas=(0.9, 0.99), weight_decay = 5e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max= epochs, eta_min= 1e-6)

if load_model:
    checkpoint = torch.load(state_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    starting_epoch = checkpoint['epoch'] + 1
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    print(f'Loaded model at epoch {starting_epoch}')






"""transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)"""



transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.Resize(image_size),
    transforms.RandomHorizontalFlip(), #CIFAR10Policy(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# print the number of trainable parameters in the model:
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Number of trainable parameters: {num_params}')

start_time = time.time()
#model = model.to(device)    
criterion = nn.CrossEntropyLoss()
#lambda1 = lambda epoch: 0.89**(2*epoch)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

#train_accs = np.zeros(epochs)
#test_accs = np.zeros(epochs)
#learning_rates = np.zeros(epochs)

for epoch in range(200):
    
    lr = optimizer.param_groups[0]["lr"]
    print(f'Learning Rate: {lr}')
    #learning_rates[epoch] = lr
    train_correct = 0
    train_total = 0    
    for batch_idx, (data, target) in enumerate(tqdm(trainloader)):
        if torch.cuda.is_available():
            data, target = data.to(device), target.to(device)
        model.train()
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        _, preds = torch.max(outputs.data, 1)
        train_correct += (preds == target).sum().item()
        train_total += target.size(0)
#         if batch_idx%100 == 0:
#             print(f'Loss: {loss.item()}')
    scheduler.step()
    test_correct = 0
    test_total = 0
        
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            # calculate outputs by running images through the network
            model.eval()
            outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
    train_acc, test_acc = train_correct/train_total, test_correct/test_total
    #train_accs[epoch] = train_acc
    #test_accs[epoch] = test_acc
    '''if epoch >= 2 and False:
        if test_accs[epoch] - test_accs[epoch-1] < 0.01:
            lr = lr * 0.75
            for g in optimizer.param_groups:
                g['lr'] = lr'''
    print(f'Epoch: {epoch + 1 }, Train Acc: {train_acc}, Test Acc: {test_acc}')
total_time = time.time() - start_time
print(total_time)

if save_model:
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            }, state_path)
