import torch
from models.discriminator import discriminator
from models.generator import generator
import torch.nn as nn
import os

lr = 2e-4
l1_lambda = 100
num_of_epochs = 500
OptimizerD = torch.optim.Adam(discriminator.parameters(),lr=lr)
OptimizerG = torch.optim.Adam(generator.parameters(),lr=lr)
BCE_Loss = nn.BCEWithLogitsLoss()
L1_Loss = nn.L1Loss()

device = "cuda is available" if torch.cuda.is_available() else "we are training on CPU"
if torch.cuda.device_count() > 1:
    generator = nn.DataParallel(generator).to(device)
    discriminator = nn.DataParallel(discriminator).to(device)

def create_folders():
    try:
        os.mkdir("./generated")
        os.mkdir("./label")
        os.mkdir("./input")
    except: print("Folders Already Created")

    print(device)
    if torch.cuda.device_count() > 1:
        print("We are Using", torch.cuda.device_count(), "GPUs!")

    return "Folders Created"