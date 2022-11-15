import os
import argparse
import random
import logging
logging.disable(logging.INFO) 
logging.disable(logging.WARNING)
import numpy as np
import pandas as pd
import torch
import torch.utils.data

from model import get_model
from train import train_model
from utils import get_maxlen_classes

# Location to the data folder
datapath = 'data/'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Default seed for reproducibility
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Parser arguments
parser = argparse.ArgumentParser(description='NILM analysis')
parser.add_argument('--modelname', type=str, default='bert-small',
                    help='Model to run')
parser.add_argument('--tasks', type=str, nargs='+', help='Tasks to train')

args = parser.parse_args()

# We assume the tasks are in the form of a list
tasks = [item for item in args.tasks]
modelname = args.modelname

# Make a folder with the modelname
if(os.path.exists(modelname) == False):
    os.mkdir(modelname)

# Store all the logs to the logs folder
if(os.path.exists('logs') == False):
    os.mkdir('logs')

# We create a seperate results folder inside the model to save results
results_dir = modelname + '/results/'
if(os.path.exists(results_dir) == False):
    os.mkdir(results_dir)

print("##########################################################################")
print("Model chosen: {} ".format(modelname))
print("##########################################################################")
print("Tasks chosen: {} ".format(tasks))
print("##########################################################################")

for task in tasks:
    print("Training task: {} ".format(task))
    directory = task
    
    # Inside results directory we create separate folder for each task
    resultspath = os.path.join(results_dir, directory)
    if(os.path.exists(resultspath) == False):
        os.mkdir(resultspath)

    # Loading data
    train_data = pd.read_csv(datapath + task + '/' + task + '_' + 'train.csv')
    dev_data = pd.read_csv(datapath + task + '/' + task + '_' + 'test.csv')
    test_data = pd.read_csv(datapath + task + '/' + task + '_' + 'test.csv')

    max_sen_len, numclasses = get_maxlen_classes(task)
    tokenizer, base_model = get_model(modelname, numclasses)
    train_model(train_data, dev_data, test_data, task, max_sen_len, resultspath, tokenizer, base_model)

    print("##########################################################################")
