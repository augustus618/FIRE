import os
import random

import numpy as np
import torch

random.seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")

trainning_epoches = 10
min_block_freq = 5
print_step = 100
lr = 0.01
lr_decay = 0.1
batch_size = 64
grad_clip = 5.
workers = 5


instr_emb_dim = 30
instr_rnn_dim = 300
instr_rnn_layers = 1
block_emb_dim = 300
block_rnn_dim = 200
block_rnn_layers = 1

dropout = 0.1

training_ratio = 0.4
val_ratio = 0.1
test_ratio = 0.5

version = "0.4.25-optimized"

contracts_dir = f"/home//func_body/ground-truth/{version}/data/"
data_dir = f"/home//func_body/dataset/{version}/"
temp_dir = ""

dtype = f"neural-FEBI-{version}"

model_output = "/data//func_body/checkpoint"

address_path = f"/home//func_body/dataset/{version}.split"

if not os.path.exists(data_dir):
    os.makedirs(data_dir)
if not os.path.exists(model_output):
    os.makedirs(model_output)