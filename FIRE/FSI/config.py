import os
import random

import numpy as np
import torch

from utils.FileUtil import FileUtil

random.seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")

# path
version = "0.5.17-optimized"
neural_FEBI_ground_truth_path = f"/data//func_body/ground-truth/{version}/data/"
data_dir = f"/data//func_body/my-dataset/"
FileUtil.create_dir_if_not_exist(data_dir)
data_path = f"{data_dir}/{version}.pkl"
address_path = f"/data//func_body/dataset/{version}.split"
java_path = "/data//env/jdk-17.0.9/bin/java"
ether_solve_path = "/data//func_body/neural-FEBI/Me/utils/EtherSolve.jar"
ether_solve_result_dir = f"/data//func_body/ether_solve_tmp/{version}/"
FileUtil.create_dir_if_not_exist(ether_solve_result_dir)
model_save_dir = "/data//func_body/checkpoint"
FileUtil.create_dir_if_not_exist(model_save_dir)
model_name = "FIRE"
model_save_path = os.path.join(model_save_dir, f"{model_name}-{version}.pkl")

fsi_result_path = f"/data//func_body/fsi-result/{version}-{model_name}/"


training_ratio = 0.4
val_ratio = 0.1
test_ratio = 0.5
batch_size = 24
training_epochs = 50
lr = 1e-2
embedding_dim = 64
hidden_dim = 128
dropout = 0
beta1 = 0.9
beta2 = 0.999
eps = 1e-8
weight_decay = 1e-4


if not os.path.exists(fsi_result_path):
    os.makedirs(fsi_result_path)