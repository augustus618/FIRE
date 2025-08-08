import os
import re

import numpy as np
from tqdm import tqdm

import config
from utils.EvalUtil import compare_fs
from utils.FileUtil import FileUtil
from utils.PKLUtil import PKLUtil

split = PKLUtil.load_pkl(config.address_path)
addresses = split[2]

precisions = []
f1s = []
recalls = []

fail_count = 0
for address in tqdm(addresses):
    files = os.listdir(os.path.join(config.neural_FEBI_ground_truth_dir, config.version, "data", address))
    boundary_file = [filename for filename in files if re.match(".*\.boundary", filename)][0]
    function_boundaries_path = os.path.join(config.neural_FEBI_ground_truth_dir, config.version, "data", address,
                                            boundary_file)
    boundary, tag_to_pc, _ = PKLUtil.load_pkl(function_boundaries_path)

    public_pc_set = set([tag_to_pc[tag_id] for tag_id, _ in boundary[0][1].items()])
    fallback_pc_set = set([tag_to_pc[tag_id] for tag_id, _ in boundary[0][3].items()])
    priv_pc_set = set([tag_to_pc[tag_id] for tag_id, _ in boundary[0][2].items()])

    pred_file_path = os.path.join(config.res_dir, config.mode, config.version, address, "out", "Function.csv")
    if not os.path.exists(pred_file_path):
        precisions.append(0)
        recalls.append(0)
        f1s.append(0)
        fail_count += 1
        continue
    pub_path = os.path.join(config.res_dir, config.mode, config.version, address, "out", "PublicFunction.csv")
    pubs = {int(pc.split()[0], 16) for pc in FileUtil.read_lines(pub_path)}

    preds = {int(pc, 16) for pc in FileUtil.read_lines(pred_file_path)}
    preds.remove(0)
    preds = preds - pubs - public_pc_set - fallback_pc_set

    f1, precision, recall = compare_fs(priv_pc_set, preds)

    precisions.append(precision)
    recalls.append(recall)
    f1s.append(f1)

print(np.average(precisions))
print(np.average(recalls))
print(np.average(f1s))

print("fail", fail_count)
