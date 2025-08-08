import os
import pickle
import re

import numpy as np
from tqdm import tqdm

import config
from utils.EvalUtil import compare_fb
from utils.FileUtil import FileUtil
from utils.PKLUtil import PKLUtil


def load_ground_truth(ground_path):
    files = os.listdir(ground_path)
    boundary_files = [filename for filename in files if re.match(".*\.boundary", filename)]
    assert len(boundary_files) == 1
    with open(os.path.join(ground_path, boundary_files[0]), "rb") as f:
        boundary = pickle.load(f)

    ground_fb = {}
    func_boundary = boundary[0][0]
    public = boundary[0][2]
    private = boundary[0][1]
    fallback = boundary[0][3]
    tag_id_to_pc = boundary[1]

    for public_interface, public_body in public.items():
        if public_interface in func_boundary[0]:
            pub_interface_pc = tag_id_to_pc[public_interface]

            if public_body in func_boundary[1]:
                ground_fb[pub_interface_pc] = set([tag_id_to_pc[tag_id] for tag_id in func_boundary[1][public_body]])

    for private_entry in private:
        if private_entry in func_boundary[2]:
            priv_entry_pc = tag_id_to_pc[private_entry]
            ground_fb[priv_entry_pc] = set([tag_id_to_pc[tag_id] for tag_id in func_boundary[2][private_entry]])

    if fallback in func_boundary[3]:
        fallback_entry_pc = tag_id_to_pc[fallback]
        ground_fb[fallback_entry_pc] = set([tag_id_to_pc[tag_id] for tag_id in func_boundary[3][fallback]])

    return ground_fb, tag_id_to_pc


def load_pred(path, reachable_block_starts, rpc2index):
    lines = [list(map(lambda x: int(x.split('0x')[1], 16), line.split())) for line in FileUtil.read_lines(path)]
    pred = {}
    for line in lines:
        entry = line[1]
        body = line[0]
        if entry == 0:
            continue
        if entry not in pred:
            pred[entry] = set()
        if body in reachable_block_starts:
            pred[entry].add(body)

    for entry in pred:
        tmp = sorted(list(pred[entry]), reverse=True)
        for i in range(len(tmp) - 1):
            if rpc2index[tmp[i]] == rpc2index[tmp[i + 1]] + 1:
                pred[entry].remove(tmp[i])
    for entry in pred:
        if len(pred[entry]) > 1 and entry in pred[entry]:
            pred[entry].remove(entry)
    return pred


split = PKLUtil.load_pkl(config.address_path)
addresses = split[2]

precisions = []
f1s = []
recalls = []

fail_count = 0
for address in tqdm(addresses):
    ground_path = os.path.join(config.neural_FEBI_ground_truth_dir, config.version, "data", address)
    fb_golden, tag_id_to_pc = load_ground_truth(ground_path)

    pred_file_path = os.path.join(config.res_dir, config.mode, config.version, address, "out", "InFunction.csv")
    if not os.path.exists(pred_file_path):
        precisions.append(0)
        recalls.append(0)
        f1s.append(0)
        fail_count+=1
        continue

    reachable_block_starts = sorted(list(tag_id_to_pc.values()))
    rpc2index = {pc: i for i, pc in enumerate(reachable_block_starts)}

    pred_fb = load_pred(pred_file_path, reachable_block_starts, rpc2index)

    f1, precision, recall = compare_fb(fb_golden, pred_fb)

    precisions.append(precision)
    recalls.append(recall)
    f1s.append(f1)


print(np.average(precisions))
print(np.average(recalls))
print(np.average(f1s))

print("fail", fail_count)
