import os
import pickle
import re
import time
import warnings
from collections import Counter

import numpy as np
import sklearn
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from torch import optim, nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch_geometric.data import Dataset as PYGDataset, Data as PYGData, DataLoader as PYGDataLoader
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import to_undirected
from tqdm import tqdm

import config
from utils.PKLUtil import PKLUtil


class FIRE(nn.Module):
    def __init__(self, vocab_size):
        super(FIRE, self).__init__()

        self.embedding = nn.Embedding(vocab_size + 2, config.embedding_dim)

        self.lstm = nn.LSTM(input_size=config.embedding_dim, hidden_size=32, num_layers=1, bidirectional=True, batch_first=True,
                            dropout=config.dropout)
        self.gat = GATv2Conv(vocab_size + 64, config.hidden_dim, dropout=config.dropout)
        self.gat_norm = nn.BatchNorm1d(config.hidden_dim)
        self.gat_relu = nn.ReLU()

        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(64, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(16, 2)
        )

    def forward(self, x, edge_index, block_instructions, lengths):
        embedded_sequences = self.embedding(block_instructions)

        packed_input = pack_padded_sequence(embedded_sequences, lengths.tolist(), batch_first=True,
                                            enforce_sorted=False)
        packed_lstm_out, _ = self.lstm(packed_input)
        lstm_out, _ = pad_packed_sequence(packed_lstm_out, batch_first=True)

        mask = torch.arange(lstm_out.size(1), device=config.device).unsqueeze(0) < lengths.unsqueeze(1)
        mask = mask.unsqueeze(2).float()
        masked_sum = (lstm_out * mask).sum(dim=1)
        out = masked_sum / lengths.unsqueeze(1).float()

        features = torch.cat((x, out), dim=-1)
        return self.classifier(self.gat_relu(self.gat_norm(self.gat(features, edge_index))))



class Dataset(PYGDataset):
    def __init__(self, data):
        super().__init__()
        self.graph_list = []
        xs = data[0]
        block_instructions = data[1]
        lengths = data[2]
        edge_indexes = data[3]
        ys = data[4]
        for i in range(len(xs)):
            self.graph_list.append(
                PYGData(x=xs[i], edge_index=edge_indexes[i], y=ys[i], block_instructions=block_instructions[i],
                        lengths=lengths[i]))

    def __len__(self) -> int:
        return len(self.graph_list)

    def __getitem__(self, idx: int):
        return self.graph_list[idx]


def train(model, train_dataset, val_dataset):
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, betas=(config.beta1, config.beta2), eps=config.eps,
                            weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=config.training_epochs // 2,
        eta_min=config.lr / 100
    )
    criterion = nn.CrossEntropyLoss().to(config.device)
    best_f1 = -1

    train_loader = PYGDataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

    for epoch in range(config.training_epochs):
        train_one_epoch(epoch, model, criterion, optimizer, train_loader)
        val_res = val(model, val_dataset)
        if val_res[-1] > best_f1:
            state = {
                'epoch': epoch,
                'val_f1': val_res[-1],
                'model': model
            }
            best_f1 = val_res[-1]
            torch.save(state, config.model_save_path)
            print(f"\033[31m val pre recall f1:{val_res} \033[0m")
        else:
            print(f"val pre recall f1:{val_res}")
        scheduler.step()
        print()


def train_one_epoch(epoch, model, criterion, optimizer, train_loader):
    model.train()
    f1s = []
    running_loss = 0
    total = 0

    for batch in tqdm(train_loader):
        optimizer.zero_grad()
        x, edge_index, y, block_instructions, lengths = batch.x, batch.edge_index, batch.y, batch.block_instructions, batch.lengths
        x, edge_index, y, block_instructions, lengths = x.to(config.device), edge_index.to(config.device), y.to(
            config.device), block_instructions.to(config.device), lengths.to(config.device)

        out = model(x, edge_index, block_instructions, lengths)

        loss = criterion(out, y)
        loss.backward()

        optimizer.step()

        _, y_pred = torch.max(out.data, 1)

        f1s.append(sklearn.metrics.f1_score(y.cpu(), y_pred.cpu(), zero_division=0))

        running_loss += loss.item()
        total += 1

    print(f"epoch {epoch},train-f1-avg:{np.average(f1s)},avg-loss:{running_loss / total}")


def val(model, dataset):
    model.eval()
    f1s = []
    pres = []
    recalls = []
    for data in tqdm(dataset, desc=f"val"):
        x, edge_index, y, block_instructions, lengths = data.x, data.edge_index, data.y, data.block_instructions, data.lengths
        x, edge_index, y, block_instructions, lengths = x.to(config.device), edge_index.to(config.device), y.to(
            config.device), block_instructions.to(config.device), lengths.to(config.device)

        out = model(x, edge_index, block_instructions, lengths)
        _, y_pred = torch.max(out.data, 1)
        y_cpu = y.cpu()
        y_pred_cpu = y_pred.cpu()
        pres.append(sklearn.metrics.precision_score(y_cpu, y_pred_cpu, zero_division=0))
        recalls.append(sklearn.metrics.recall_score(y_cpu, y_pred_cpu, zero_division=0))
        f1s.append(sklearn.metrics.f1_score(y_cpu, y_pred_cpu, zero_division=0))

    return np.average(pres), np.average(recalls), np.average(f1s)


def get_word2index(train_x) -> dict:
    word2index = {}
    for blocks in train_x:
        for block in blocks:
            for instr in block:
                if instr not in word2index:
                    word2index[instr] = len(word2index) + 1
    return word2index


def get_x(instr, word2index):
    xs = []
    for blocks in instr:
        tmp = []
        for block in blocks:
            instr_count = Counter(block)
            x = np.zeros(len(word2index))
            for instr, count in instr_count.items():
                if instr in word2index:
                    x[word2index[instr] - 1] = count / len(block)
            tmp.append(x)
        xs.append(torch.tensor(np.array(tmp), dtype=torch.float))
    return xs


def get_input():
    data = PKLUtil.load_pkl(config.data_path)
    split = PKLUtil.load_pkl(config.address_path)
    train_instr = [[[instr[1] for instr in block[2]] for block in data[address][0]] for address in split[0]]
    val_instr = [[[instr[1] for instr in block[2]] for block in data[address][0]] for address in split[1]]
    test_instr = [[[instr[1] for instr in block[2]] for block in data[address][0]] for address in split[2]]
    word2index = get_word2index(train_instr)

    train_x, val_x, test_x = get_x(train_instr, word2index), get_x(val_instr, word2index), get_x(test_instr, word2index)

    train_i = [[[word2index.get(instr, len(word2index) + 1) for instr in block] for block in blocks]
               for blocks in train_instr]
    val_i = [[[word2index.get(instr, len(word2index) + 1) for instr in block] for block in blocks]
             for blocks in val_instr]
    test_i = [[[word2index.get(instr, len(word2index) + 1) for instr in block] for block in blocks]
              for blocks in test_instr]
    train_lengths = [torch.tensor([len(block) for block in blocks]) for blocks in train_i]
    val_lengths = [torch.tensor([len(block) for block in blocks]) for blocks in val_i]
    test_lengths = [torch.tensor([len(block) for block in blocks]) for blocks in test_i]
    max_train_length = max([blocks.max().item() for blocks in train_lengths])
    max_val_length = max([blocks.max().item() for blocks in val_lengths])
    max_test_length = max([blocks.max().item() for blocks in test_lengths])
    train_i = [torch.tensor([block + [0] * (max_train_length - len(block)) for block in blocks]) for blocks in train_i]
    val_i = [torch.tensor([block + [0] * (max_val_length - len(block)) for block in blocks]) for blocks in val_i]
    test_i = [torch.tensor([block + [0] * (max_test_length - len(block)) for block in blocks]) for blocks in test_i]

    train_edge_index = [to_undirected(torch.nonzero(torch.tensor(data[address][1]) > 0).t()) for address in split[0]]
    val_edge_index = [to_undirected(torch.nonzero(torch.tensor(data[address][1]) > 0).t()) for address in split[1]]
    test_edge_index = [to_undirected(torch.nonzero(torch.tensor(data[address][1]) > 0).t()) for address in split[2]]

    train_y = [torch.tensor(data[address][2]) for address in split[0]]
    val_y = [torch.tensor(data[address][2]) for address in split[1]]
    test_y = [torch.tensor(data[address][2]) for address in split[2]]

    return (train_x, train_i, train_lengths, train_edge_index, train_y), (
        val_x, val_i, val_lengths, val_edge_index, val_y), \
        (test_x, test_i, test_lengths, test_edge_index, test_y), word2index


def eval():
    data = PKLUtil.load_pkl(config.data_path)
    split = PKLUtil.load_pkl(config.address_path)
    addresses = split[2]

    train_instr = [[[instr[1] for instr in block[2]] for block in data[address][0]] for address in split[0]]
    test_instr = [[[instr[1] for instr in block[2]] for block in data[address][0]] for address in addresses]
    word2index = get_word2index(train_instr)
    test_x = get_x(test_instr, word2index)
    test_edge_index = [to_undirected(torch.nonzero(torch.tensor(data[address][1]) > 0).t()) for address in addresses]

    test_i = [[[word2index.get(instr, len(word2index) + 1) for instr in block] for block in blocks]
              for blocks in test_instr]
    test_lengths = [torch.tensor([len(block) for block in blocks]) for blocks in test_i]
    max_test_length = max([blocks.max().item() for blocks in test_lengths])
    test_i = [torch.tensor([block + [0] * (max_test_length - len(block)) for block in blocks]) for blocks in test_i]
    test_y = [torch.tensor(data[address][2]) for address in addresses]

    test_data = (test_x, test_i, test_lengths, test_edge_index, test_y)

    pcs = [[block[0] for block in data[address][0]] for address in addresses]

    with open(config.model_save_path, 'rb') as f:
        checkpoint = torch.load(f)
        model = checkpoint['model'].to(config.device)
        print(f"val f1:{checkpoint['val_f1']}")

    model.eval()
    preds = []
    for id, data in enumerate(tqdm(Dataset(test_data), desc=f"inference")):
        x, edge_index, y, block_instructions, lengths = data.x, data.edge_index, data.y, data.block_instructions, data.lengths
        x, edge_index, y, block_instructions, lengths = x.to(config.device), edge_index.to(config.device), y.to(
            config.device), block_instructions.to(config.device), lengths.to(config.device)

        out = model(x, edge_index, block_instructions, lengths)
        _, y_pred = torch.max(out.data, 1)
        preds.append(y_pred.tolist())

    precisions = []
    f1s = []
    recalls = []
    for id, pred in enumerate(tqdm(preds, desc=f"compute")):
        files = os.listdir(os.path.join(config.neural_FEBI_ground_truth_path, addresses[id]))
        boundary_files = [filename for filename in files if re.match(".*\.boundary", filename)]
        function_boundaries_path = os.path.join(config.neural_FEBI_ground_truth_path, addresses[id], boundary_files[0])
        boundary, tag_to_pc, _ = PKLUtil.load_pkl(function_boundaries_path)
        public_pc_set = set([tag_to_pc[tag_id] for tag_id, _ in boundary[0][1].items()])
        fallback_pc_set = set([tag_to_pc[tag_id] for tag_id, _ in boundary[0][3].items()])
        priv_pc_set = set([tag_to_pc[tag_id] for tag_id, _ in boundary[0][2].items()])

        pred_start_pcs = set()
        for id1, pred_tag in enumerate(pred):
            if pred_tag == 1 and test_instr[id][id1][0] == 'JUMPDEST':
                pred_start_pcs.add(pcs[id][id1])

        pred_start_pcs = pred_start_pcs - public_pc_set - fallback_pc_set

        f1, precision, recall = compare_fs(priv_pc_set, pred_start_pcs)

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    print(np.average(precisions))
    print(np.average(recalls))
    print(np.average(f1s))


def compare_fs(ground_fs, target_fs):
    ground_fs_set = ground_fs
    target_fs_set = target_fs
    tp = ground_fs_set & target_fs_set
    fp = target_fs_set - ground_fs_set
    fn = ground_fs_set - target_fs_set
    return cal_score(len(tp), len(fp), len(fn))


def precision_score(tp, fp):
    if fp == 0:
        return 1, ""
    # if tp + fp == 0:
    #     return 0, "warn"
    else:
        return tp / (tp + fp), ""


def recall_score(tp, fn):
    if fn == 0:
        return 1, ""
    # if tp + fn == 0:
    #     return 0, "warn"
    else:
        return tp / (fn + tp), ""


def f1_score(p, r):
    if p + r == 0:
        return 0, "warn"
    else:
        return 2 * p * r / (p + r), ""


def cal_score(tp, fp, fn):
    p, warn_p = precision_score(tp, fp)

    if warn_p:
        warnings.warn("Precision: Div by Zero")

    r, warn_r = recall_score(tp, fn)
    if warn_r:
        warnings.warn("Recall: Div by Zero")

    f1, warn_f1 = f1_score(p, r)
    if warn_f1:
        warnings.warn("F1-Score: Div by Zero")

    return f1, p, r


def dump_pre_result():
    data = PKLUtil.load_pkl(config.data_path)
    split = PKLUtil.load_pkl(config.address_path)
    addresses = split[2]

    train_instr = [[[instr[1] for instr in block[2]] for block in data[address][0]] for address in split[0]]
    test_instr = [[[instr[1] for instr in block[2]] for block in data[address][0]] for address in addresses]
    word2index = get_word2index(train_instr)
    test_x = get_x(test_instr, word2index)
    test_edge_index = [to_undirected(torch.nonzero(torch.tensor(data[address][1]) > 0).t()) for address in addresses]

    test_i = [[[word2index.get(instr, len(word2index) + 1) for instr in block] for block in blocks]
              for blocks in test_instr]
    test_lengths = [torch.tensor([len(block) for block in blocks]) for blocks in test_i]
    max_test_length = max([blocks.max().item() for blocks in test_lengths])
    test_i = [torch.tensor([block + [0] * (max_test_length - len(block)) for block in blocks]) for blocks in test_i]
    test_y = [torch.tensor(data[address][2]) for address in addresses]

    test_data = (test_x, test_i, test_lengths, test_edge_index, test_y)

    pcs = [[block[0] for block in data[address][0]] for address in addresses]

    with open(config.model_save_path, 'rb') as f:
        checkpoint = torch.load(f)
        model = checkpoint['model'].to(config.device)
        print(f"val f1:{checkpoint['val_f1']}")

    model.eval()
    for id, data in enumerate(tqdm(Dataset(test_data), desc=f"inference")):
        start_time = time.time()
        x, edge_index, y, block_instructions, lengths = data.x, data.edge_index, data.y, data.block_instructions, data.lengths
        x, edge_index, y, block_instructions, lengths = x.to(config.device), edge_index.to(config.device), y.to(
            config.device), block_instructions.to(config.device), lengths.to(config.device)
        logits = model(x, edge_index, block_instructions, lengths)
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        out = probabilities[:, 1].tolist()
        pred_scores = {pcs[id][i]: out[i] for i in range(len(pcs[id]))}
        infer_time = time.time() - start_time
        with open(os.path.join(config.fsi_result_path, str(id)), "wb+") as f:
            pickle.dump((addresses[id], pred_scores, infer_time), f)


if __name__ == '__main__':
    train_data, val_data, test_data, word2index = get_input()
    print("======")
    print(len(word2index))
    print("======")
    model = FIRE(len(word2index)).to(config.device)
    train(model, Dataset(train_data), Dataset(val_data))

    print("=========")
    with open(config.model_save_path, 'rb') as f:
        model = torch.load(f)['model'].to(config.device)
        print(f"val pre recall f1:{val(model, Dataset(test_data))}")

    eval()
    # dump_pre_result()
