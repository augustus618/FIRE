import os.path
import re

import config
from utils.EtherSolve import EtherSolve
from utils.FileUtil import FileUtil
from utils.JSONUtil import JSONUtil
from utils.PKLUtil import PKLUtil
from utils.Tasks import AbstractTask
from utils.ThreadPool import FixedTaskThreadPool


def parse_ether_solve_result(result_path: str, start_addresses: set):
    result_json = JSONUtil.load(result_path)
    nodes = result_json["runtimeCfg"]["nodes"]
    successors = result_json["runtimeCfg"]["successors"]
    starts = [node["offset"] for node in nodes]
    ends = [node["offset"] + node["length"] - 1 for node in nodes]
    parsed_opcodes = [node["parsedOpcodes"] for node in nodes]

    blocks = []
    for i in range(len(nodes)):
        instructions = parsed_opcodes[i].replace(":", "").split("\n")
        instructions = [instruction.split() for instruction in instructions]
        for j in range(len(instructions)):
            instructions[j][0] = int(instructions[j][0])
            if len(instructions[j]) >= 3:
                instructions[j][2] = int(instructions[j][2], 16) if instructions[j][2] != "BLOCK" else 0
        blocks.append((starts[i], ends[i], tuple(instructions)))

    start2index = {s: idx for idx, s in enumerate(starts)}
    edges = [[0 for _ in range(len(nodes))] for _ in range(len(nodes))]
    for successor in successors:
        for to in successor["to"]:
            if to in start2index:
                edges[start2index[successor["from"]]][start2index[to]] = 1

    return blocks, edges, [1 if block[0] in start_addresses else 0 for block in blocks]


def get_function_start_addresses(function_boundaries):
    boundary, tag_to_pc, _ = function_boundaries
    return {tag_to_pc[entry] for i in range(1, 4) for entry in boundary[0][i]}


class MakeDatasetTask(AbstractTask):
    def __init__(self, contract_path: str, contract_address: str, dataset: dict):
        self.contract_address = contract_address
        self.dataset = dataset
        self.contract_path = contract_path

    def run(self):
        files = os.listdir(os.path.join(self.contract_path, self.contract_address))

        boundary_file = [filename for filename in files if re.match(".*\.boundary", filename)][0]
        function_boundaries = PKLUtil.load_pkl(
            os.path.join(self.contract_path, self.contract_address, boundary_file))
        start_addresses = get_function_start_addresses(function_boundaries)

        runtime_file = [filename for filename in files if re.match(".*\.bin-runtime", filename)][0]
        bytecode = FileUtil.read(os.path.join(self.contract_path, self.contract_address, runtime_file))
        et_result_path = os.path.join(config.ether_solve_result_dir, f"{self.contract_address}.json")

        if not os.path.exists(et_result_path):
            EtherSolve.run(config.java_path, config.ether_solve_path,
                           config.ether_solve_result_dir, f"{self.contract_address}.json", bytecode)
        self.dataset[self.contract_address] = parse_ether_solve_result(et_result_path, start_addresses)
        if os.path.exists(et_result_path):
            self.dataset[self.contract_address] = parse_ether_solve_result(et_result_path, start_addresses)

    def get_description(self):
        return self.contract_address


def make_dataset(contract_path=config.neural_FEBI_ground_truth_path):
    dataset = {}

    thread_pool = FixedTaskThreadPool(num_threads=64)
    for contract_address in os.listdir(contract_path):
        thread_pool.add_task(MakeDatasetTask(contract_path, contract_address, dataset))
    thread_pool.start()
    thread_pool.wait_completion()
    thread_pool.stop()

    print(len(dataset), len(os.listdir(contract_path)))
    PKLUtil.dump_pkl(dataset, config.data_path)



def get_word2index(train_x) -> dict:
    word2index = {'<pad>': 0}
    for blocks in train_x:
        for block in blocks:
            for instr in block:
                if instr not in word2index:
                    word2index[instr] = len(word2index)
    word2index['<unk>'] = len(word2index)
    return word2index


def check():
    pass

if __name__ == '__main__':
    make_dataset()
    # split_dataset()
    # make_tensor_dataset()
    # check()