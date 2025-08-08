import os.path
import re
import shutil

from tqdm import tqdm

import config
from utils.FileUtil import FileUtil
from utils.PKLUtil import PKLUtil


def data_prepare(version: str):
    ground_dir = os.path.join(config.neural_FEBI_ground_truth_dir, version, "data")
    data_save_dir = os.path.join(config.giga_data_prepare_dir, version)
    FileUtil.create_dir_if_not_exist(data_save_dir)

    split = PKLUtil.load_pkl(config.address_path)
    addresses = split[2]

    for address in tqdm(os.listdir(ground_dir)):
        if address not in addresses:
            continue
        files = os.listdir(os.path.join(ground_dir, address))
        runtime_file = [filename for filename in files if re.match(".*\.bin-runtime", filename)][0]
        shutil.copy(os.path.join(ground_dir, address, runtime_file), os.path.join(data_save_dir, f"{address}.hex"))


if __name__ == '__main__':
    data_prepare(config.version)
