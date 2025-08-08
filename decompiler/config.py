from utils.FileUtil import FileUtil

decompiler_path = "/home//gigahorse-toolchain/gigahorse.py"
neural_FEBI_ground_truth_dir = "/data//func_body/ground-truth"
giga_data_prepare_dir = "/home//gigahorse-toolchain/func_body_data"

res_dir = "/home//gigahorse-toolchain/result"
elipmoc_res_dir = "/home//gigahorse-toolchain/result/elipmoc"
shrnkr_res_dir = "/home//gigahorse-toolchain/result/shrnkr"

jobs = 64
timeout_seconds = 120

mode = "shrnkr"  # gigahorse,elipmoc,shrnkr
version = "0.4.25-optimized"

address_path = f"/home//func_body/dataset-basic_block/{version}.split"