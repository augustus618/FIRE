import os

current_threshold = 0.5
delay = 0.05
low_bounder_threshold = 0.3

debug = False

visited_max_times = 100
init_stack_heigth = 32

is_cfg = False

version = "0.5.17-optimized"
workers = 8

contracts_dir = f"/home//func_body/ground-truth/{version}/data/"
fsi_result_path = f'/data//func_body/fsi-result/{version}-neural-FEBI/'
result_path = f'/data//func_body/fbd-result/{version}-neural-FEBI/'


if not os.path.exists(result_path):
    os.makedirs(result_path)