import os

current_threshold = 0.5
delay = 0.05
low_bounder_threshold = 0.3

debug = False

visited_max_times = 100
init_stack_heigth = 32


version = "0.4.25-optimized"


contracts_dir = f"/home//func_body/ground-truth/{version}/data/"
fsi_result_path = f'/data//func_body/fsi-result/{version}-graph-FEBI/'
result_path = f'/data//func_body/fbd-result/{version}-graph-FEBI/'


if not os.path.exists(result_path):
    os.makedirs(result_path)