import os.path

import config
from utils.FileUtil import FileUtil


def get_cmd() -> str:
    workdir = os.path.join(config.res_dir, config.mode, config.version)
    FileUtil.create_dir_if_not_exist(workdir)
    result_path = os.path.join(workdir, "result.json")
    cmd = f"{config.decompiler_path} -w {workdir} -r {result_path} -j {config.jobs} -T {config.timeout_seconds} " \
          f"--disable_scalable_fallback  --disable_inline "
    if config.mode == 'gigahorse':
        context_sensitivity = '"CONTEXT_SENSITIVITY=CallSiteContextPlus"'
    if config.mode == 'elipmoc':
        context_sensitivity = '"CONTEXT_SENSITIVITY=TransactionalContext"'
    if config.mode == 'shrnkr':
        context_sensitivity = '"CONTEXT_SENSITIVITY=TransactionalWithShrinkingContext"'
    cmd += f"-M {context_sensitivity} {os.path.join(config.giga_data_prepare_dir, config.version)}"
    return cmd


if __name__ == '__main__':
    # run command in terminal
    print(get_cmd())
