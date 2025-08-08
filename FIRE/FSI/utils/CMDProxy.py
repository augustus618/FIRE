import platform
import subprocess


class CMDProxy:
    @staticmethod
    def __cmd_os_cast(command: str):
        return command.replace(";", "&") if platform.system() == "Windows" else command

    @staticmethod
    def run_cmd(command: str, stdout=None, timeout=10 * 60):
        if stdout is not None:
            command += f' > {stdout} 2>&1'
        command = CMDProxy.__cmd_os_cast(command)
        try:
            subprocess.check_call(command, shell=True, timeout=timeout)
            return True
        except subprocess.CalledProcessError as e:
            print(f"{command} executed failure \n return code：{e.returncode}")
        except subprocess.TimeoutExpired:
            print(f"{command} executed timeout")
        return False

    @staticmethod
    def run_copy_cmd(source_path: str, target_path: str):
        command = f"cp -r {source_path} {target_path}"
        CMDProxy.run_cmd(command)
