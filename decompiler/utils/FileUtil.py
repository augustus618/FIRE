import os.path
import shutil

from typing import List


class FileUtil:
    @staticmethod
    def delete_dir_if_exist_and_create(directory: str):
        try:
            if os.path.exists(directory):
                shutil.rmtree(directory)
            os.makedirs(directory)
        except Exception as e:
            print("directory:{} can not be rm".format(directory))
            print(e)

    @staticmethod
    def create_dir_if_not_exist(directory: str):
        if not os.path.exists(directory):
            os.makedirs(directory)

    @staticmethod
    def check_dir_exist_and_create_if_not_exist(directory: str) -> bool:
        if os.path.exists(directory):
            return True
        else:
            os.makedirs(directory)
            return False

    @staticmethod
    def check_dir_exist_and_create_by_reset_flag(directory: str, reset: bool) -> bool:
        return FileUtil.delete_dir_if_exist_and_create(directory) \
            if reset else FileUtil.check_dir_exist_and_create_if_not_exist(directory)

    @staticmethod
    def read_lines(path: str) -> List[str]:
        with open(path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f.readlines()]

    @staticmethod
    def read(path: str) -> str:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()

    @staticmethod
    def write(path: str, content: str) -> None:
        with open(path, 'w') as f:
            f.write(content)
