from .CMDProxy import CMDProxy


class EtherSolve:
    @staticmethod
    def run(java_path: str, ether_solve_path: str, result_dir: str, result_file: str, bytecode: str) -> bool:
        cmd = f"cd {result_dir}; {java_path} -jar {ether_solve_path} -j -r {bytecode} -o {result_file}"
        return CMDProxy.run_cmd(cmd)
