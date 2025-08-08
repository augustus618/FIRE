import pickle


class PKLUtil:
    @staticmethod
    def load_pkl(pkl_path: str):
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
            return data

    @staticmethod
    def dump_pkl(data, path: str):
        with open(path, 'wb') as f:
            pickle.dump(data, f)
