from pathlib import Path
import pickle


def get_habe_filepath(directory, year, tag):
    directory = Path(directory)
    files = [x for x in directory.iterdir() if (year in x.name) and (tag in x.name)]
    assert len(files) == 1
    return files[0]


def write_pickle(data, filepath):
    """Write ``data`` to a file with .pickle extension"""
    with open(filepath, "wb") as f:
        pickle.dump(data, f)


def read_pickle(filepath):
    """Read ``data`` from a file with .pickle extension"""
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    return data