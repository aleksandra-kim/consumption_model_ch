from pathlib import Path


def get_habe_filepath(directory, year, tag):
    directory = Path(directory)
    files = [x for x in directory.iterdir() if (year in x.name) and (tag in x.name)]
    assert len(files) == 1
    return files[0]
