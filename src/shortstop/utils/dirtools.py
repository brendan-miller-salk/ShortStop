import os


def check_dir(folder):
    if not os.path.exists(folder):
        os.mkdir(folder)

def check_multi_dirs(folders):
    for folder in folders:
        if not os.path.exists(folder):
            os.mkdir(folder)

