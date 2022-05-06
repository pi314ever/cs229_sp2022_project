"""Make a zip file for submission."""

import os
import pandas as pd

SRC_EXT = '.xlsx'

def make_zip():
    script_dir = os.path.dirname(os.path.realpath(__file__))

    # Create zip file
    zip_path = os.path.join(script_dir, 'submission.zip')
    print('Creating {}'.format(zip_path))
    for base_path, dir_names, file_names in os.walk('../dataset'):
        for file_name in file_names:
            if file_name.endswith(SRC_EXT):
                # Read file
                file_path = os.path.join(base_path, file_name)
                rel_path = os.path.relpath(file_path, script_dir)
                print(rel_path)


if __name__ == '__main__':
    make_zip()