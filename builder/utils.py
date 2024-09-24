from pathlib import Path
import os
import shutil
from typing import Callable
import tarfile

def get_resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    base_path = Path(__file__).absolute().parent.parent
    return str(Path(base_path, relative_path))

def copy_files(source_dir, destination_dir, filter: Callable[[str], bool]=None):
    # Ensure destination directory exists
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    # List all files in the source directory
    files = os.listdir(source_dir)

    for file in files:
        if filter and not filter(file):
            continue
        # Full path to the source file
        full_file_name = os.path.join(source_dir, file)

        # Check if it's a file (not a directory)
        if os.path.isfile(full_file_name):
            shutil.copy(full_file_name, destination_dir)

def create_tar_gz(output_filename, source_dir):
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_dir, arcname=".")