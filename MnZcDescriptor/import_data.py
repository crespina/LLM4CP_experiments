import os
from llama_index.core import Document
from pathlib import Path
import os
import shutil
import git
import stat


def fetch(
    repo_url,
    local_repo_dir="MnZcDescriptor/temp_repo",
    output_dir="MnZcDescriptor/models_mzn",
):

    def handle_remove_readonly(func, path, exc_info):
        os.chmod(path, stat.S_IWRITE)
        func(path)

    if os.path.exists(local_repo_dir):
        shutil.rmtree(local_repo_dir, onerror=handle_remove_readonly)
    git.Repo.clone_from(repo_url, local_repo_dir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    mzn_files = []
    for root, dirs, files in os.walk(local_repo_dir):
        for file in files:
            if file.endswith(".mzn"):
                mzn_files.append(os.path.join(root, file))

    for file_path in mzn_files:
        shutil.copy(file_path, output_dir)

    shutil.rmtree(local_repo_dir, onerror=handle_remove_readonly)


def convert_mzn_to_txt():
    """
    Converts all .mzn files in the specified folder to .txt files.

    Args:
        folder_path (str): The path to the folder containing the .mzn files.
    """

    for root, dirs, files in os.walk("MnZcDescriptor/models_mzn"):
        for file in files:
            if file.endswith(".mzn"):

                with open("MnZcDescriptor/models_mzn/" + file, "r") as mzn_file:
                    content = mzn_file.read()
                    txt_file_path = os.path.join(
                        "MnZcDescriptor/models_mzn", file.replace(".mzn", ".txt")
                    )
                # Write the content to the .txt file
                with open(txt_file_path, "w") as txt_file:
                    txt_file.write(content)

    print("Conversion completed: All .mzn files have been converted to .txt.")


def instantiate():

    fetch("https://github.com/MiniZinc/minizinc-examples.git")
    convert_mzn_to_txt()


