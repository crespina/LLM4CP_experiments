import os

from app.utils.CONSTANTS import CORRESPONDENCES


def retrieve_descriptions(descr_folder, level):
    """
    Create a dictionary containing the name of the problem as the key
    and the description for the specified level as value
    """
    descriptions = {}
    for folder_name in os.listdir(descr_folder):
        folder_path = os.path.join(descr_folder, folder_name)
        if os.path.isdir(folder_path):
            path = os.path.join(folder_path, level + ".txt")
            with open(path, "r", encoding="utf-8") as f:
                text_description = f.read()
            descriptions[folder_name] = text_description
    return descriptions


def retrieve_descriptions_csplib(desc_dir_path):
    """
    Retrieve CSPLib problem descriptions and map them to corresponding model names
    """

    def corresponding_name(problem_name):
        if problem_name not in CORRESPONDENCES:
            print("error " + problem_name)
        model_name = CORRESPONDENCES[problem_name]
        return model_name

    descriptions = {}
    for filename in os.listdir(desc_dir_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(desc_dir_path, filename)
            with open(file_path, "r", encoding="utf-8") as file:
                name = os.path.splitext(filename)[0]
                model_name = corresponding_name(name)
                descriptions[model_name] = file.read()

    return descriptions


def compute_mrr(result_path):
    """
    Compute Mean Reciprocal Rank for results in the given file
    """
    reciprocal_ranks = []
    total = 0
    with open(result_path, 'r') as f:
        for line in f:
            words = line.strip().split()
            problem_name = words[0]
            family_names = words[1:6]

            if problem_name in family_names:
                reciprocal_ranks.append(1 / (family_names.index(problem_name) + 1))
            else:
                reciprocal_ranks.append(0)

            total += 1

    mrr = sum(reciprocal_ranks) / total
    return mrr

