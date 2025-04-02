import os
import time

from llama_index.core import Settings
from llama_index.core.callbacks import TokenCountingHandler, CallbackManager
from llama_index.llms.groq import Groq
from tqdm import tqdm

from app.data_processing.data_loaders import load_index
from app.utils.CONSTANTS import CORRESPONDENCES, LEVELS, INDICES, INDICES_EXP


def retrieve_descriptions(descr_folder, level):
    """
    create the dictionary containing the name of the problem as the key
    and the description for the level in args as value
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


def ranking(index, model, descriptions, result_path, k=10):
    """
    Args
    -----------
        index = BaseIndex from llama_index
        model = Groq LLM
        descriptions = dict[str:str] containing one description per problem
        result_path = file to the txt file where the results will be appened
    """

    token_counter = TokenCountingHandler()
    callback_manager = CallbackManager([token_counter])
    Settings.callback_manager = callback_manager
    model.callback_manager = Settings.callback_manager
    model_tpm = 30_000

    query_engine = index.as_query_engine(
        llm=model, similarity_top_k=k
    )

    with open(result_path, "a") as f:  # Open file in append mode

        for problem_name, problem_descr in tqdm(descriptions.items(), desc="Generating Answers"):
            start_time = time.time()
            response = query_engine.query(problem_descr)
            elapsed_time = time.time() - start_time

            if token_counter.total_llm_token_count >= model_tpm:
                sleep_time = 60 - elapsed_time
                if sleep_time > 0:
                    time.sleep(sleep_time)
                token_counter.reset_counts()

            family_name_1 = response.source_nodes[0].metadata["model_name"]
            family_name_2 = response.source_nodes[1].metadata["model_name"]
            family_name_3 = response.source_nodes[2].metadata["model_name"]
            family_name_4 = response.source_nodes[3].metadata["model_name"]
            family_name_5 = response.source_nodes[4].metadata["model_name"]

            f.write(
                problem_name
                + " "
                + family_name_1
                + " "
                + family_name_2
                + " "
                + family_name_3
                + " "
                + family_name_4
                + " "
                + family_name_5
                + "\n"
            )

    return


def compute_mrr(result_path):
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


def experiment1():
    model = Groq(
        model="llama3-70b-8192",
        model_kwargs={"seed": 19851900},
        temperature=0,
    )

    for level in LEVELS:
        descr_folder = "data/generated_descriptions"
        descriptions = retrieve_descriptions(descr_folder, level)
        for index_level in INDICES:
            if level not in index_level:
                index_path = "data/vector_dbs/code_as_text/" + index_level
                index = load_index(index_path)
                result_path = (
                        "_results/txt/exp1/no_rerank/"
                        + "index_"
                        + index_level
                        + "_level_"
                        + level
                        + ".txt"
                )

                ranking(index, model, descriptions, result_path, k=5)

    with open("_results/txt/exp1/no_rerank/exp1.txt", "a") as f:
        for level in LEVELS:
            for index_level in INDICES:
                if level not in index_level:
                    result_path = (
                            "_results/txt/exp1/no_rerank/"
                            + "index_"
                            + index_level
                            + "_level_"
                            + level
                            + ".txt"
                    )
                    mrr = compute_mrr(result_path)
                    print(f"Level {level}, Index {index_level}, MRR = {mrr}")
                    f.write(f"Level {level}, Index {index_level}, MRR = {mrr}\n")


def experiment2(csplib_desc_dir_path):
    model = Groq(
        model="llama3-70b-8192",
        model_kwargs={"seed": 19851900},
        temperature=0,
    )

    descriptions = retrieve_descriptions_csplib(csplib_desc_dir_path)

    for index_level in INDICES_EXP:
        index_path = "data/vector_dbs/code_as_text/" + index_level
        index = load_index(index_path)

        result_path = "_results/txt/" + "exp2/no_rerank/" + "index_" + index_level + ".txt"
        ranking(index, model, descriptions, result_path, k=5)

        mrr = compute_mrr(result_path)
        print(index_level + " " + str(mrr))

    with open("_results/txt/exp2/no_rerank/exp2.txt", "a") as f:
        for index_level in INDICES_EXP:
            result_path = "_results/txt/exp2/no_rerank/index_" + index_level + ".txt"
            mrr = compute_mrr(result_path)
            print(f"Index {index_level}, MRR = {mrr}")
            f.write(f"Index {index_level}, MRR = {mrr}\n")


experiment1()
experiment2(csplib_desc_dir_path = "data/output/generated_descriptions")
