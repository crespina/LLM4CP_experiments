import os
import time

from llama_index.core import Settings, StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from llama_index.core.callbacks import TokenCountingHandler, CallbackManager

from tqdm import tqdm


os.environ["GROQ_API_KEY"] = "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"


corresp = {
    "All-Interval_Series": "all_interval",
    "Balanced_Academic_Curriculum_Problem__BACP_": "curriculum",
    "Balanced_Incomplete_Block_Designs": "bibd",
    "Bus_Driver_Scheduling": "bus_scheduling_csplib",
    "Car_Sequencing": "car",
    "Crossfigures": "crossfigure",
    "Diamond-free_Degree_Sequences": "diamond_free_degree_sequence",
    "Golomb_rulers": "golomb",
    "Graceful_Graphs": "graph",
    "Killer_Sudoku": "killer_sudoku",
    "Langford_s_number_problem": "langford",
    "Magic_Hexagon": "magic_hexagon",
    "Magic_Squares_and_Sequences": "magic_sequence",
    "Maximum_Clique": "clique",
    "Maximum_density_still_life": "maximum_density_still_life",
    "N-Queens": "queens",
    "Nonogram": "nonogram_create_automaton2",
    "Number_Partitioning": "partition",
    "Optimal_Financial_Portfolio_Design": "opd",
    "Quasigroup_Completion": "QuasigroupCompletion",
    "Quasigroup_Existence": "QuasiGroupExistence",
    "Rotating_Rostering_Problem": "RosteringProblem",
    "Schur_s_Lemma": "schur",
    "Social_Golfers_Problem": "golfers",
    "Solitaire_Battleships": "sb",
    "Steiner_triple_systems": "steiner",
    "Stochastic_Assignment_and_Scheduling_Problem": "stoch_fjsp",
    "Synchronous_Optical_Networking__SONET__Problem": "sonet_problem",
    "Template_Design": "template_design",
    "The_n-Fractions_Puzzle": "fractions",
    "The_Rehearsal_Problem": "rehearsal",
    "Traffic_Lights": "traffic_lights_table",
    "Traveling_Tournament_Problem_with_Predefined_Venues__TTPPV_": "TTPPV",
    "Vessel_Loading": "vessel-loading",
    "Warehouse_Location_Problem": "warehouses",
    "Water_Bucket_Problem": "water_buckets1",
}

def load_index(index_path):
    Settings.embed_model = HuggingFaceEmbedding(model_name="Alibaba-NLP/gte-modernbert-base")

    Settings.show_progress = True

    if os.path.exists(index_path):
        storage_context = StorageContext.from_defaults(
            persist_dir=index_path
        )
        index = load_index_from_storage(storage_context, show_progress=True)
        print("Loaded index from storage.")
        return index
    else:
        print("Index storage directory not found. Parse and store the index first.")
        exit()


def retrieve_descriptions(descr_folder, level):
    """
    create the dictionnary containing the name of the problem as the key
    and the description for the level in args as value
    """

    dict = {}
    for folder_name in os.listdir(descr_folder):
        folder_path = os.path.join(descr_folder, folder_name)
        if os.path.isdir(folder_path):

            path = os.path.join(folder_path, level+".txt")

            with open(path, "r", encoding="utf-8") as f:
                text_description = f.read()
            dict[folder_name] = text_description
    return dict


def retrieve_descriptions_csplib():

    def corresponding_name(problem_name):
        if problem_name not in corresp:
            print("error " + problem_name)
        model_name = corresp[problem_name]
        return model_name

    descriptions = {}

    for filename in os.listdir("data/csplib_masked"):
        if filename.endswith(".txt"):
            file_path = os.path.join("data/csplib_masked", filename)
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

    MRR = sum(reciprocal_ranks) / total
    return MRR


def experiment1():

    model = Groq(
        model="llama3-70b-8192",
        model_kwargs={"seed": 19851900},
        temperature=0,
    )

    levels = ["expert", "medium", "beginner"]
    indexes = [
        "code",
        "expert",
        "medium",
        "beginner",
        "mediumexpert",
        "beginnerexpert",
        "beginnermedium",
    ]

    for level in levels:
        descr_folder = "data/generated_descriptions"
        descriptions = retrieve_descriptions(descr_folder, level)
        for index_level in indexes:
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
        for level in levels:
            for index_level in indexes:
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


def experiment2():

    model = Groq(
        model="llama3-70b-8192",
        model_kwargs={"seed": 19851900},
        temperature=0,
    )

    descriptions = retrieve_descriptions_csplib()

    indexes = [
        "code",
        "expert",
        "medium",
        "beginner",
        "beginnermedium",
        "beginnerexpert",
        "mediumexpert",
        "beginnermediumexpert",
    ]

    for index_level in indexes:
        index_path = "data/vector_dbs/code_as_text/" + index_level
        index = load_index(index_path)

        result_path = "_results/txt/" + "exp2/no_rerank/" + "index_" + index_level + ".txt"
        ranking(index, model, descriptions, result_path, k=5)

        mrr = compute_mrr(result_path)
        print(index_level + " " + str(mrr))

    with open("_results/txt/exp2/no_rerank/exp2.txt", "a") as f:
        for index_level in indexes:
            result_path = "_results/txt/exp2/no_rerank/index_" + index_level + ".txt"
            mrr = compute_mrr(result_path)
            print(f"Index {index_level}, MRR = {mrr}")
            f.write(f"Index {index_level}, MRR = {mrr}\n")


experiment1()
experiment2()
