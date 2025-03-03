import os

from llama_index.core import Settings, StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from llama_index.postprocessor.cohere_rerank import CohereRerank

from tqdm import tqdm

os.environ["GROQ_API_KEY"] = "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"


def load_index(index_path):
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

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


def ranking(index, model, reranker, descriptions, result_path):
    """
    Args
    -----------
        index = BaseIndex from llama_index
        model = Groq LLM
        reranker = CohereRerank
        descriptions = dict[str:str] containing one description per problem
        result_path = file to the txt file where the results will be appened
    """

    query_engine = index.as_query_engine(
        llm=model, similarity_top_k=10, node_postprocessors=[reranker]
    )

    with open(result_path, "a") as f:  # Open file in append mode

        for problem_name, problem_descr in tqdm(descriptions.items(), desc="Generating Answers"):
            
            response = query_engine.query(problem_descr)
            family_name_1 = response.source_nodes[0].metadata["model_name"]
            family_name_2 = response.source_nodes[1].metadata["model_name"]
            family_name_3 = response.source_nodes[2].metadata["model_name"]
            family_name_4 = response.source_nodes[3].metadata["model_name"]
            family_name_5 = response.source_nodes[4].metadata["model_name"]

            print(
                problem_name,
                family_name_1,
                family_name_2,
                family_name_3,
                family_name_4,
                family_name_5,
            )
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

    with open(result_path, 'r') as f:
        for line in f:
            words = line.strip().split()
            problem_name = words[0]
            family_names = words[1:6]

            total = 0
            reciprocal_ranks = []

            if problem_name in family_names:
                reciprocal_ranks.append(1 / (family_names.index(problem_name) + 1))
            else:
                reciprocal_ranks.append(0)

            total += 1

    MRR = sum(reciprocal_ranks) / total
    return MRR


def experiment():

    model = Groq(
        model="llama-3.3-70b-versatile",
        model_kwargs={"seed": 19851900},
        temperature=0.1,
    )
    index = load_index()
    reranker = CohereRerank(api_key="STPahNFoWeYX4FSAoMx7NzHNgH2ejINXLDKIYOr4", top_n=5)

    levels = ["expert", "medium", "beginner"]
    indexes = ["code", "expert", "medium", "beginner", "expertmedium", "expertbeginner","beginnermedium"]

    mrr_dict = {}

    for level in levels :
        descr_folder = "data/generated_descriptions"
        descriptions = retrieve_descriptions(descr_folder, level)
        for index in indexes : 
            if level not in index:
                result_path = "_results/txt/"+"index_"+index+"_level_"+level+".txt"
                
                directory = os.path.dirname(result_path)
                if not os.path.exists(directory):
                    os.makedirs(directory)

                ranking(index, model, reranker, descriptions, result_path)
                mrr = compute_mrr(result_path)
                mrr_dict[(level, index)] = mrr

    with open("_results/txt/exp1.txt", "a") as f:
        for key, value in mrr_dict.items():
            print(f"Level {key[0]}, Index {key[1]}: MRR = {value}")
            f.write(f"Level {key[0]}, Index {key[1]}: MRR = {value}")
