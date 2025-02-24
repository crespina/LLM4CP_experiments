import os
from time import sleep

from langchain.output_parsers import StructuredOutputParser
from llama_index.core import PromptTemplate, Document
from llama_index.core import VectorStoreIndex
from llama_index.core.output_parsers import LangchainOutputParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from tqdm import tqdm
from langchain.output_parsers import ResponseSchema
from llama_index.core import Settings
from llama_index.core import Settings, StorageContext, load_index_from_storage
from llama_index.postprocessor.cohere_rerank import CohereRerank

os.environ["GROQ_API_KEY"] = "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"


def get_response_schema(query_config_dict):
    response_schema = []
    for field, description in query_config_dict.items():
        response_schema.append(
            ResponseSchema(name=field.strip(), description=description.strip())
        )
    return response_schema


family_name = { 
    "all_interval" : "All-Interval_Series",
    "bibd" : "Balanced_Incomplete_Block_Designs",
    "bus_scheduling_csplib" : "Bus_Driver_Scheduling",
    "car" : "Car_Sequencing",
    "clique" : "Maximum_Clique",
    "crossfigure" : "Crossfigures",
    "curriculum" : "Balanced_Academic_Curriculum_Problem__BACP_",
    "diamond_free_degree_sequence" : "Diamond-free_Degree_Sequences",
    "fractions" : "The_n-Fractions_Puzzle",
    "golfers" : "Social_Golfers_Problem",
    "golomb" : "Golomb_rulers",
    "K4xP2Graceful" : "Graceful_Graphs",
    "killer_sudoku" : "Killer_Sudoku",
    "langford" : "Langford_s_number_problem",
    "magic_hexagon" : "Magic_Hexagon",
    "magic_sequence" : "Magic_Squares_and_Sequences",
    "maximum_density_still_life" : "Maximum_density_still_life",
    "nonogram_create_automaton2" : "Nonogram",
    "opd" : "Optimal_Financial_Portfolio_Design",
    "partition" : "Number_Partitioning",
    "QuasigroupCompletion" : "Quasigroup_Completion",
    "QuasiGroupExistence" : "Quasigroup_Existence",
    "queens" : "N-Queens",
    "rehearsal" : "The_Rehearsal_Problem",
    "RosteringProblem" : "Rotating_Rostering_Problem",
    "sb" : "Solitaire_Battleships",
    "schur" : "Schur_s_Lemma",
    "sonet_problem" : "Synchronous_Optical_Networking__SONET__Problem",
    "steiner" : "Steiner_triple_systems",
    "stoch_fjsp" : "Stochastic_Assignment_and_Scheduling_Problem",
    "template_design" : "Template_Design",
    "traffic_lights_table" : "Traffic_Lights",
    'TTPPV' : "Traveling_Tournament_Problem_with_Predefined_Venues__TTPPV_",
    "vessel-loading" : "Vessel_Loading",
    "warehouses" : "Warehouse_Location_Problem",
    "water_buckets1" : "Water_Bucket_Problem",
}

def replace(filename):
    for (model, family) in family_name.items():
        if model == filename :
            return family
    
    return None

def only_model():

    documents = []
    embeddings_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

    for filename in tqdm(
            os.listdir("data/csplib_models_concat"), desc="Models"
        ):
        file_path = os.path.join("data/csplib_models_concat", filename)
        if os.path.isfile(file_path):
            with open(file_path, "r", encoding="utf-8") as file:
                file_content = file.read()

                cp_model = Document(
                        text=file_content,
                        metadata={
                            "problem_family": replace(os.path.splitext(filename)[0])
                            },
                        id_=os.path.splitext(filename)[0],
                    )

                documents.append(cp_model)
                sleep(3)

    index = VectorStoreIndex.from_documents(
            documents=documents,
            embed_model=embeddings_model,
            show_progress=True,
        )

    index.storage_context.persist(persist_dir="data/vector_dbs/ablation/model")


def description():

    Settings.chunk_size = 4096

    description_template = PromptTemplate(
        "You are an expert in high-level constraint modelling and solving discrete optimization problems. \n"
        "In particular, you know Minizinc. You are provided with one or several Minizinc models that represents a single classical "
        "problem in constraint programming. Your task is to identify what is the problem modelled and give a "
        "complete description of the problem to the user. \n"
        "This is the source code of the model(s):\n"
        "--------------\n"
        "{source_code}"
        "--------------\n"
        "The format of the answer should be without any variation a JSON-like format with the following keys and "
        "explanation of what the corresponding values should be:\n"
        "name: The name of the problem\n"
        "description: A description of the problem in English\n"
        "variables: A string containing the list of all the decision variables in mathematical notation, "
        "followed by an explanation of what they are in English\n"
        "constraints: A string containing the list of all the constraints in mathematical notation, followed by an "
        "explanation of what they are in English\n"
        "objective: The objective of the problem (minimize or maximize what value)"
    )

    documents = []
    embeddings_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

    output_parser = LangchainOutputParser(
        StructuredOutputParser.from_response_schemas(
            get_response_schema(
                query_config_dict={
                    "name": "The name of the problem",
                    "description": "A description of the problem",
                    "variables": "All the decision variables in mathematical notation, followed by an explanation of what "
                    "they are in English",
                    "constraints": "All the constraints in mathematical notation only",
                    "objective": "The objective of the problem (minimize or maximize what value)",
                }
            )
        )
    )

    descriptor_model = Groq(
        model="llama3-70b-8192",
        api_key="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",
        model_kwargs={"seed": 42},
        temperature=0.1,
        output_parser=output_parser,
    )

    for filename in tqdm(os.listdir("data/csplib_models_concat"), desc="Models"):
        file_path = os.path.join("data/csplib_models_concat", filename)
        if os.path.isfile(file_path):
            with open(file_path, "r", encoding="utf-8") as file:
                file_content = file.read()

                prompt = description_template.format(source_code=file_content)
                text_description = descriptor_model.complete(prompt=prompt, formatted=True)

                cp_model = Document(
                    text=text_description.text,
                    metadata={
                        "problem_family": replace(os.path.splitext(filename)[0]),
                        "source_code": file_content,
                    },
                    id_=os.path.splitext(filename)[0],
                )

                documents.append(cp_model)
                sleep(3)

    index = VectorStoreIndex.from_documents(
        documents=documents,
        embed_model=embeddings_model,
        show_progress=True,
    )

    index.storage_context.persist(persist_dir="data/vector_dbs/ablation/description")


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

def study(index_path, txt, results_txt_path):

    reranker = CohereRerank(api_key="STPahNFoWeYX4FSAoMx7NzHNgH2ejINXLDKIYOr4", top_n=5)

    model = Groq(
        model="llama3-70b-8192",
        model_kwargs={"seed": 19851900},
        temperature=0.1,
    )

    index = load_index(index_path)

    query_engine = index.as_query_engine(
        llm=model, similarity_top_k=10, node_postprocessors=[reranker]
    )

    total = 0
    correct1 = 0
    correct2 = 0
    correct3 = 0
    correct4 = 0
    correct5 = 0
    incorrect = 0

    with open(results_txt_path, "a") as f:  # Open file in append mode

        for problem_name, problem_descr in tqdm(txt.items(), desc="Generating Answers"):

            response = query_engine.query(problem_descr)
            family_name_1 = response.source_nodes[0].metadata["problem_family"]
            family_name_2 = response.source_nodes[1].metadata["problem_family"]
            family_name_3 = response.source_nodes[2].metadata["problem_family"]
            family_name_4 = response.source_nodes[3].metadata["problem_family"]
            family_name_5 = response.source_nodes[4].metadata["problem_family"]

            if problem_name != family_name_1:
                if problem_name != family_name_2:
                    if problem_name != family_name_3:
                        if problem_name != family_name_4:
                            if problem_name != family_name_5:
                                incorrect += 1
                            else:
                                correct5 += 1
                        else:
                            correct4 += 1
                    else:
                        correct3 += 1
                else:
                    correct2 += 1
            else:
                correct1 += 1

            total += 1

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

        f.write(
            "total = "
            + str(total)
            + " correct1 = "
            + str(correct1)
            + " correct2 = "
            + str(correct2)
            + " correct3 = "
            + str(correct3)
            + " correct4 = "
            + str(correct4)
            + " correct5 = "
            + str(correct5)
            + " incorrect "
            + str(incorrect)
            + "\n",
        )

        print(
            "total = "
            + str(total)
            + " correct1 = "
            + str(correct1)
            + " correct2 = "
            + str(correct2)
            + " correct3 = "
            + str(correct3)
            + " correct4 = "
            + str(correct4)
            + " correct5 = "
            + str(correct5)
            + " incorrect "
            + str(incorrect)
            + "\n",
        )

    return

#only_model()
#description()

txt_dict = {}

for filename in os.listdir("data/csplib_masked"):
    if filename.endswith(".txt"):
        file_path = os.path.join("data/csplib_masked", filename)
        with open(file_path, "r", encoding="utf-8") as file:
            txt_dict[os.path.splitext(filename)[0]] = file.read()

index_path = "data/vector_dbs/ablation/description"
results_txt_path = "_results/figures/ablation/description/results.txt"
study(index_path=index_path, txt=txt_dict, results_txt_path=results_txt_path)
