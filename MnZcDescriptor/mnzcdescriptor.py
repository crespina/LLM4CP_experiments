import os
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from pathlib import Path
from langchain_core.prompts import ChatPromptTemplate
from llama_index.core.schema import TextNode
from llama_index.core import VectorStoreIndex
from llama_index.postprocessor.colbert_rerank import ColbertRerank
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import git
import pickle
import stat

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["LLAMA_CLOUD_API_KEY"] = "llx-69uKeyDcP4GtxQ9TaJkczA29jOlnzKSvuyu9HKlpuaRoNfx3"
os.environ["GROQ_API_KEY"] = "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"
# os.environ["OPENAI_API_KEY"] = "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")

class CPModel:
    #one document per piece of information and associate all docuemnts thorugh a tag ('model_name' = 'TSP')
    def __init__(self, cp_model_source, text_description=None, questions=None):
        self.cp_model_source = cp_model_source #Document llama idnex
        self.text_description = text_description
        self.questions = questions

class TextDescription(BaseModel):
    '''A description in English of a problem represented by a MiniZinc model'''
    name: str = Field(description="The name of the problem")
    description: str = Field(description="A description of the problem")
    variables: str = Field(description="All the decision variables in mathematical notation, followed by an explanation of what they are in English")
    constraints: str = Field(description="All the constraints in mathematical notation only")
    objective: str = Field(description="The objective of the problem (minimize or maximize what value)")

class Questions(BaseModel):
    """Situations or problems that a user could be facing that would be modelled as the given described model"""
    question1: str = Field(
        description="A question/scenario that is from a user very skilled in modelling and solving constraint problems"
    )
    question2: str = Field(
        description="A question/scenario that is from a user that knows nothing about formal modelling and solving constraint problems"
    )
    question3: str = Field(description="A question/scenario that is from a young user")
    question4: str = Field(description="A question/scenario that is very short")
    question5: str = Field(
        description="A question/scenario that is very long and specific"
    )


def fetch(repo_url, local_repo_dir="MnZcDescriptor/temp_repo", output_dir="MnZcDescriptor/models_mzn"):

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

                with open("MnZcDescriptor/models_mzn/"+file, "r") as mzn_file:
                    content = mzn_file.read()
                    txt_file_path = os.path.join(
                        "MnZcDescriptor/models_mzn", file.replace(".mzn", ".txt")
                    )
                # Write the content to the .txt file
                with open(txt_file_path, "w") as txt_file:
                    txt_file.write(content)      

    print("Conversion completed: All .mzn files have been converted to .txt.")


def convert(instances, folder_name):

    folder_path = Path(folder_name)

    for file_path in folder_path.glob("*.txt"):
        file_name = file_path.stem
        with file_path.open('r') as file:
            file_content = file.read()
            cpmodel = CPModel(cp_model_source = file_content)
            instances[file_name] = cpmodel


def create_text_description(llm, instances):

    prompt_text_description = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You are an expert in high-level constraint modelling and solving discrete optimization problems.
                In particular, you know Minizinc.
                You are provided with a Minizinc model that represents a classical problem in constraint programming.
                Your task is to identify what is the problem modelled and give a complete description of the problem to the user.
                The format of the answer should be without any variation a JSON-like format with the following keys and explanation of what the corresponding values should be:
                    name: The name of the problem
                    description: A description of the problem in English
                    variables: A string containing the list of all the decision variables in mathematical notation, followed by an explanation of what they are in English
                    constraints: A string containing the list of all the constraints in mathematical notation, followed by an explanation of what they are in English
                    objective: The objective of the problem (minimize or maximize what value)
                """,
            ),
            ("human", "{input}"),
        ]
    )

    structured_llm_text_description = llm.with_structured_output(TextDescription, method="json_mode")

    for (key, value) in instances.items():
        chain = prompt_text_description | structured_llm_text_description
        input = value.cp_model_source
        aimessage = chain.invoke(
            {
                "input": input
            }
        )
        value.text_description = aimessage


def create_questions(llm, instances):
    # QuestionsAnsweredExtractor llama index
    #can add own metadata : exemple family (i can even do it as a tree -> hierchal)

    prompt_questions = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                "Given the following description of a constraint problem,
                generate five realistic and practical user questions or scenarios that would be naturally answered by solving the problem
                but do not necessarily use the traditional or classical context of the problem. Think beyond the usual applications—use creative analogies or different contexts
                The questions should incorporate real-life constraints, preferences, and priorities that reflect the problem's structure.
                For example, focus on specific goals the user wants to achieve, the constraints they face, and the trade-offs they might need to consider.
                The questions should never incorporate the name of the given problem.
                You can decide to incorporate numeric dummy data into the questions.

                The first question/scenario should be from a user very skilled in modelling and solving constraint problems.
                The second question/scenario should be from a user that knows nothing about formal modelling and solving constraint problems.
                The third question/scenario should be from a young user.
                The fourth question/scenario should be very short
                The fifth question/scenario should be very long and specific.

                Problem Description: {description}
                """,
            )
        ]
    )

    structured_llm_questions = llm.with_structured_output(Questions)

    for (key, value) in instances.items():
        chain = prompt_questions | structured_llm_questions
        aimessage = chain.invoke(
            {
                "description": value.text_description
            }
        )
        value.questions = aimessage

def create_index(instances):
    #quesry the embedding model directly, no  index
    counter = 0
    nodes = []

    for (key, value) in instances.items():
        for question in value.questions :
            node = TextNode(text=question[1], id_=key+"_"+str(counter),metadata_template=None)
            nodes.append(node)
            counter+=1

    index = VectorStoreIndex(nodes, embed_model=HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5"))
    return index

def rerank(query, index, llm):
    #not really necessary

    colbert_reranker = ColbertRerank(
        #params to be modified
        top_n=5,
        model="colbert-ir/colbertv2.0",
        tokenizer="colbert-ir/colbertv2.0",
        keep_retrieval_score=True,
    )
    query_engine = index.as_query_engine(
        similarity_top_k=10,
        node_postprocessors=[colbert_reranker],
        llm = llm
    )
    response = query_engine.query(
        query,
    )
    print("----------------------------")
    for node in response.source_nodes:
        print(node.id_)
        print(node.node.get_content()[:120])
        print("reranking score: ", node.score)
        print("retrieval score: ", node.node.metadata["retrieval_score"])
        print("**********")
    print(response)

def confusion_matrix(index, instances):
    # also questions x code, questions x description, question x code+string (join string)
    vectors = index._vector_store._data.embedding_dict
    num_categories = 5
    texts_per_category = 5

    # Step 1: Extract the embeddings and group them by category
    embeddings = np.array(list(vectors.values()))

    # Step 2: Average embeddings by category
    category_embeddings = []
    category_labels = []
    for i in range(num_categories):
        # Extract embeddings for this category
        category_embs = embeddings[
            i * texts_per_category : (i + 1) * texts_per_category
        ]

        # Average the embeddings for the current category
        avg_embedding = np.mean(category_embs, axis=0)

        # Store the average embedding and the category label
        category_embeddings.append(avg_embedding)

    # Convert to NumPy array for cosine similarity
    for key, value in instances.items():
        category_labels.append(key)
    category_embeddings = np.array(category_embeddings)

    # Step 3: Compute the cosine similarity matrix between categories
    similarity_matrix = cosine_similarity(category_embeddings)

    # Step 4: Plot the confusion matrix using a heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        similarity_matrix,
        annot=True,
        cmap="coolwarm",
        xticklabels=category_labels,
        yticklabels=category_labels,
    )
    plt.title("Cosine Similarity between Categories")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.show()


def save_instances(savename, instances):

    save_path = "MnZcDescriptor\instances\\" + savename + ".pkl"
    with open(save_path, "wb") as file:
        pickle.dump(instances, file)

    print(f"Pickle file saved to {save_path}")


def load_instances(savename):

    filename = "MnZcDescriptor\instances\\" + savename + ".pkl"

    with open(filename, "rb") as file:
        instances = pickle.load(file)

    print("Instances loaded")
    return instances


def instantiate():

    fetch("https://github.com/MiniZinc/minizinc-examples.git")
    convert_mzn_to_txt()

def pre_process():

    instances = {}

    llm = ChatGroq(
        model="llama-3.1-70b-versatile",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )

    convert(instances, "MnZcDescriptor\models_mzn")
    create_text_description(llm, instances)
    create_questions(llm, instances)
    save_instances("final", instances)


# fetch("https://github.com/MiniZinc/minizinc-examples.git")
# convert_mzn_to_txt()
# load_instances()


# create_index()
# confusion_matrix()
# rerank("My car trunk has a space of 3 m3, and i would like to take surf plank that would bring me much joy but takes 2 m3, some sand that i would be a little bit happy to have and that takes 0.3 m3")

pre_process()

instances = load_instances("final")
print(instances["knapsack"].questions)
