import json
import os
import time
from llama_index.llms.ollama import Ollama
from typing import Sequence
from pydantic import BaseModel, Field
from llama_index.core import Document
from langchain_groq import ChatGroq
from pathlib import Path
from llama_index.core import ChatPromptTemplate
from llama_index.core.llms import ChatMessage
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
import numpy as np
import util

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["LLAMA_CLOUD_API_KEY"] = "llx-69uKeyDcP4GtxQ9TaJkczA29jOlnzKSvuyu9HKlpuaRoNfx3"
os.environ["GROQ_API_KEY"] = "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"
# os.environ["OPENAI_API_KEY"] = "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")


class TextDescription(BaseModel):
    '''A description in English of a problem represented by a MiniZinc model'''
    name: str = Field(description="The name of the problem")
    description: str = Field(description="A description of the problem")
    variables: str = Field(description="All the decision variables in mathematical notation, followed by an explanation of what they are in English")
    constraints: str = Field(description="All the constraints in mathematical notation only")
    objective: str = Field(description="The objective of the problem (minimize or maximize what value)")

class Questions(BaseModel):
    """Situations or problems that a user could be facing that would be modelled as the given described model"""
    question1: str = Field(description="A question/scenario that is from a user very skilled in modelling and solving constraint problems")
    question2: str = Field(description="A question/scenario that is from a user that knows nothing about formal modelling and solving constraint problems")
    question3: str = Field(description="A question/scenario that is from a young user")
    question4: str = Field(description="A question/scenario that is very short")
    question5: str = Field(description="A question/scenario that is very long and specific")


def convert_txt_to_Document(instances, folder_name):

    folder_path = Path(folder_name)

    for file_path in folder_path.glob("*.txt"):
        file_name = file_path.stem
        with file_path.open("r") as file:
            file_content = file.read()

            cpmodel = Document(
                text=file_content,
                metadata={
                    "model_name": file_name,
                },
            )

            instances[file_name] = cpmodel

    return instances


def create_text_description(llm, instances):

    if (type(llm) == ChatGroq):

        structured_llm_text_description = llm.with_structured_output(TextDescription, method="json_mode")

        for (key, value) in instances.items():

            input = value.text

            chat_text_qa_msgs = [
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
                ("user", input),
            ]

            text_qa_template = ChatPromptTemplate.from_messages(chat_text_qa_msgs)

            prompt = text_qa_template.format(input=input)
            aimessage = structured_llm_text_description.invoke(prompt)
            value.metadata["text_description"] = aimessage

    elif type(llm) == Ollama :      

        for (key, value) in instances.items():
            start_time = time.time()

            input = value.text

            aimessage = llm.chat(
                [
                    ChatMessage(
                        role="system",
                        content="""
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
                    ChatMessage(role="user", content=input),
                ]
            )

            response = aimessage.message.content
            print(response)

            try:

                response = response[7:-3]  # remove the first 7 characters and last 3
                parsed_response = json.loads(response)

                value.metadata["text_description"] = TextDescription(
                    name=str(parsed_response["name"]),
                    description=str(parsed_response["description"]),
                    variables=str(parsed_response["variables"]),
                    constraints=str(parsed_response["constraints"]),
                    objective=str(parsed_response["objective"]),
                )

            except json.JSONDecodeError:
                # Handle cases where the response is not valid JSON
                print(
                    f"Failed to parse response as JSON for instance {key}. Response: {aimessage}"
                )
                value.metadata["text_description"] = None

            end_time = time.time()
            execution_time = end_time - start_time
            print(f"Execution time: {execution_time} seconds")

    util.save_instances("llama32_90b", instances)

def create_questions(llm, instances):
    # QuestionsAnsweredExtractor llama index
    # can add own metadata : exemple family (i can even do it as a tree -> hierchal)

    structured_llm_questions = llm.with_structured_output(Questions, method="json_mode")

    for (key, value) in instances.items():

        input = value.metadata["text_description"].__str__()

        chat_text_qa_msgs = [
            (
                "system",
                """
                You are provided with a description of a constraint problem.
                Generate five realistic and practical user questions or scenarios that would be naturally answered by solving the problem
                but do not necessarily use the traditional or classical context of the problem. Think beyond the usual applications—use creative analogies or different contexts
                The questions should incorporate real-life constraints, preferences, and priorities that reflect the problem's structure.
                For example, focus on specific goals the user wants to achieve, the constraints they face, and the trade-offs they might need to consider.
                The questions should never incorporate the name of the given problem.
                You can decide to incorporate numeric dummy data into the questions.

                The format of the answer should be without any variation a JSON with the following keys :
                question1 : The first question/scenario should be from a user very skilled in modelling and solving constraint problems.
                question2 : The second question/scenario should be from a user that knows nothing about formal modelling and solving constraint problems.
                question3 : The third question/scenario should be from a young user.
                question4 : The fourth question/scenario should be very short
                question5 : The fifth question/scenario should be very long and specific.
                """,
            ),
            ("user", input),
        ]

        text_qa_template = ChatPromptTemplate.from_messages(chat_text_qa_msgs)

        prompt = text_qa_template.format(input=input)
        aimessage = structured_llm_questions.invoke(prompt)
        print(aimessage)
        value.metadata["question1"] = aimessage.question1
        value.metadata["question2"] = aimessage.question2
        value.metadata["question3"] = aimessage.question3
        value.metadata["question4"] = aimessage.question4
        value.metadata["question5"] = aimessage.question5


def embedding(instances):

    for doc in instances.values() :
        if 'embedding_vector' in doc.metadata.keys():
            del doc.metadata['embedding_vector']

    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    documents: Sequence[Document] = [content for content in instances.values()]

    Settings.chunk_size = 2048
    index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)

    vectors = index._vector_store._data.embedding_dict
    embeddings = np.array(list(vectors.values()))
    for doc, vector in zip(instances.values(), embeddings):
        doc.metadata["embedding_vector"] = vector

    return index


def rank(query, index, llm):

    query_engine = index.as_query_engine(
        similarity_top_k=5,
        llm = llm
    )
    response = query_engine.query(
        query,
    )
    print("----------------------------")
    for node in response.source_nodes:
        print(node.id_)
        print(node.node.get_content()[:120])
        print("retrieval score: ", node.node.metadata["retrieval_score"])
        print("**********")
    print(response)

def pre_process(folder_name, instance_name):

    instances = {}

    # curl -fsSL https://ollama.com/install.sh |sh
    # ollama serve & ollama pull qwen2.5-coder
    llm_code = Ollama(
        model="qwen2.5-coder",
        temperature=0,
        request_timeout=600.0
        
    )

    llm_questions = ChatGroq(
        model="llama-3.2-90b-text-preview",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )

    convert_txt_to_Document(instances, folder_name)
    create_text_description(llm_code, instances)
    create_questions(llm_questions, instances)
    util.save_instances(instance_name, instances)


# fetch("https://github.com/MiniZinc/minizinc-examples.git")
# convert_mzn_to_txt()
# load_instances()


# create_index()
# confusion_matrix()
# rerank("My car trunk has a space of 3 m3, and i would like to take surf plank that would bring me much joy but takes 2 m3, some sand that i would be a little bit happy to have and that takes 0.3 m3")

# pre_process()

# model_checkpoints = load_instances("final")
# print(model_checkpoints["knapsack"].questions)

#pre_process("MnZcDescriptor\\test_models", "qwen25_small_test")
instances = util.load_instances("MnZcDescriptor\data\model_checkpoints\llama32_90b_both_base_embedding")
#embedding(instances)
#util.save_instances("qwen25_small_test", instances)

print(instances["flattening10"].metadata["text_description"])
# print(model_checkpoints["knapsack"].metadata["embedding_vector"])
