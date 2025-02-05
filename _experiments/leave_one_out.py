import json
import os
import re
import time
from typing import Sequence
from pydantic import BaseModel, Field
from llama_index.core import Document
from langchain_groq import ChatGroq
from pathlib import Path
from llama_index.core import ChatPromptTemplate
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.core import Settings
import util
import copy
from llama_index.core.indices.query.schema import QueryBundle
from llama_index.core.retrievers import BaseRetriever, VectorIndexRetriever

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["LLAMA_CLOUD_API_KEY"] = (
    "llx-69uKeyDcP4GtxQ9TaJkczA29jOlnzKSvuyu9HKlpuaRoNfx3"
)
os.environ["GROQ_API_KEY"] = "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"
# os.environ["OPENAI_API_KEY"] = "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")


class TextDescription(BaseModel):
    """A description in English of a problem represented by a MiniZinc model"""
    name: str = Field(description="The name of the problem")
    description: str = Field(description="A description of the problem")
    variables: str = Field(
        description="All the decision variables in mathematical notation, followed by an explanation of what they are in English"
    )
    constraints: str = Field(
        description="All the constraints in mathematical notation only"
    )
    objective: str = Field(
        description="The objective of the problem (minimize or maximize what value)"
    )


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


def replace(string, dict):
    for family_name, problems in dict.items():
        for problem in problems:
            if problem == string:
                string = string.replace(problem, family_name)
    return string

global families
families = {
    "assign": ["assign", "assign_dual", "assign_inverse"],
    "aust_color": ["aust_color", "aust_colord"],
    "buggy": [
        "context",
        "debug1",
        "debug2",
        "debug3",
        "division",
        "evenproblem",
        "missing_solution",
        "test",
        "trace",
    ],
    "carpet_cutting": ["carpet_cutting", "carpet_cutting_geost", "cc_geost"],
    "cell_block": ["cell_block", "cell_block_func"],
    "cluster": ["cluster"],
    "compatible_assignment": ["compatible_assignment", "compatible_assignment_opt"],
    "constrained_connected": ["constrained_connected"],
    "crazy_sets": ["crazy_sets", "crazy_sets_global"],
    "doublechannel": ["doublechannel"],
    "flattening": [
        "flattening0",
        "flattening1",
        "flattening2",
        "flattening3",
        "flattening4",
        "flattening5",
        "flattening6",
        "flattening7",
        "flattening8",
        "flattening9",
        "flattening10",
        "flattening11",
        "flattening12",
        "flattening13",
        "flattening14",
    ],
    "jobshop": ["jobshop", "jobshop2", "jobshop3"],
    "knapsack": [
        "knapsack",
        "knapsack01",
        "knapsack01bool",
        "knapsack01set",
        "knapsack01set_concise",
    ],
    "langford": ["combinedlangford", "inverselangford", "langford"],
    "linetsp": ["ltsp"],
    "loan": ["loan1", "loan2"],
    "mip": ["mip1", "mip2", "mip3", "mip4", "mip5"],
    "nurses": ["nurses", "nurses_let"],
    "photo": ["photo"],
    "production_planning": ["simple-prod-planning"],
    "project_scheduling": ["project_scheduling", "project_scheduling_nonoverlap"],
    "rcpsp": ["rcpsp"],
    "rel_sem": ["rel_sem", "rel_sem2"],
    "restart": ["restart", "restart2", "restarta"],
    "search": ["assign", "domwdeg"],
    "setselect": [
        "setselect",
        "setselectr",
        "setselectr2",
        "setselectr3",
        "setselect2",
        "setselect3",
    ],
    "shipping": ["shipping"],
    "stable_roommates": ["stableroommates", "stable_roommates_func"],
    "submultisetsum": ["submultisetsum"],
    "table_seating": ["table_seating", "table_seating_gcc"],
    "team_select": ["teamselect", "teamselect_advanced"],
    "array_quest": ["array_quest"],
    "graph": ["graph"],
    "itemset_mining": ["itemset_mining"],
    "lots": ["lots"],
    "missingsolution": ["missingsolution"],
    "myabs": ["myabs"],
    "mydiv": ["mydiv"],
    "queens": ["queens"],
    "square_pack": ["square_pack"],
    "table_example": ["table_example"],
    "toomany": ["toomany"],
    "toy_problem": ["toy_problem"],
}

def embedding(instances):

    for doc in instances.values():
        if "embedding_vector" in doc.metadata.keys():
            del doc.metadata["embedding_vector"]

    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
    documents: Sequence[Document] = [content for content in instances.values()]

    Settings.chunk_size = 2048
    vector_index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)

    vector_retriever = VectorIndexRetriever(
        index=vector_index,
        similarity_top_k=20,
        embed_model=embed_model,
    )

    """
    vectors = index._vector_store._data.embedding_dict
    embeddings = np.array(list(vectors.values()))
    for doc, vector in zip(instances.values(), embeddings):
        doc.metadata["embedding_vector"] = vector
    """
    print("embedding done")
    return vector_index, vector_retriever


def rank(query, vector_index, vector_retriever, llm):

    """
    reranker = CohereRerank(api_key="bk0m8vejhUPHMYwrtrBtL06juj0HDC0d3wGkSXOj", top_n=5)

    query_engine = vector_index.as_query_engine(
        llm=llm, similarity_top_k=5, node_postprocessors=[reranker]
    )

    query_templated = QueryBundle(query)

    response = query_engine.query(
        query_templated,
    )

    time.sleep(3)
    return response
    """

    query_templated = QueryBundle(query)
    retrieved_nodes = vector_retriever.retrieve(query_templated)
    reranker = CohereRerank(api_key="STPahNFoWeYX4FSAoMx7NzHNgH2ejINXLDKIYOr4", top_n=5)
    reranked_nodes = reranker.postprocess_nodes(retrieved_nodes, query_bundle=query_templated)
    time.sleep(6.5)
    return reranked_nodes


def create_last_question(llm, instances):

    for key, value in instances.items():

        input = value.metadata["text_description"].__str__()

        chat_text_qa_msgs = [
            (
                "system",
                """
                You are provided with a description of a constraint problem.
                Generate a question that would be answered by said model. 
                """,
            ),
            ("user", input),
        ]

        text_qa_template = ChatPromptTemplate.from_messages(chat_text_qa_msgs)

        prompt = text_qa_template.format(input=input)
        aimessage = llm.invoke(prompt)
        value.metadata["last_q"] = aimessage.content
    util.save_instances(
        "MnZcDescriptor\data\model_checkpoints\leave_one_out_full", instances
    )


def leave_one_out(instances, llm):  
     
    for key,value in instances.items() :
        del value.metadata["last_q"]

    vector_index, vector_retriever = embedding(instances)

    
    hot_llm = llm = ChatGroq(
        model="llama-3.2-90b-text-preview",
        temperature=0.5,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )

    create_last_question(hot_llm,instances)
    

    total = 0
    correct1 = 0
    correct2 = 0
    correct3 = 0
    correct4 = 0
    correct5 = 0
    incorrect = 0
    count = 0

    # for name, question in last_question.items() :
    for key, val in instances.items() :
        count +=1
        name = replace(key, families)
        if (count < 73):
            pass
        else : 
            question = val.metadata["last_q"]
            response = rank(query=question, vector_index=vector_index, vector_retriever= vector_retriever,llm=llm)
            retrieved1 = replace(response.source_nodes[0].metadata["model_name"], families)
            retrieved2 = replace(response.source_nodes[1].metadata["model_name"], families)
            retrieved3 = replace(response.source_nodes[2].metadata["model_name"], families)
            retrieved4 = replace(response.source_nodes[3].metadata["model_name"], families)
            retrieved5 = replace(response.source_nodes[4].metadata["model_name"], families)

            if (name != retrieved1) :
                if (name != retrieved2) :
                    if (name != retrieved3) :
                        if (name != retrieved4) :
                            if (name != retrieved5) :
                                incorrect +=1
                            else : 
                                correct5 += 1 
                        else : 
                            correct4 +=1
                    else : 
                        correct3 +=1
                else : 
                    correct2 += 1
            else : 
                correct1 += 1

            total += 1
            print(name, retrieved1, retrieved2, retrieved3, retrieved4, retrieved5)

    print(
            "total = " + str(total),
            "correct1 = " + str(correct1),
            "correct2 = " + str(correct2),
            "correct3 = " + str(correct3),
            "correct4 = " + str(correct4),
            "correct5 = " + str(correct5),
            "incorrect " + str(incorrect)
            )


def leave_one_out_5_questions(instances, llm):

    out = ["question1"]

    for qout in out : 
        print("start", qout)
        instances_copy = copy.deepcopy(instances)

        last_question = {}
        for key, value in instances_copy.items():
            last_question[key] = value.metadata[qout]
            del value.metadata[qout]
            del value.metadata["last_q"]

        vector_index, vector_retriever = embedding(instances_copy)

        total = 0
        correct1 = 0
        correct2 = 0
        correct3 = 0
        correct4 = 0
        correct5 = 0
        incorrect = 0

        for name, question in last_question.items() :
        #for key, val in instances_copy.items():
            name = replace(name, families)
            #question = val.metadata["last_q"]
            response = rank(query=question, vector_index=vector_index, vector_retriever=vector_retriever, llm=llm)
            retrieved1 = replace(
                response[0].metadata["model_name"], families
            )
            retrieved2 = replace(
                response[1].metadata["model_name"], families
            )
            retrieved3 = replace(
                response[2].metadata["model_name"], families
            )
            retrieved4 = replace(
                response[3].metadata["model_name"], families
            )
            retrieved5 = replace(
                response[4].metadata["model_name"], families
            )
            if name != retrieved1:
                if name != retrieved2:
                    if name != retrieved3:
                        if name != retrieved4:
                            if name != retrieved5:
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
            print(name, retrieved1, retrieved2, retrieved3, retrieved4, retrieved5)

        print(
            "total = " + str(total),
            "correct1 = " + str(correct1),
            "correct2 = " + str(correct2),
            "correct3 = " + str(correct3),
            "correct4 = " + str(correct4),
            "correct5 = " + str(correct5),
            "incorrect " + str(incorrect),
        )

path = "MnZcDescriptor\data\model_checkpoints\leave_one_out_full.pkl"
instances = util.load_instances(path)

llm = ChatGroq(
    # model="llama-3.2-90b-text-preview",
    #model="llama3-70b-8192",
    model = "llava-v1.5-7b-4096-preview",
    temperature=0,
    max_tokens=100,
    timeout=None,
    max_retries=2,
)

#leave_one_out_5_questions(instances=instances, llm=llm)
print("lol")