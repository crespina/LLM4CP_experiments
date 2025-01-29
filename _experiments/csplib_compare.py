import json
import os
import pickle
import re
import time
from llama_index.llms.ollama import Ollama
from typing import Sequence
from llama_index.core import Document
from langchain_groq import ChatGroq
from pathlib import Path
from llama_index.core import ChatPromptTemplate
from llama_index.core.llms import ChatMessage
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.core import Settings
import numpy as np
import util
import copy
from llama_index.core.indices.query.schema import QueryBundle
from llama_index.core.retrievers import BaseRetriever, VectorIndexRetriever
import sys
import cohere
import pydantic
from pydantic import BaseModel as PydanticBaseModel
from pydantic import Field

os.environ["GROQ_API_KEY"] = "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"

class BaseModel(PydanticBaseModel):
    class Config:
        arbitrary_types_allowed = True


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


def embedding(instances):

    for doc in instances.values():
        if "embedding_vector" in doc.metadata.keys():
            del doc.metadata["embedding_vector"]

    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
    documents: Sequence[Document] = [content for content in instances.values()]

    # Settings.chunk_size = 2048
    vector_index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)

    vector_retriever = VectorIndexRetriever(
        index=vector_index,
        similarity_top_k=5,
        embed_model=embed_model,
    )

    print("embedding done")
    return vector_retriever


def rank(query, vector_retriever, llm):

    query_templated = QueryBundle(query)
    retrieved_nodes = vector_retriever.retrieve(query_templated)
    return retrieved_nodes


def csplib_compare(instances, llm):

    vector_retriever = embedding(instances)

    with open("data\csplib_corresp\my_dict.pkl", "rb") as f:
        cpslib = pickle.load(f)

    for name, descr in cpslib.items():

        response = rank(query=descr, vector_retriever=vector_retriever, llm=llm)
        retrieved1 = response[0].metadata["model_name"]
        retrieved2 = response[1].metadata["model_name"]
        retrieved3 = response[2].metadata["model_name"]
        retrieved4 = response[3].metadata["model_name"]
        retrieved5 = response[4].metadata["model_name"]

        print(name)
        print("---------")
        print(retrieved1,retrieved2,retrieved3,retrieved4, retrieved5)
        print("---------")
        print('\n')
        print('\n')


path = "data\model_checkpoints\llama32_90b_both_base_embedding.pkl"
instances = util.load_instances(path)

llm = ChatGroq(
    model = "llama-3.3-70b-versatile",
    temperature=0,
    timeout=None,
    max_retries=2,
)

csplib_compare(instances=instances, llm=llm)
