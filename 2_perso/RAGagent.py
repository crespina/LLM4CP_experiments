#!pip install -U langchain-nomic langchain_community tiktoken langchainhub chromadb langchain langgraph tavily-python gpt4all firecrawl-py
#!pip install llama-parse llama-index llama-index-postprocessor-sbert-rerank
#!python3 -m pip install -U mypy

import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from llama_parse import LlamaParse
from typing import List
import asyncio

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["LLAMA_CLOUD_API_KEY"] = (
    "llx-69uKeyDcP4GtxQ9TaJkczA29jOlnzKSvuyu9HKlpuaRoNfx3"
)
local_llm = "llama3.1:cpu"


async def load_pdf() -> List[Document]:
    parser = LlamaParse(result_type="markdown",)
    documents = await parser.aload_data("2_perso/paper.pdf")
    return documents

# TODO : metadata tagging

def text_splitter(documents : List[Document]) -> List[Document]:
    # Assuming each tuple in docs_list is of the form (content, metadata)
    docs_list = [Document(id = item.id_, page_content=item.text) for item in documents]

    # Now use the text splitter as intended
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(docs_list)

    return texts

def create_retriever(texts : List[Document]) -> VectorStoreRetriever :

    vectorstore = Chroma.from_documents(
        documents = texts,
        collection_name = "rag-chroma",
        embedding = GPT4AllEmbeddings()
        )

    retriever = vectorstore.as_retriever()

    return retriever


def retrieval_grader(retriever: VectorStoreRetriever) :

    llm = ChatOllama(model=local_llm, format="json",temperature=0)

    prompt = PromptTemplate(
        template=
        """
    <|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing relevance of a retrieved document to a user question. If the document contains keywords related to the user question, grade it as relevant. It does not need to be a stringent test. The goal is to filter out erronous retrievals. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
    Provide the binary score as a JSON with a single key 'score' and no premable or explanation.
    <|eot_id|><|start_header_id|>user<|end_header_id|>
    Here is the retrived document : \n\n{document}
    Here is the user question : \n\n{question} \n <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """,
    input_variables=["question", "document"],
    )

    retrival_grader = prompt | llm | JsonOutputParser()
    question = "How do you train a LLM ?"
    docs = retriever.invoke(question)
    doc_txt = docs[1].page_content
    print(retrival_grader.invoke({"question": question, "document": doc_txt}))


if __name__ == "__main__":
    documents = asyncio.run(load_pdf())
    texts = text_splitter(documents)
    retriever = create_retriever(texts)
    retrieval_grader(retriever)
