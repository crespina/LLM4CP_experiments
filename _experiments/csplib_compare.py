import os
from langchain_groq import ChatGroq
import util
from pydantic import BaseModel as PydanticBaseModel
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.core import Settings, StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

os.environ["GROQ_API_KEY"] = "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"
#os.environ["GROQ_API_KEY"] = "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"


def load_index():
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

    Settings.show_progress = True

    if os.path.exists("data/vector_dbs/selected_db"):
        storage_context = StorageContext.from_defaults(
            persist_dir="data/vector_dbs/selected_db"
        )
        index = load_index_from_storage(storage_context, show_progress=True)
        print("Loaded index from storage.")
        return index
    else:
        print("Index storage directory not found. Parse and store the index first.")
        exit()


def csplib_compare(llm):

    csplib = {}
    # Iterate over every file in the directory
    for filename in os.listdir("data/csplib_corresp"):
        if filename.endswith(".txt"):
            # Get the file name without the '.txt' extension
            key = filename[:-4]
            file_path = os.path.join("data/csplib_corresp", filename)
            # Open and read the file content
            with open(file_path, "r", encoding="utf-8") as file:
                csplib[key] = file.read()

    index = load_index()
    reranker = CohereRerank(api_key="STPahNFoWeYX4FSAoMx7NzHNgH2ejINXLDKIYOr4", top_n=5)

    query_engine = index.as_query_engine(
        llm=llm, similarity_top_k=10, node_postprocessors=[reranker]
    )

    for name, descr in csplib.items():
        if (name == "shpping" or name == "langford"):

            response = query_engine.query(descr)
            retrieved1 = response.source_nodes[0].metadata["model_name"]
            retrieved2 = response.source_nodes[1].metadata["model_name"]
            retrieved3 = response.source_nodes[2].metadata["model_name"]
            retrieved4 = response.source_nodes[3].metadata["model_name"]
            retrieved5 = response.source_nodes[4].metadata["model_name"]

            print(name)
            print("---------")
            print(retrieved1,retrieved2,retrieved3,retrieved4, retrieved5)
            print("---------")
            print('\n')
            print('\n')


llm = ChatGroq(
    model = "llama-3.3-70b-specdec",
    temperature=0,
    timeout=None,
    max_retries=2,
)

csplib_compare(llm=llm)
