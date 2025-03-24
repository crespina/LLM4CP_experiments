import tiktoken
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler
import os
import time
from transformers import AutoModel, AutoTokenizer

from llama_index.core import Settings, StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.core.callbacks import TokenCountingHandler, CallbackManager

from tqdm import tqdm


def load_index(index_path):
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

    Settings.show_progress = True

    if os.path.exists(index_path):
        storage_context = StorageContext.from_defaults(persist_dir=index_path)
        index = load_index_from_storage(storage_context, show_progress=True)
        print("Loaded index from storage.")
        return index
    else:
        print("Index storage directory not found. Parse and store the index first.")
        exit()


model_path = "Alibaba-NLP/gte-modernbert-base"
tokenizer = AutoTokenizer.from_pretrained(model_path)

token_counter = TokenCountingHandler(tokenizer=tokenizer)

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

max_tok = 0
for index_level in indexes:

    index_path = "data/vector_dbs/code_as_text/" + index_level
    index = load_index(index_path)
    nodes = index.docstore.docs.values()
    for i, node in enumerate(nodes):
        # Extract the text content of the node
        node_text = node.get_text()

        # Tokenize the text to get the number of tokens
        num_tokens = len(tokenizer(node_text)["input_ids"])

        # Output the token count for the node
        print(f"Node {i + 1} contains {num_tokens} tokens.")
        if num_tokens > max_tok:
            max_tok = num_tokens
        token_counter.reset_counts()
print(max_tok)
