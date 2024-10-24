import pickle

from llama_index.core import Settings, StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from app.engine.utils.app_utils import pprint_console


def load_document(args):
    doc_path = f"{args.output_path}/{args.persistent_dir}/{args.doc_name}/_doc"
    with open(f"{doc_path}/doc.pickle", 'rb') as pkl_file:
        document = pickle.load(pkl_file)

    Settings.embed_model = HuggingFaceEmbedding(model_name=document.embedding_model_name, trust_remote_code=True)
    index_path = f"{args.output_path}/{args.persistent_dir}/{args.doc_name}/_idx"
    storage_context = StorageContext.from_defaults(persist_dir=index_path)
    index = load_index_from_storage(storage_context)

    pprint_console(f"Document path = {doc_path}")
    pprint_console(f"Index dir = {index_path}")

    return document, index
