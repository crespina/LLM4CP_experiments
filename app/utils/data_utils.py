import os

from langchain.output_parsers import ResponseSchema
from tqdm import tqdm
from transformers import AutoTokenizer

from app.data_processing.data_loaders import load_index
from app.utils.CONSTANTS import FAMILIES


def convert_mzn_to_txt(mzn_path, txt_path):
    for dir_path, _, filenames in os.walk(mzn_path):
        for filename in tqdm(filenames, desc="Converting .mzn to .txt"):
            if filename.endswith(".mzn"):
                mzn_file_path = os.path.join(dir_path, filename)
                with open(mzn_file_path, 'r') as mzn_file:
                    content = mzn_file.read()

                relative_path = os.path.relpath(dir_path, mzn_path)
                output_dir_path = os.path.join(txt_path, relative_path)

                os.makedirs(output_dir_path, exist_ok=True)

                txt_file_path = os.path.join(output_dir_path, filename.replace('.mzn', '.txt'))

                with open(txt_file_path, 'w') as txt_file:
                    txt_file.write(content)


def get_response_schema(query_config_dict):
    response_schema = []
    for field, description in query_config_dict.items():
        response_schema.append(ResponseSchema(name=field.strip(), description=description.strip()))
    return response_schema


def problem_family(problem_name):
    # Given the name of the problem, returns its family

    for family_name, problems in FAMILIES.items():
        for problem in problems:
            if problem == problem_name:
                problem_name = problem_name.replace(problem, family_name)

    return problem_name


def count_tokens(model_name, args):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    try:
        index = load_index(args)
    except Exception as e:
        print(f"Error loading index from {args.storage_dir}: {e}")
        return None

    nodes = index.docstore.docs.values()

    max_tokens = 0
    total_tokens = 0
    per_node_tokens = []

    for i, node in enumerate(tqdm(nodes, desc="Processing nodes")):
        node_text = node.get_text()
        num_tokens = len(tokenizer(node_text)["input_ids"])

        per_node_tokens.append((i, num_tokens))
        total_tokens += num_tokens
        if num_tokens > max_tokens:
            max_tokens = num_tokens

    # Prepare the results
    results = {
        "max_tokens": max_tokens,
        "total_tokens": total_tokens,
        "avg_tokens": total_tokens / len(nodes) if nodes else 0,
        "node_count": len(nodes),
        "per_node_tokens": per_node_tokens
    }

    if results:
        print(f"Max tokens in any node: {results['max_tokens']}")
        print(f"Total tokens across all nodes: {results['total_tokens']}")
        print(f"Average tokens per node: {results['avg_tokens']:.2f}")
        print(f"Number of nodes analyzed: {results['node_count']}")

        return results

    else:
        print("No nodes found in the index.")
        exit(1)
