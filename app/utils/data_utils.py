from tqdm import tqdm
from transformers import AutoTokenizer

from app.data_processing.data_loaders import load_index


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
