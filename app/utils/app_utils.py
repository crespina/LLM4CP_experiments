import os
import sys


def pprint_ranking(question, query, response):
    print(f"Question:\n{query}\n")
    print(f"Question modified:\n{question}\n")
    print("Ranking:")
    for ind, source_node in enumerate(response.source_nodes):
        print(f"Rank {ind + 1}:")
        print(f"Problem Name: {source_node.metadata['model_name']}")
        print(f"Similarity: {source_node.score}")
        print("_" * 50)


def get_input_safely(prompt):
    """Get user input safely, handling pasted text with potential newlines."""
    print(prompt, end='', flush=True)
    lines = []
    while True:
        line = sys.stdin.readline().rstrip('\n')
        if not line.strip() and not lines:
            continue
        lines.append(line)
        if len(lines) == 1 and not sys.stdin.isatty():
            break
        elif sys.stdin.isatty() and not os.name == 'nt':
            import select
            if not select.select([sys.stdin], [], [], 0.1)[0]:
                break
        else:
            break

    return '\n'.join(lines)
