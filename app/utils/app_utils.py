import sys
import time
from functools import wraps


def pprint_ranking(response):
    print("Ranking:")
    for ind, source_node in enumerate(response.source_nodes):
        print(f"Rank {ind + 1}:")
        print(f"Problem Name: {source_node.metadata['model_name']}")
        print(f"Similarity: {source_node.score}")
        print("_" * 50)


def get_input_safely(prompt):
    """Get user input safely, handling multiline text input.
    
    Users can enter multiple lines and terminate input by pressing Enter twice at the end.
    """
    print(prompt, end='', flush=True)

    lines = []

    if sys.stdin.isatty():
        # Interactive terminal mode
        while True:
            line = input()
            if line.strip() == "":
                break
            lines.append(line)
    else:
        # Non-interactive mode (piped input)
        for line in sys.stdin:
            line = line.rstrip('\n')
            if line.strip() == "":
                break
            lines.append(line)

    user_input = '\n'.join(lines)

    return user_input


def throttle_requests():
    """
    Decorator to throttle requests based on token usage.

    Returns:
        function: Wrapped function with throttling.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            start_time = time.time()
            result = func(self, *args, **kwargs)
            elapsed_time = time.time() - start_time
            if self.token_counter.total_llm_token_count >= self.model_tpm:
                sleep_time = 60 - elapsed_time
                if sleep_time > 0:
                    time.sleep(sleep_time)
                self.token_counter.reset_counts()
            return result

        return wrapper

    return decorator
