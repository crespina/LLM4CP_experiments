import os
import time

from llama_index.core import Settings
from llama_index.core.callbacks import TokenCountingHandler, CallbackManager
from llama_index.llms.groq import Groq
from tqdm import tqdm

from app.utils.cache_utils import ExperimentCache


class Experiment:
    def __init__(self, args):
        self.model = Groq(
            model="llama3-70b-8192",
            model_kwargs={"seed": 42},
            api_key=args.groq_api_key,
            temperature=0.0,
        )
        self.model_tpm = 30_000

        self.output_dir = args.output_dir

        # Initialize cache
        self.cache = ExperimentCache(args)
        self.experiment_name = self.__class__.__name__.lower()

        # Will store problem rankings
        self.rankings = {}

        # Set of already processed problems to avoid recomputation
        self.processed_problems = set()

        self.token_counter = None
        self._setup_token_counter()

    def _setup_token_counter(self):
        """Set up token counting for rate limiting"""
        self.token_counter = TokenCountingHandler()
        callback_manager = CallbackManager([self.token_counter])
        Settings.callback_manager = callback_manager
        self.model.callback_manager = Settings.callback_manager

    def ranking(self, index, descriptions, result_path, k=5):
        """Perform ranking and save results with caching support"""
        query_engine = index.as_query_engine(
            llm=self.model, similarity_top_k=k
        )

        os.makedirs(os.path.dirname(result_path), exist_ok=True)

        # Load any previously processed problems
        index_name = os.path.basename(result_path).replace(".txt", "")
        cache_key = f"rankings_{index_name}"

        # Load existing rankings from cache if available
        cached_rankings = self.cache.load_cache(self.experiment_name, cache_key, default={})
        self.rankings[index_name] = cached_rankings

        # Determine which problems have already been processed
        processed_problems_key = f"processed_{index_name}"
        self.processed_problems = self.cache.load_cache(
            self.experiment_name, processed_problems_key, default=set()
        )

        # Write any existing cached results to file
        with open(result_path, "w") as f:
            for problem_name, top_results in cached_rankings.items():
                f.write(problem_name + " " + " ".join(top_results) + "\n")

        # Process remaining problems
        with open(result_path, "a") as f:
            problems_to_process = [(k, v) for k, v in descriptions.items() if k not in self.processed_problems]
            for problem_name, problem_descr in tqdm(problems_to_process, desc="Generating Answers"):
                start_time = time.time()
                response = query_engine.query(problem_descr)
                elapsed_time = time.time() - start_time

                # Rate limiting
                if self.token_counter.total_llm_token_count >= self.model_tpm:
                    sleep_time = 60 - elapsed_time
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                    self.token_counter.reset_counts()

                # Extract top 5 results
                top_results = [node.metadata["model_name"] for node in response.source_nodes[:5]]

                # Save to cache and file
                self.rankings[index_name][problem_name] = top_results
                self.processed_problems.add(problem_name)

                # Update cache after each problem (to handle potential crashes)
                self.cache.save_cache(self.experiment_name, cache_key, self.rankings[index_name])
                self.cache.save_cache(self.experiment_name, processed_problems_key, self.processed_problems)

                # Write to file
                f.write(problem_name + " " + " ".join(top_results) + "\n")

        return self.rankings[index_name]

    @staticmethod
    def save_results(results, output_file):
        """Save experiment results to file"""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "a") as f:
            for key, mrr in results.items():
                line = f"{key}, MRR = {mrr}\n"
                print(line.strip())
                f.write(line)

    def run(self):
        """Abstract method to run the experiment"""
        raise NotImplementedError("Subclasses must implement run()")

    @staticmethod
    def compute_mrr_from_rankings(rankings):
        """Compute MRR directly from rankings dictionary"""
        if not rankings:
            return 0

        reciprocal_ranks = []
        for problem_name, family_names in rankings.items():
            if problem_name in family_names:
                reciprocal_ranks.append(1 / (family_names.index(problem_name) + 1))
            else:
                reciprocal_ranks.append(0)

        return sum(reciprocal_ranks) / len(rankings) if rankings else 0
