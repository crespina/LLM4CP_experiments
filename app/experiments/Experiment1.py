from app.experiments.Experiment import Experiment
from app.data_processing.data_loaders import load_index
from app.utils.CONSTANTS import LEVELS, INDICES
from app.utils.exp_utils import retrieve_descriptions, compute_mrr


class Experiment1(Experiment):
    def __init__(self, args):
        super().__init__(args)
        self.descr_folder = args.descriptions_dir
        self.results_dir = f"{args.results_dir}/exp1"

        self.args = args

    def run(self):
        """Run Experiment 1 with caching support"""
        # Load cached results if available
        results = self.cache.load_cache(self.experiment_name, "results", default={})
        
        # Load processed combinations to avoid re-computation
        processed_combinations = self.cache.load_cache(
            self.experiment_name, "processed_combinations", default=set()
        )

        for level in LEVELS:
            descriptions = retrieve_descriptions(self.descr_folder, level)
            for index_level in INDICES:
                # Skip if this combination has already been processed
                key = f"Level {level}, Index {index_level}"
                if key in results and key in processed_combinations:
                    print(f"Skipping already processed: {key}")
                    continue
                    
                if level not in index_level:  # Leave-one-out
                    self.args.storage_dir = f"data/vector_dbs/code_as_text/{index_level}"
                    index = load_index(self.args)

                    result_path = f"{self.results_dir}/index_{index_level}_level_{level}.txt"
                    rankings = self.ranking(index, descriptions, result_path, k=5)

                    # Calculate MRR directly from rankings
                    mrr = self.compute_mrr_from_rankings(rankings)
                    results[key] = mrr
                    
                    # Mark this combination as processed
                    processed_combinations.add(key)
                    
                    # Update cache after each combination
                    self.cache.save_cache(self.experiment_name, "results", results)
                    self.cache.save_cache(self.experiment_name, "processed_combinations", processed_combinations)

        # Save all results
        self.save_results(results, f"{self.results_dir}/exp1.txt")
        return results
