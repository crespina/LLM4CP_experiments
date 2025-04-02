from app.data_processing.data_loaders import load_index
from app.experiments.Experiment import Experiment
from app.utils.CONSTANTS import INDICES_EXP
from app.utils.exp_utils import retrieve_descriptions_csplib


class Experiment2(Experiment):
    def __init__(self, args):
        super().__init__(args)
        self.csplib_desc_dir_path = "./data/input/csplib_descriptions_obfuscated"
        self.results_dir = f"{args.results_dir}/exp2"

        self.args = args

    def run(self):
        """Run Experiment 2 with caching support"""
        # Load cached results if available
        results = self.cache.load_cache(self.experiment_name, "results", default={})

        # Load processed indices to avoid recomputation
        processed_indices = self.cache.load_cache(
            self.experiment_name, "processed_indices", default=set()
        )

        descriptions = retrieve_descriptions_csplib(self.csplib_desc_dir_path)

        for index_level in INDICES_EXP:
            # Skip if this index has already been processed
            key = f"Index {index_level}"
            if key in results and index_level in processed_indices:
                print(f"Skipping already processed: {key}")
                continue

            self.args.storage_dir = f"./data/vector_dbs/code_as_text/{index_level}"
            index = load_index(self.args)

            result_path = f"{self.results_dir}/index_{index_level}.txt"
            rankings = self.ranking(index, descriptions, result_path, k=5)

            # Calculate MRR directly from rankings
            mrr = self.compute_mrr_from_rankings(rankings)
            results[key] = mrr

            # Mark this index as processed
            processed_indices.add(index_level)

            # Update cache after each index
            self.cache.save_cache(self.experiment_name, "results", results)
            self.cache.save_cache(self.experiment_name, "processed_indices", processed_indices)

        # Save all results
        self.save_results(results, f"{self.results_dir}/exp2.txt")
        return results
