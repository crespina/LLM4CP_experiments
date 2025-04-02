import os
import warnings

from app.inference.inference import Inference
from app.utils.app_utils import pprint_ranking, get_input_safely
from configuration import config_parser

if __name__ == "__main__":
    warnings.simplefilter(action='ignore', category=FutureWarning)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    parser = config_parser()
    args = parser.parse_args()

    # args.storage_dir = "./data/vector_dbs/code_as_text/medium"

    agent = Inference(args=args)
    print("Constraint Programming Question Answering System")
    print("Ask questions about constraint programming algorithms and problems.")
    print("You can enter multiline questions.\nPress Enter twice to submit your question.")
    print("Type 'quit' to exit the program.")
    
    while True:
        query = get_input_safely("Question: ")
        if query.strip().lower() == "quit":
            print("Exiting program.")
            break
            
        response, user_query = agent.query_llm(question=query)
        pprint_ranking(response=response)
        print("\n")
