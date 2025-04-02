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

    agent = Inference(args=args)
    query = get_input_safely("Question: ")
    while True:
        response, user_query = agent.query_llm(question=query)
        pprint_ranking(response=response, question=user_query, query=query)
        query = get_input_safely("\nQuestion: ")
