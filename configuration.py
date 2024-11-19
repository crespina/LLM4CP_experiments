"""
Augustin CRESPIN
augustin.crespin@student.uclouvain.be | crespin.augustin@gmail.com

Ioannis KOSTIS
ioannis.kostis@uclouvain.be | ioannis.aris.kostis@gmail.com

Config object that handles the various parameters and configurations of the system.
"""

import configargparse


def config_parser():
    parser = configargparse.ArgumentParser(
        description="LLM 4 CP.")
    parser.add_argument('--config', is_config_file=True,
                        help='config file path')
    parser.add_argument("--keys", is_config_file=True, required=False,
                        help='Path to the API keys file.',
                        default='./app/assets/.api_keys/keys.txt')

    # I/O params
    parser.add_argument('--mzn_path', type=str,
                        default="./data/input/mzn",
                        help='.mzn directory input path.')
    parser.add_argument('--txt_path', type=str,
                        default="./data/input/txt",
                        help='.txt directory input path.')
    parser.add_argument('--storage_dir', type=str,
                        default='./data/vector_dbs/test_db',
                        help='Vector DB directory path.')

    # API Keys
    parser.add_argument('--llama_parse_key', type=str, help='Your LlamaParse token key (llx-<...>)')
    parser.add_argument('--openai_api_key', type=str, help='Your OPENAI API token key (sk-<...>)')
    parser.add_argument('--groq_api_key', type=str, help='Your Groq API token key gsk_<...>)')
    parser.add_argument('--cohere_api_key', type=str, help='Your Cohere API token key <...>)')
    
    return parser
