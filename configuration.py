"""
Augustin CRESPIN
augustin.crespin@student.uclouvain.be | crespin.augustin@gmail.com

Ioannis KOSTIS
ioannis.kostis@uclouvain.be | ioannis.aris.kostis@gmail.com

Config object that handles the various parameters and configurations of the system.
"""

import os

import configargparse
from dotenv import load_dotenv

load_dotenv(dotenv_path='./app/assets/env/.env')


def config_parser():
    parser = configargparse.ArgumentParser(
        description="LLM 4 CP.")
    parser.add_argument('--config', is_config_file=True,
                        help='config file path')

    # I/O params
    parser.add_argument('--mzn_path', type=str,
                        default="./data/input/mzn",
                        help='.mzn directory input path.')
    parser.add_argument('--txt_path', type=str,
                        default="./data/input/txt",
                        help='.txt directory input path.')
    parser.add_argument('--storage_dir', type=str,
                        default='./data/vector_dbs/code_as_text/beginnermedium',
                        help='Vector DB directory path.')

    # API Keys
    parser.add_argument('--groq_api_key', type=str,
                        default=os.environ.get('GROQ_API_KEY'),
                        help='Your Groq API token key gsk_<...>)')

    return parser
