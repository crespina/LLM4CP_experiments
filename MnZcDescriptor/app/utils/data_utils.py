import os

from langchain.output_parsers import ResponseSchema
from tqdm import tqdm


def convert_mzn_to_txt(mzn_path, txt_path):
    # Walk through every directory and subdirectory in the input directory
    for dir_path, _, filenames in os.walk(mzn_path):
        for filename in tqdm(filenames, desc="Converting .mzn to .txt"):
            # Process only .mzn files
            if filename.endswith(".mzn"):
                mzn_file_path = os.path.join(dir_path, filename)
                with open(mzn_file_path, 'r') as mzn_file:
                    content = mzn_file.read()

                relative_path = os.path.relpath(dir_path, mzn_path)
                output_dir_path = os.path.join(txt_path, relative_path)

                os.makedirs(output_dir_path, exist_ok=True)

                txt_file_path = os.path.join(output_dir_path, filename.replace('.mzn', '.txt'))

                with open(txt_file_path, 'w') as txt_file:
                    txt_file.write(content)


def get_response_schema(query_config_dict):
    response_schema = []
    for field, description in query_config_dict.items():
        response_schema.append(ResponseSchema(name=field.strip(), description=description.strip()))
    return response_schema
