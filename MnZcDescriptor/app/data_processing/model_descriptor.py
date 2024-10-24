import os
from time import sleep

from langchain.output_parsers import StructuredOutputParser
from llama_index.core import Document
from llama_index.core import PromptTemplate
from llama_index.core.output_parsers import LangchainOutputParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from tqdm import tqdm

from MnZcDescriptor.app.utils.data_utils import get_response_schema


class Storage:
    def __init__(self, args):
        self.args = args
        self.output_parser = LangchainOutputParser(
            StructuredOutputParser.from_response_schemas(get_response_schema(query_config_dict={
                "name": "The name of the problem",
                "description": "A description of the problem",
                "variables": "All the decision variables in mathematical notation, followed by an explanation of what they are in English",
                "constraints": "All the constraints in mathematical notation only",
                "objective": "The objective of the problem (minimize or maximize what value)"
            })))

        self.description_template = PromptTemplate(
            " You are an expert in high-level constraint modelling and solving discrete optimization problems. \n"
            "In particular, you know Minizinc. You are provided with a Minizinc model that represents a classical "
            "problem in constraint programming. Your task is to identify what is the problem modelled and give a "
            "complete description of the problem to the user. \n"
            "This is the source code of the model:\n"
            "--------------\n"
            "{source_code}"
            "--------------\n"
            "The format of the answer should be without any variation a JSON-like format with the following keys and explanation of what the corresponding values should be:\n"
            "name: The name of the problem\n"
            "description: A description of the problem in English\n"
            "variables: A string containing the list of all the decision variables in mathematical notation, followed by an explanation of what they are in English\n"
            "constraints: A string containing the list of all the constraints in mathematical notation, followed by an explanation of what they are in English\n"
            "objective: The objective of the problem (minimize or maximize what value)"
        )

        self.descriptor_model = Groq(model="llama-3.2-90b-text-preview", api_key=args.groq_api_key,
                                     model_kwargs={"seed": 42}, temperature=0.1, output_parser=self.output_parser)

        self.embeddings_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

        self.documents = []

    def generate_descriptions(self):
        for filename in tqdm(os.listdir(self.args.txt_path), desc="Generating descriptions"):
            file_path = os.path.join(self.args.txt_path, filename)
            if os.path.isfile(file_path):
                with open(file_path, "r") as file:
                    file_content = file.read()
                    prompt = self.description_template.format(source_code=file_content)
                    text_description = self.descriptor_model.complete(prompt=prompt, formatted=True)
                    print(text_description)
                    cp_model = Document(
                        text=file_content,
                        metadata={
                            "model_name": os.path.splitext(filename)[0],
                            "source_code": file_content
                        },
                    )
                    self.documents.append(cp_model)
                    sleep(5)

    def run(self):
        self.generate_descriptions()
        print("checkpoint")
