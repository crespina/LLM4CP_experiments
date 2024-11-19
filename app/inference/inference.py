from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from llama_index.postprocessor.cohere_rerank import CohereRerank
from MnZcDescriptor.app.data_processing.data_loaders import load_index
from llama_index.core import PromptTemplate


class Inference:
    def __init__(self, args):
        self.model = Groq(model="llama-3.2-90b-text-preview", api_key=args.groq_api_key,
                          model_kwargs={"seed": 42}, temperature=0.1)

        self.index = load_index(args)
        self.embedding_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
        self.reranker = CohereRerank(api_key=args.cohere_api_key, top_n=5)

        self.prompt = PromptTemplate(
            "You are an expert in high-level constraint modelling and solving discrete optimization problems. \n"
            "The user asks you this question regarding their problem:"
            "--------------\n"
            "{question}"
            "--------------\n"
            "Based upon what you know, provide them with an algorithm that will solve their problem through constraint"
            "programming. If you are not sure about the answer, reply 'I do not know, please restate your problem more"
            "clearly'. Instead of providing an elaborate answer, provide the name of the algorithm and a short "
            "description of it."
        )

    def query_llm(self, question):
        query_engine = self.index.as_query_engine(llm=self.model,
                                                  similarity_top_k=5,
                                                  node_postprocessors=[self.reranker])
        user_query = self.prompt.format(question=question)
        response = query_engine.query(user_query)
        return response
