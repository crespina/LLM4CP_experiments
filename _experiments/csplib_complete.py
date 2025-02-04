import os

from llama_index.core import Settings, StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from llama_index.postprocessor.cohere_rerank import CohereRerank

from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

os.environ["GROQ_API_KEY"] = "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"

def load_index():
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

    Settings.show_progress = True

    if os.path.exists("data/vector_dbs/csplib_inference"):
        storage_context = StorageContext.from_defaults(persist_dir="data/vector_dbs/csplib_inference")
        index = load_index_from_storage(storage_context, show_progress=True)
        print("Loaded index from storage.")
        return index
    else:
        print("Index storage directory not found. Parse and store the index first.")
        exit()


def replace(index_name, problem_data):
    for family_name, names in problem_data.items():
        if index_name in names:
            return family_name, len(names)
    return None


def similary_description(index, spec, problem_data, avg=True):

    """
    index_docs = index.docstore.docs
    for index_name, index_document in index_docs.items() :
        for probl_family, probl_names in problem_data.items():
            for probl_name in probl_names :

                if probl_name == index_name : 
                    #corresponding
                    csplib_description = spec[probl_family]
                    generated_description = index_document.text
    """

    generated_descriptions_embeds = {}
    csplib_descriptions_embeds = {}
    max_sim = {}
    embedding_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

    for name, doc in spec.items():
        csplib_descriptions_embeds[name] = embedding_model.get_text_embedding(doc)

    for index_name, index_document in index.docstore.docs.items() :
        family_name, nb_files = replace(index_name, problem_data)

        if avg : 

            if not (family_name in generated_descriptions_embeds.keys()):
                embed = embedding_model.get_text_embedding(index_document.text)
                div_embed = [x / nb_files for x in embed]
                generated_descriptions_embeds[family_name] = div_embed

            else :
                embed = embedding_model.get_text_embedding(index_document.text)
                div_embed = [x / nb_files for x in embed]
                summed_embed = np.add(div_embed, generated_descriptions_embeds[family_name])
                generated_descriptions_embeds[family_name] = summed_embed

        else : #max

            embed = embedding_model.get_text_embedding(index_document.text)
            sim = cosine_similarity(np.array(embed).reshape(1, -1), np.array(csplib_descriptions_embeds[family_name]).reshape(1, -1))[0, 0] 

            if not (family_name in generated_descriptions_embeds.keys()):
                generated_descriptions_embeds[family_name] = embed
                max_sim[family_name] = sim

            else : 
                if sim > max_sim[family_name] : 
                    generated_descriptions_embeds[family_name] = embed

    similarity_matrix = cosine_similarity(np.array(list(generated_descriptions_embeds.values())), np.array(list(csplib_descriptions_embeds.values())))

    xlabels = list(csplib_descriptions_embeds.keys())
    ylabels = list(generated_descriptions_embeds.keys())

    return similarity_matrix, xlabels, ylabels

def model_output(index, model, reranker, problem_data, spec):

    query_engine = index.as_query_engine(llm= model,
                                                similarity_top_k=10,
                                                node_postprocessors=[reranker])

    go = False

    with open("_results/figures/csplib_all/classification/results.txt", "a") as f:  # Open file in append mode

        for problem_name, problem_descr in tqdm(spec.items(), desc= "Generating Answers"):
            if problem_name == "The_Rehearsal_Problem":
                go = True

            if go : 
                response = query_engine.query(problem_descr)
                family_name_1, _ = replace(response.source_nodes[0].metadata["model_name"], problem_data)
                family_name_2, _ = replace(response.source_nodes[1].metadata["model_name"], problem_data)
                family_name_3, _ = replace(response.source_nodes[2].metadata["model_name"], problem_data)
                family_name_4, _ = replace(response.source_nodes[3].metadata["model_name"], problem_data)
                family_name_5, _ = replace(response.source_nodes[4].metadata["model_name"], problem_data)

                total = 0
                correct1 = 0
                correct2 = 0
                correct3 = 0
                correct4 = 0
                correct5 = 0
                incorrect = 0

                if problem_name != family_name_1:
                    if problem_name != family_name_2:
                        if problem_name != family_name_3:
                            if problem_name != family_name_4:
                                if problem_name != family_name_5:
                                    incorrect += 1
                                else:
                                    correct5 += 1
                            else:
                                correct4 += 1
                        else:
                            correct3 += 1
                    else:
                        correct2 += 1
                else:
                    correct1 += 1

                total += 1

                print(
                    problem_name,
                    family_name_1,
                    family_name_2,
                    family_name_3,
                    family_name_4,
                    family_name_5,
                )
                f.write(
                    problem_name + " " + family_name_1 + " " + family_name_2 + " " + family_name_3 + " " + family_name_4 + " " + family_name_5 + "\n"
                )

        f.write(
                "total = " + str(total) +
                " correct1 = " + str(correct1) +
                " correct2 = " + str(correct2) +
                " correct3 = " + str(correct3) +
                " correct4 = " + str(correct4) +
                " correct5 = " + str(correct5) +
                " incorrect " + str(incorrect) +
                "\n",
            )

    return


def plot_sim_matrix(similarity_matrix, heatmap, x_labels, y_labels, title, save_name=None):

    if not heatmap:
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            similarity_matrix,
            annot=True,
            cmap="viridis",
            xticklabels=x_labels,
            yticklabels=y_labels,
        )
        plt.title(title)
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)

        if save_name != None:
            plt.savefig(
                "MnZcDescriptor\\figures\\" + save_name + ".pdf",
                format="pdf",
                bbox_inches="tight",
            )
        plt.show()

    elif heatmap:

        plt.figure(figsize=(12, 10))  # Adjust figure size
        sns.heatmap(
            similarity_matrix,
            cmap="coolwarm",
            xticklabels=x_labels,
            yticklabels=y_labels,
        )
        plt.title(title)
        if save_name != None:
            plt.savefig(
                save_name,
                format="pdf",
            )
        plt.show()

def plot_model_output():
    # Data
    labels = ["First", "Second", "Third", "Fourth", "Fifth", "Incorrect"]
    values = [30, 3, 2, 0, 0, 1]

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.bar(labels, values, color=["green", "green", "green", "green", "green", "red"])
    plt.xlabel("Propositions")
    plt.ylabel("Counts")
    plt.title("Identification of the problem : CSPLib problem descriptions")
    plt.savefig(
        "_results/figures/csplib_all/classification/csplib_model_output.pdf",
        format="pdf",
        bbox_inches="tight",
    )
    plt.show()


def experiment():

    problem_data = {}
    specifications = {}

    for problem in os.listdir("data/csplib"):
        problem_path = os.path.join("data/csplib", problem)
        if os.path.isdir(problem_path):
            mzn_files = [f[:-4] for f in os.listdir(problem_path) if f.endswith(".mzn")]
            spec_file = os.path.join(problem_path, "specification.md")

            if os.path.exists(spec_file):
                with open(spec_file, "r", encoding="utf-8") as f:
                    specifications[problem] = f.read()

            problem_data[problem] = mzn_files

    model = Groq(
        model="llama3-70b-8192",
        model_kwargs={"seed": 19851900},
        temperature=0.1,
    )

    index = load_index()
    # similarity_matrix, x_labels, y_labels = similary_description(index=index, spec=specifications, problem_data=problem_data, avg=False)
    # plot_sim_matrix(similarity_matrix, True, x_labels, y_labels, title= "Cosine Similarity Matrix : Maximum", save_name="_results/figures/csplib_all/descriptions_comparison/csm_max.pdf")

    reranker = CohereRerank(api_key="STPahNFoWeYX4FSAoMx7NzHNgH2ejINXLDKIYOr4", top_n=5)
    # model_output(index, model, reranker, problem_data, specifications)
    # plot_model_output()

experiment()
plot_model_output()
