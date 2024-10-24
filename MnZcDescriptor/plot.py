from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from pydantic import BaseModel, Field
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import util


### TODO : get rid of that by modifying the pickle loader
class TextDescription(BaseModel):
    """A description in English of a problem represented by a MiniZinc model"""

    name: str = Field(description="The name of the problem")
    description: str = Field(description="A description of the problem")
    variables: str = Field(
        description="All the decision variables in mathematical notation, followed by an explanation of what they are in English"
    )
    constraints: str = Field(
        description="All the constraints in mathematical notation only"
    )
    objective: str = Field(
        description="The objective of the problem (minimize or maximize what value)"
    )


class Questions(BaseModel):
    """Situations or problems that a user could be facing that would be modelled as the given described model"""

    question1: str = Field(
        description="A question/scenario that is from a user very skilled in modelling and solving constraint problems"
    )
    question2: str = Field(
        description="A question/scenario that is from a user that knows nothing about formal modelling and solving constraint problems"
    )
    question3: str = Field(description="A question/scenario that is from a young user")
    question4: str = Field(description="A question/scenario that is very short")
    question5: str = Field(
        description="A question/scenario that is very long and specific"
    )


def cosine_confusion_matrix(
    save_name, sentences=None, embedding_vectors = {}, model_name="BAAI/bge-small-en-v1.5", labels=None, heatmap = False
):

    # if absent, first compute the embedding vectors of the sentences
    if not embedding_vectors : 
        embed_model = HuggingFaceEmbedding(model_name=model_name)
        for sentence in sentences:
            embedding_vectors[sentence] = embed_model.get_text_embedding(sentence)

    # then compute the cosine similarity
    if labels == None:
        labels = list(embedding_vectors.keys())

    embeddings = np.array(list(embedding_vectors.values()))

    similarity_matrix = cosine_similarity(embeddings)

    # plot
    if (not heatmap) :
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            similarity_matrix,
            annot=True,
            cmap="coolwarm",
            xticklabels=labels,
            yticklabels=labels,
        )
        plt.title("Cosine Similarity Matrix")
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)

        if save_name !=None :
            plt.savefig(
                "MnZcDescriptor\\figures\\" + save_name + ".pdf",
                format="pdf",
                bbox_inches="tight",
            )
        plt.show()

    elif (heatmap):

        plt.figure(figsize=(12, 10))  # Adjust figure size
        sns.heatmap(
            similarity_matrix, cmap="coolwarm", xticklabels=False, yticklabels=False
        )
        if save_name !=None :
            plt.savefig(
                "MnZcDescriptor\\figures\\" + save_name + ".pdf",
                format="pdf",
            )
        plt.show()


def KMeans_clustering_plot(
    save_name, best_n = None, sentences = None, embedding_vectors = {}, model_name="BAAI/bge-small-en-v1.5", labels=None
):

    # if absent, first compute the embedding vectors of the sentences
    if (not embedding_vectors):
        embed_model = HuggingFaceEmbedding(model_name=model_name)
        for sentence in sentences:
            embedding_vectors[sentence] = embed_model.get_text_embedding(sentence)

    embeddings = np.array(list(embedding_vectors.values()))

    # Silhouette analysis to find the best nb of clusters
    max_score = -2

    if (not best_n):
        best_n = -1

        for n_cluster in range (2,50,1):
            kmeans = KMeans(n_clusters=n_cluster, random_state=19851900)
            score = silhouette_score(embeddings, kmeans.fit_predict(embeddings))
            if (score > max_score):
                max_score = score
                best_n = n_cluster

    # K-Means Clustering
    kmeans = KMeans(n_clusters=best_n, random_state=19851900)
    cluster_labels = kmeans.fit_predict(embeddings)

    # Reduce the dimensionality (for visualization)
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings)

    # Plot
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x=reduced_embeddings[:, 0],
        y=reduced_embeddings[:, 1],
        hue=cluster_labels,
        palette="viridis",
        s=100,
    )

    if labels == None:
        labels = sentences

    for i, label in enumerate(labels):
        plt.text(
            reduced_embeddings[i, 0],
            reduced_embeddings[i, 1],
            label,
            fontsize=9,
            ha="right",
        )

    plt.title(f"K-Means Clustering with {best_n} Clusters")

    if save_name != None:
        plt.savefig(
            "MnZcDescriptor\\figures\\" + save_name + ".pdf",
            format="pdf",
            bbox_inches="tight",
        )

    plt.show()

def hierarchical_clustering_plot(
    save_name, sentences = None, embedding_vectors={}, model_name="BAAI/bge-small-en-v1.5", labels=None
):

    # if absent, first compute the embedding vectors of the sentences
    if (not embedding_vectors):
        embed_model = HuggingFaceEmbedding(model_name=model_name)
        for sentence in sentences:
            embedding_vectors[sentence] = embed_model.get_text_embedding(sentence)

    embeddings = np.array(list(embedding_vectors.values()))

    # Step 2: Apply Hierarchical Clustering (using Ward's method)
    Z = linkage(embeddings, method="ward")

    # Step 3: Plot the Dendrogram
    if labels == None:
        labels = sentences
    plt.figure(figsize=(10, 8))
    dendrogram(Z, labels=labels, leaf_rotation=90, leaf_font_size=10)
    plt.title("Hierarchical Clustering Dendrogram")
    plt.xlabel("Sentences")
    plt.ylabel("Distance")

    if save_name != None:
        plt.savefig(
            "MnZcDescriptor\\figures\\" + save_name + ".pdf",
            format="pdf",
            bbox_inches="tight",
        )

    plt.show()

instances = util.load_instances("qwen_text_32_90b_quest")
labels = []
embedding_vectors = {}
for key, value in instances.items():
    labels.append(key)
    embedding_vectors[key] = value.metadata["embedding_vector"]


#cosine_confusion_matrix(save_name="qwen_text_32_90b_quest_cosine_sim",labels=labels,embedding_vectors=embedding_vectors)

KMeans_clustering_plot("qwen_text_32_90b_quest_kmeans" , embedding_vectors=embedding_vectors, labels=labels)

#hierarchical_clustering_plot("qwen_text_32_90b_quest_hierarchical",embedding_vectors=embedding_vectors,labels=labels)
