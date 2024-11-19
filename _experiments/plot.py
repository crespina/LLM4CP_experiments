from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from pydantic import BaseModel, Field
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import SpectralClustering
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import util
from sklearn.feature_extraction.text import TfidfVectorizer


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
    save_name, sentences=None, embedding_vectors = {}, model_name="BAAI/bge-base-en-v1.5", labels=None, heatmap = False
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
            cmap="viridis",
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
            similarity_matrix,
            cmap="coolwarm",
            xticklabels=labels,
            yticklabels=labels,
        )
        plt.title("Cosine Similarity Matrix")
        if save_name !=None :
            plt.savefig(
                "MnZcDescriptor\\figures\\" + save_name + ".pdf",
                format="pdf",
            )
        plt.show()


def KMeans_clustering_plot(
    save_name, best_n = None, sentences = None, embedding_vectors = {}, model_name="BAAI/bge-base-en-v1.5", labels=None
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

        for n_cluster in range (2,24,1):
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


def SpectralClustering_plot(
    save_name,
    best_n=None,
    sentences=None,
    embedding_vectors={},
    model_name="BAAI/bge-base-en-v1.5",
    labels=None,
):

    # if absent, first compute the embedding vectors of the sentences
    if not embedding_vectors:
        embed_model = HuggingFaceEmbedding(model_name=model_name)
        for sentence in sentences:
            embedding_vectors[sentence] = embed_model.get_text_embedding(sentence)

    embeddings = np.array(list(embedding_vectors.values()))

    # Silhouette analysis to find the best number of clusters
    max_score = -1

    if not best_n:
        best_n = -1
        for n_cluster in range(2, 25):
            spectral = SpectralClustering(
                n_clusters=n_cluster,
                affinity="nearest_neighbors",
                random_state=19851900,
            )
            labels = spectral.fit_predict(embeddings)
            score = silhouette_score(embeddings, labels)
            if score > max_score:
                max_score = score
                best_n = n_cluster

    # Spectral Clustering
    spectral = SpectralClustering(
        n_clusters=best_n, affinity="nearest_neighbors", random_state=19851900
    )
    cluster_labels = spectral.fit_predict(embeddings)

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

    # Use labels if provided; otherwise, use sentences as labels
    if labels is None:
        labels = sentences

    for i, label in enumerate(labels):
        plt.text(
            reduced_embeddings[i, 0],
            reduced_embeddings[i, 1],
            label,
            fontsize=9,
            ha="right",
        )

    plt.title(f"Spectral Clustering with {best_n} Clusters")

    # Save the figure if a filename is provided
    if save_name is not None:
        plt.savefig(
            'MnZcDescriptor\\_results\\figures\\llama32_90b_base_test\\whole_doc' + save_name + ".pdf",
            format="pdf",
            bbox_inches="tight",
        )

    plt.show()


def hierarchical_clustering_plot(
    save_name, sentences = None, embedding_vectors={}, model_name="BAAI/bge-base-en-v1.5", labels=None
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
            "_results\\figures\\" + save_name + ".pdf",
            format="pdf",
            bbox_inches="tight",
        )

    plt.show()


instances = util.load_instances(
    "MnZcDescriptor\data\model_checkpoints\llama32_90b_both_base_embedding.pkl"
)
labels = []
sentences = []
embedding_dict = {}
for key, value in instances.items():
    labels.append(key)
    embedding_dict[key] = value.metadata["embedding_vector"]


def barplot_families():

    """
    1 : total = 95 correct1 = 45 correct2 = 15 correct3 = 4 correct4 = 8 correct5 = 2 incorrect 21
    2 : total = 95 correct1 = 38 correct2 = 16 correct3 = 9 correct4 = 6 correct5 = 2 incorrect 24
    3 : total = 95 correct1 = 40 correct2 = 15 correct3 = 8 correct4 = 3 correct5 = 5 incorrect 24
    4 : total = 95 correct1 = 73 correct2 = 7 correct3 = 3 correct4 = 1 correct5 = 0 incorrect 11
    5 : total = 95 correct1 = 29 correct2 = 12 correct3 = 4 correct4 = 7 correct5 = 2 incorrect 41
    """
    # Data
    labels = ["First", "Second", "Third", "Fourth", "Fifth", "Incorrect"]
    values = [29, 12, 4, 7, 2, 41]

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.bar(labels, values, color=["green", "green", "green", "green", "green", "red"])
    plt.xlabel("Propositions")
    plt.ylabel("Counts")
    plt.title("Identification of the problem : leave question 5")
    plt.savefig(
        "_results\\figures\leave_one_out_5\\fifth.pdf",
        format="pdf",
        bbox_inches="tight",
    )
    plt.show()

# barplot_families()
# cosine_confusion_matrix(save_name="llama32_90b_base_test_cosine_sim",labels=labels,sentences=sentences, heatmap=True)

# KMeans_clustering_plot("llama32_90b_base_test_kmeans", best_n=5 ,sentences=sentences, labels=labels)

# SpectralClustering_plot("llama32_90b_base_test_spectral", embedding_vectors=embedding_dict, labels=labels, best_n=5)

# hierarchical_clustering_plot("llama32_90b_both_base\llama32_90b_both_base_embedding_hierarchical_larger.pdf",embedding_vectors=embedding_dict,labels=labels)


barplot_families()
