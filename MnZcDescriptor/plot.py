from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def cosine_confusion_matrix(
    sentences, model_name="BAAI/bge-small-en-v1.5", labels=None, save_name = None
):
    
    # first compute the embedding vectors of the sentences
    embedding_vectors = {}
    embed_model = HuggingFaceEmbedding(model_name=model_name)
    for sentence in sentences:
        embedding_vectors[sentence] = embed_model.get_text_embedding(sentence)

    # then compute the cosine similarity
    if labels == None:
        labels = list(embedding_vectors.keys())

    embeddings = np.array(list(embedding_vectors.values()))

    similarity_matrix = cosine_similarity(embeddings)

    # plot

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
    plt.show()
    if save_name !=None :
        plt.savefig(
            "MnZcDescriptor\\figures\\" + save_name + ".pdf",
            format="pdf",
            bbox_inches="tight",
        )


def KMeans_clustering_plot(
    sentences, n_clusters=3, model_name="BAAI/bge-small-en-v1.5", labels=None, save_name = None
):

    # first compute the embedding vectors of the sentences
    embedding_vectors = {}
    embed_model = HuggingFaceEmbedding(model_name=model_name)
    for sentence in sentences:
        embedding_vectors[sentence] = embed_model.get_text_embedding(sentence)

    embeddings = np.array(list(embedding_vectors.values()))

    # K-Means Clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=19851900)
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

    plt.title(f"K-Means Clustering with {n_clusters} Clusters")
    plt.show()

    if save_name != None:
        plt.savefig(
            "MnZcDescriptor\\figures\\" + save_name + ".pdf",
            format="pdf",
            bbox_inches="tight",
        )


def hierarchical_clustering_plot(
    sentences, model_name="BAAI/bge-small-en-v1.5", labels=None, save_name = None
):

    # first compute the embedding vectors of the sentences
    embedding_vectors = {}
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
    plt.show()

    if save_name != None:
        plt.savefig(
            "MnZcDescriptor\\figures\\" + save_name + ".pdf",
            format="pdf",
            bbox_inches="tight",
        )


"""
def confusion_matrix(index, instances):
    # also questions x code, questions x description, question x code+string (join string)
    vectors = index._vector_store._data.embedding_dict
    num_categories = 5
    texts_per_category = 5

    # Step 1: Extract the embeddings and group them by category
    embeddings = np.array(list(vectors.values()))

    # Step 2: Average embeddings by category
    category_embeddings = []
    category_labels = []
    for i in range(num_categories):
        # Extract embeddings for this category
        category_embs = embeddings[
            i * texts_per_category : (i + 1) * texts_per_category
        ]

        # Average the embeddings for the current category
        avg_embedding = np.mean(category_embs, axis=0)

        # Store the average embedding and the category label
        category_embeddings.append(avg_embedding)

    # Convert to NumPy array for cosine similarity
    for key, value in instances.items():
        category_labels.append(key)
    category_embeddings = np.array(category_embeddings)

    # Step 3: Compute the cosine similarity matrix between categories
    similarity_matrix = cosine_similarity(category_embeddings)

    # Step 4: Plot the confusion matrix using a heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        similarity_matrix,
        annot=True,
        cmap="coolwarm",
        xticklabels=category_labels,
        yticklabels=category_labels,
    )
    plt.title("Cosine Similarity between Categories")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.show()
"""
