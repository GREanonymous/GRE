import matplotlib.pyplot as plt
import tiktoken
from bs4 import BeautifulSoup as Soup
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import set_start_method
from openai import OpenAI
import umap.umap_ as umap
from scipy import spatial
import json
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from FlagEmbedding import FlagModel
from langchain_anthropic import ChatAnthropic
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from sklearn.mixture import GaussianMixture
from multiprocessing import Value, Manager

# Set a fixed random seed for reproducibility
RANDOM_SEED = 224

# Initialize the OpenAI client with a specific API key and endpoint
client = OpenAI(
    api_key="your-key-here",
    base_url="https://api.deepseek.com",
)

# Chat function to interact with the OpenAI API
def chat(query: str) -> str:
    """
    Send a chat query to the OpenAI API and return the response.

    Parameters:
    - query: The input query string.

    Returns:
    - A string containing the response from the API.
    """
    setting = [{"role": "user", "content": query}]
    while True:
        try:
            completion = client.chat.completions.create(
                model="deepseek-chat",
                messages=setting,
                temperature=0.0,
                max_tokens=200
            )
            result = completion.choices[0].message.content
            return result
        except Exception as e:
            error_message = str(e)
            if "400" in error_message and "high risk" in error_message:
                print(f"Bad request error: {error_message}. Skipping this query.")
                return None
            print(f"Request failed: {error_message}. Retrying in 10 seconds...")
            time.sleep(1)

# Function to calculate the number of tokens in a string
def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """
    Calculate the number of tokens in a text string using a specific encoding.

    Parameters:
    - string: The text string to tokenize.
    - encoding_name: The name of the encoding used for tokenization.

    Returns:
    - The number of tokens in the text string.
    """
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

# Function for global dimensionality reduction using UMAP
def global_cluster_embeddings(
        embeddings: np.ndarray,
        dim: int,
        n_neighbors: Optional[int] = None,
        metric: str = "cosine",
) -> np.ndarray:
    """
    Perform global dimensionality reduction on embeddings using UMAP.

    Parameters:
    - embeddings: A numpy array of input embeddings.
    - dim: Target dimensionality for reduction.
    - n_neighbors: Number of neighbors for UMAP; defaults to sqrt(number of embeddings).
    - metric: Distance metric to use in UMAP.

    Returns:
    - A numpy array with reduced embeddings.
    """
    if n_neighbors is None:
        n_neighbors = int((len(embeddings) - 1) ** 0.5)
    return umap.UMAP(
        n_neighbors=n_neighbors, n_components=dim, metric=metric
    ).fit_transform(embeddings)

# Function for local dimensionality reduction using UMAP
def local_cluster_embeddings(
        embeddings: np.ndarray, dim: int, num_neighbors: int = 10, metric: str = "cosine"
) -> np.ndarray:
    """
    Perform local dimensionality reduction on embeddings using UMAP.

    Parameters:
    - embeddings: A numpy array of input embeddings.
    - dim: Target dimensionality for reduction.
    - num_neighbors: Number of neighbors for UMAP.
    - metric: Distance metric to use in UMAP.

    Returns:
    - A numpy array with reduced embeddings.
    """
    return umap.UMAP(
        n_neighbors=num_neighbors, n_components=dim, metric=metric
    ).fit_transform(embeddings)

# Function to determine optimal number of clusters using Gaussian Mixture Model (GMM)
def get_optimal_clusters(
        embeddings: np.ndarray, max_clusters: int = 50, random_state: int = RANDOM_SEED
) -> int:
    """
    Use Bayesian Information Criterion (BIC) to determine optimal clusters with GMM.

    Parameters:
    - embeddings: A numpy array of embeddings.
    - max_clusters: Maximum clusters to consider.
    - random_state: Random seed for reproducibility.

    Returns:
    - Optimal number of clusters.
    """
    max_clusters = min(max_clusters, len(embeddings))
    n_clusters = np.arange(1, max_clusters)
    bics = []
    for n in n_clusters:
        gm = GaussianMixture(n_components=n, random_state=random_state)
        gm.fit(embeddings)
        bics.append(gm.bic(embeddings))
    return n_clusters[np.argmin(bics)]

# Function to cluster embeddings with GMM based on a probability threshold
def GMM_cluster(embeddings: np.ndarray, threshold: float, random_state: int = 0):
    """
    Cluster embeddings using GMM with a probability threshold.

    Parameters:
    - embeddings: A numpy array of embeddings.
    - threshold: Probability threshold for clustering.
    - random_state: Random seed.

    Returns:
    - Tuple of cluster labels and number of clusters.
    """
    n_clusters = get_optimal_clusters(embeddings)
    gm = GaussianMixture(n_components=n_clusters, random_state=random_state)
    gm.fit(embeddings)
    probs = gm.predict_proba(embeddings)
    labels = [np.where(prob > threshold)[0] for prob in probs]
    return labels, n_clusters

# Main function to perform clustering using UMAP and GMM
def perform_clustering(
        embeddings: np.ndarray,
        dim: int,
        threshold: float,
) -> List[np.ndarray]:
    """
    Perform clustering by reducing dimensionality, applying GMM, and clustering locally.

    Parameters:
    - embeddings: A numpy array of input embeddings.
    - dim: Dimensionality for UMAP.
    - threshold: Probability threshold for clustering.

    Returns:
    - List of numpy arrays with cluster labels for each embedding.
    """
    if len(embeddings) <= dim + 1:
        return [np.array([0]) for _ in range(len(embeddings))]

    # Global dimensionality reduction
    reduced_embeddings_global = global_cluster_embeddings(embeddings, dim)
    # Global clustering
    global_clusters, n_global_clusters = GMM_cluster(
        reduced_embeddings_global, threshold
    )

    all_local_clusters = [np.array([]) for _ in range(len(embeddings))]
    total_clusters = 0

    # Iterate over each global cluster for local clustering
    for i in range(n_global_clusters):
        global_cluster_embeddings_ = embeddings[
            np.array([i in gc for gc in global_clusters])
        ]

        if len(global_cluster_embeddings_) == 0:
            continue
        if len(global_cluster_embeddings_) <= dim + 1:
            local_clusters = [np.array([0]) for _ in global_cluster_embeddings_]
            n_local_clusters = 1
        else:
            reduced_embeddings_local = local_cluster_embeddings(
                global_cluster_embeddings_, dim
            )
            local_clusters, n_local_clusters = GMM_cluster(
                reduced_embeddings_local, threshold
            )

        # Assign local cluster IDs
        for j in range(n_local_clusters):
            local_cluster_embeddings_ = global_cluster_embeddings_[
                np.array([j in lc for lc in local_clusters])
            ]
            indices = np.where(
                (embeddings == local_cluster_embeddings_[:, None]).all(-1)
            )[1]
            for idx in indices:
                all_local_clusters[idx] = np.append(
                    all_local_clusters[idx], j + total_clusters
                )

        total_clusters += n_local_clusters

    return all_local_clusters

# Function to generate embeddings for a list of text documents
def embed(texts):
    """
    Generate embeddings for a list of text documents.

    Parameters:
    - texts: List of text documents.

    Returns:
    - A numpy array of embeddings for the documents.
    """
    cleaned_texts = [text.replace("\n", " ").strip() for text in texts]
    text_embeddings = embd.encode(cleaned_texts)
    text_embeddings_np = np.array(text_embeddings)
    return text_embeddings_np

# Function to embed and cluster texts, returning results in a DataFrame
def embed_cluster_texts(texts):
    """
    Embeds and clusters texts, returning a DataFrame with texts, embeddings, and cluster labels.

    Parameters:
    - texts: List of text documents to process.

    Returns:
    - DataFrame with original texts, embeddings, and cluster labels.
    """
    text_embeddings_np = embed(texts)
    cluster_labels = perform_clustering(
        text_embeddings_np, 10, 0.5
    )
    df = pd.DataFrame()
    df["text"] = texts
    df["embd"] = list(text_embeddings_np)
    df["cluster"] = cluster_labels
    return df

# Function to format text from DataFrame into a single string
def fmt_txt(df: pd.DataFrame) -> str:
    """
    Format text documents in a DataFrame into a single string.

    Parameters:
    - df: DataFrame with 'text' column.

    Returns:
    - Formatted string with all texts joined.
    """
    unique_txt = df["text"].tolist()
    return "--- --- \n --- --- ".join(unique_txt)

# Function to embed, cluster, and summarize texts recursively up to a specified level
def recursive_embed_cluster_summarize(
        texts: List[str], level: int = 1, n_levels: int = 3
) -> Dict[int, Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Recursively embed, cluster, and summarize texts up to a certain level.

    Parameters:
    - texts: List of text documents.
    - level: Current recursion level.
    - n_levels: Maximum recursion levels.

    Returns:
    - Dictionary with recursion level as key and cluster/summarization DataFrames as values.
    """
    results = {}
    df_clusters, df_summary= embed_cluster_summarize_texts_by_deepseek(texts, level)
    results[level] = (df_clusters, df_summary)

    unique_clusters = df_summary["cluster"].nunique()
    if level < n_levels and unique_clusters > 1:
        new_texts = df_summary["summaries"].tolist()
        next_level_results = recursive_embed_cluster_summarize(
            new_texts, level + 1, n_levels
        )
        results.update(next_level_results)

    return results

# Calculate distances between embeddings based on specified metric
def distances_from_embeddings(
    query_embedding: List[float],
    embeddings: List[List[float]],
    distance_metric: str = "cosine"
) -> List[float]:
    """
    Calculate distances between a query embedding and a list of embeddings.

    Parameters:
    - query_embedding: Embedding of the query.
    - embeddings: List of embeddings to compare with the query.
    - distance_metric: Distance metric (cosine, L1, L2, or Linf).

    Returns:
    - List of distances for each embedding.
    """
    distance_metrics = {
        "cosine": spatial.distance.cosine,
        "L1": spatial.distance.cityblock,
        "L2": spatial.distance.euclidean,
        "Linf": spatial.distance.chebyshev,
    }

    if distance_metric not in distance_metrics:
        raise ValueError(
            f"Unsupported distance metric '{distance_metric}'. Supported metrics are: {list(distance_metrics.keys())}"
        )

    distances = [
        distance_metrics[distance_metric](query_embedding, embedding)
        for embedding in embeddings
    ]
    return distances

# Function to get indices of nearest neighbors
def indices_of_nearest_neighbors_from_distances(distances: List[float]) -> np.ndarray:
    """
    Return indices of embeddings sorted by proximity to a query based on distances.

    Parameters:
    - distances: List of calculated distances.

    Returns:
    - Indices sorted by distance.
    """
    return np.argsort(distances)

# Initialize the embedding model
embd = FlagModel("../bge-base-en-v1.5", use_fp16=True)

# Load JSON data for processing
file_path = '../../Eval_Data_Preparation/Hotpot/Hotpot_eval_compressed_chunks.json'

with open(file_path, 'r', encoding='utf-8') as file:
    json_data = json.load(file)

# Process each data item and build trees of clustered text summaries
def process_data(data):
    """
    Process a single data item to generate text clusters and summaries.

    Parameters:
    - data: A dictionary containing question, context, and other relevant information.

    Returns:
    - A dictionary representing a tree structure of clustered text summaries.
    """
    docs_texts = []
    passages = data['golden_context'] + data['noisy_context'] + data['distracting_context']
    for passage in passages:
        for chunk in passage[1:]:
            docs_texts.append(chunk)

    results = recursive_embed_cluster_summarize(docs_texts, level=1, n_levels=3)

    all_texts = docs_texts.copy()
    for level in sorted(results.keys()):
        summaries = results[level][1]["summaries"].tolist()
        all_texts.extend(summaries)

    tree_json = {
        'question': data['question'],
        'raptor_tree': all_texts,
        'answer': data['answer'],
        'supporting_facts_context': data['supporting_facts_context']
    }

    embeddings = embd.encode(all_texts)
    query = data['question']
    query_embedding = embd.encode([query])[0]
    distances = distances_from_embeddings(query_embedding, embeddings)
    indices = indices_of_nearest_neighbors_from_distances(distances)
    top_10_texts = [all_texts[i] for i in indices[:10]]

    QA_template = """
    Answer the question based on context.
    Question is:
    {question}
    Context is:
    {context}
    Output format is:
    Answer: [Answer]
    """
    query = QA_template.format(question=data['question'], context=top_10_texts)
    prediction = chat(query)

    print('Prediction', prediction)
    print('label', data['answer'])

    return tree_json

# Main function to run processing in parallel
def main():
    """
    Main function to execute the data processing in parallel.
    """
    import time
    all_trees = []
    start_time = time.time()
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(process_data, data) for data in json_data]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing data"):
            all_trees.append(future.result())
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Run time: {elapsed_time:.2f} seconds")
    with open("Hotpot_HSG.json", "w") as file:
        json.dump(all_trees, file, indent=4)

    print('All trees written to file')

if __name__ == "__main__":
    set_start_method('spawn')
    main()
