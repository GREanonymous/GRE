import json
from FlagEmbedding import FlagModel
import os
from typing import Dict, List, Optional, Tuple
import numpy as np
from sklearn.mixture import GaussianMixture
from tqdm import tqdm
from scipy import spatial

def distances_from_embeddings(
    query_embedding: List[float],
    embeddings: List[List[float]],
    distance_metric: str = "cosine"
) -> List[float]:
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
    return np.argsort(distances)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

file_path = 'Hotpot_HSG.json'
output_base_path = '../../LLM_Evaluation/IPG_eval_data//Hotpot/Top{topk}.json'

with open(file_path, 'r', encoding='utf-8') as file:
    all_trees = json.load(file)

bge_embed_model = FlagModel("../bge-base-en-v1.5", use_fp16=True)

topk_values = [5, 10, 15, 20, 25]
processed_data = {topk: [] for topk in topk_values}

# Process each tree to retrieve the top k texts and format a query for evaluating
for tree in tqdm(all_trees, desc="Processing trees"):
    all_texts = tree['raptor_tree']
    query = tree['question']
    label = tree['answer']

    # Generate embeddings for all texts and the query
    cleaned_all_texts = [text.replace("\n", " ").strip() for text in all_texts]
    embeddings = bge_embed_model.encode(cleaned_all_texts)
    print('embeddings', embeddings.shape)
    cleaned_query = query.replace("\n", " ").strip()
    query_embedding = bge_embed_model.encode([cleaned_query])[0]

    # Calculate distances
    distances = distances_from_embeddings(query_embedding, embeddings)

    # Get indices of nearest neighbors
    indices = indices_of_nearest_neighbors_from_distances(distances)

    for topk in topk_values:
        # Retrieve top k texts
        top_k_texts = [all_texts[i] for i in indices[:topk]]

        context_string = ""
        for i, context in enumerate(top_k_texts, start=1):
            context_string += f"context {i}: {context}\n"

        # using top_k_texts to form a query
        QA_template = """Answer the question based on the given contents.
        Contents are as follows:
        {context}
        Question is: **{question}**
        Answer in less than 6 words. Your output format is **Answer**:Answer"
        """
        query_with_context = QA_template.format(question=query, context=context_string)
        data_point = {'query': query_with_context, 'answer': label}
        processed_data[topk].append(data_point)

for topk in topk_values:
    output_path = output_base_path.format(topk=topk)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(processed_data[topk], f, ensure_ascii=False, indent=4)

print('Retrieve Top k on Raptor Tree, Query-Answer Pair Format done.')
