import networkx as nx
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import nltk
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity
from FlagEmbedding import FlagModel
from sentence_transformers import SentenceTransformer
import torch
import tiktoken
import os
import pickle
import nltk
import re
import ssl
import time

# Import necessary components from NLTK
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pickle

from collections import deque
from rank_bm25 import BM25Okapi

# Disable SSL verification for unverified HTTPS context
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context


def clean_text_for_matching(text):
    """
    Clean text for matching purposes by removing unnecessary spaces.

    Parameters:
    - text: The input string to be cleaned.

    Returns:
    - A cleaned string with multiple spaces replaced by a single space and leading/trailing spaces removed.
    """
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    text = text.strip()  # Remove leading and trailing spaces
    return text


# Add custom path for nltk data
nltk.data.path.append(r"nltk_data")
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Define the device for computation (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the FlagModel for text encoding
model = FlagModel("bge-base-en-v1.5", use_fp16=True)

# Load the FlagModel with custom instruction for query-based retrieval
model_instruction = FlagModel('../bge-base-en-v1.5',
                              query_instruction_for_retrieval="Represent this sentence for searching relevant passages: ",
                              use_fp16=True)  # Set use_fp16 to True for faster computation with slightly reduced precision


def get_bge_embedding(text):
    """
    Get the BGE (Bi-directional Graph Embedding) embedding for a given text.

    Parameters:
    - text: The input text string.

    Returns:
    - The embedding for the given text as a numpy array.
    """
    text = text.replace("\n", " ").strip()
    embedding = model.encode([text])  # Get the embedding for a single text
    return embedding


def get_sentence_transformer_embedding(texts):
    """
    Get the sentence transformer embeddings for a list of texts.

    Parameters:
    - texts: A list of input text strings.

    Returns:
    - A numpy array containing embeddings for the given texts.
    """
    embeddings = st_model.encode(texts, normalize_embeddings=True)
    return embeddings


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """
    Get the number of tokens for a given string using a specific encoding.

    Parameters:
    - string: The input text string.
    - encoding_name: The encoding format to use (e.g., 'utf-8').

    Returns:
    - The number of tokens in the given string.
    """
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def fact_in_neighbors(fact, node_idx):
    """
    Check if a given fact is present in the neighbors of a specific node.

    Parameters:
    - fact: The fact to search for.
    - node_idx: The index of the node to check.

    Returns:
    - The label of the neighbor node containing the fact, or None if not found.
    """
    # Check the content of the node itself
    if fact in node_contents[node_idx]:
        return node_labels[node_idx]

    # Check the in-neighbors (nodes connected to the current node via incoming edges)
    in_neighbors = [edge[0] for edge in G.in_edges(node_labels[node_idx])]
    for neighbor in in_neighbors:
        neighbor_idx = node_labels.index(neighbor)
        if fact in node_contents[neighbor_idx]:
            return neighbor

    # Check the out-neighbors (nodes connected to the current node via outgoing edges)
    out_neighbors = [edge[1] for edge in G.out_edges(node_labels[node_idx])]
    for neighbor in out_neighbors:
        neighbor_idx = node_labels.index(neighbor)
        if fact in node_contents[neighbor_idx]:
            return neighbor

    return None


def merge_nodes(G, source_node, target_node, new_node_label):
    """
    Merge two nodes in the graph if no path exists between them, creating a new node.

    Parameters:
    - G: The graph to perform the node merging in.
    - source_node: The source node to merge.
    - target_node: The target node to merge with.
    - new_node_label: The label for the new merged node.

    Returns:
    - True if the nodes were successfully merged, False otherwise.
    """
    # Check if a path exists between the source and target nodes
    if nx.has_path(G, source_node, target_node):
        path = nx.shortest_path(G, source=source_node, target=target_node)
        print(f'{source_node} can reach {target_node}. Path: {path}', 'No changes made.')
        return False
    elif nx.has_path(G, target_node, source_node):
        path = nx.shortest_path(G, source=target_node, target=source_node)
        print(f'{target_node} can reach {source_node}. Path: {path}', 'No changes made.')
        return False

    # Get the content of both nodes
    source_content = G.nodes[source_node]['content']
    target_content = G.nodes[target_node]['content']

    # Combine the content of both nodes into a new node
    new_node_content = source_content + target_content
    print(f'source_content: [{source_node}] ', source_content)
    print(f'target_content: [{target_node}] ', target_content)

    # Add the new merged node
    G.add_node(new_node_label, content=new_node_content)
    print(f"node [{source_node}] and node [{target_node}] fuse into {new_node_label}")

    # Reconnect incoming and outgoing edges
    source_in_edges = list(G.in_edges(source_node))
    source_out_edges = list(G.out_edges(source_node))
    target_in_edges = list(G.in_edges(target_node))
    target_out_edges = list(G.out_edges(target_node))

    for edge in source_in_edges:
        G.add_edge(edge[0], new_node_label)
    for edge in target_in_edges:
        G.add_edge(edge[0], new_node_label)

    for edge in source_out_edges:
        G.add_edge(new_node_label, edge[1])
    for edge in target_out_edges:
        G.add_edge(new_node_label, edge[1])

    print('After fusion:', G.nodes[new_node_label]['content'])

    # Remove the original nodes after merging
    G.remove_node(source_node)
    G.remove_node(target_node)
    return True


def retrieve_nodes_with_edges(G, top_indices, node_contents, node_labels, question, answer, supporting_facts,
                              hop2_indices):
    """
    Retrieve nodes and their edges along with related information for the top ranked passages.

    Parameters:
    - G: The graph containing the nodes and edges.
    - top_indices: List of top ranked indices to select relevant nodes.
    - node_contents: List of contents associated with each node.
    - node_labels: List of labels corresponding to each node.
    - question: The input question.
    - answer: The predicted answer.
    - supporting_facts: Supporting facts for the answer.
    - hop2_indices: Indices of nodes that are two hops away.

    Returns:
    - A dictionary containing question, answer, supporting facts, and node data.
    """
    nodes_data = []
    for idx in top_indices:
        node_label = node_labels[idx]
        node_content = node_contents[idx]

        # Get in-edges
        in_edges = list(G.in_edges(node_label))
        in_edges_data = []
        for edge in in_edges:
            in_node_label = edge[0]
            in_node_idx = node_labels.index(in_node_label)
            in_edges_data.append({
                "node_label": in_node_label,
                "content": node_contents[in_node_idx]
            })

        # Get out-edges, excluding original out edges
        out_edges = list(G.out_edges(node_label))
        out_edges_data = []
        for edge in out_edges:
            out_node_label = edge[1]
            out_node_idx = node_labels.index(out_node_label)
            if out_node_idx in hop2_indices:
                out_edges_data.append({
                    "node_label": out_node_label,
                    "content": node_contents[out_node_idx]
                })

        # Append node data including its edges
        nodes_data.append({
            "node_label": node_label,
            "content": node_content,
            "out_edges": out_edges_data,
            "in_edges": in_edges_data
        })

    # Prepare final data dictionary
    data = {
        "question": question,
        "answer": answer,
        "supporting_facts": supporting_facts,
        "nodes": nodes_data
    }
    return data


def process_passage(passage_index, dead_nodes, only_one_pair=False):
    """
    Process the passage and identify the most similar chunk pairs.
    """
    print(f'\nPassage {passage_index + 1}')

    # Identify the most similar passage to the question
    most_similar_passage_index = ranked_indices[passage_index]
    most_similar_passage = (data['golden_context'][most_similar_passage_index]
                            if most_similar_passage_index < len(data['golden_context'])
                            else data['noisy_context'][
        most_similar_passage_index - len(data['golden_context'])])

    # Get chunks from the most similar passage
    most_similar_chunks = most_similar_passage[1:]
    print("Most similar chunks to each chunk in the most similar passage:")

    max_similarity = -1
    best_chunk_pair = None

    # Compare each chunk with others to find the most similar pair
    for i, chunk in enumerate(most_similar_chunks):
        chunk_index = chunks.index(chunk)

        for j, other_chunk in enumerate(chunks):
            if (chunk_labels[chunk_index] != chunk_labels[j] and
                    chunk_node_mapping[j] not in dead_nodes and
                    chunk_node_mapping[chunk_index] not in dead_nodes):
                if similarity_matrix[chunk_index, j] > max_similarity:
                    max_similarity = similarity_matrix[chunk_index, j]
                    best_chunk_pair = (chunk_index, j)

    # If a similar chunk pair is found, merge the nodes
    if best_chunk_pair is not None and max_similarity > 0.0:
        chunk_index, most_similar_chunk_index = best_chunk_pair
        print(
            f"    {chunk_node_mapping[chunk_index]} Most similar to chunk ({chunk_node_mapping[most_similar_chunk_index]}) with similarity {max_similarity:.4f}")
        print(
            f"    Source node label: {chunk_node_mapping[chunk_index]}; Merge node label: {chunk_node_mapping[most_similar_chunk_index]}")
        target_node = chunk_node_mapping[chunk_index]
        source_node = chunk_node_mapping[most_similar_chunk_index]
        new_node_label = f'{source_node}_{target_node}'
        is_merged = merge_nodes(G, source_node, target_node, new_node_label)
        if is_merged:
            print(f"{source_node} and {target_node} have been merged.")
            dead_nodes.append(source_node)
            dead_nodes.append(target_node)
        else:
            print(f"{source_node} and {target_node} are connected and do not need to be aggregated.")
    else:
        print("    No similar chunk found.")

    return dead_nodes

def preprocess_text(text):
    """
    Preprocess text by tokenizing, lemmatizing, and removing stopwords.
    """
    tokens = word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalpha() and word not in stop_words]
    return ' '.join(tokens)

total_token = 0
total_doc_token = 0
sum_node = 0
all_saved_graphs = []
graph_dict = {}

input_path = r'../Eval_Data_Preparation/Hotpot/Hotpot_eval_compressed_chunks.json'
with open(input_path, 'r', encoding='utf-8') as file:
    json_data = json.load(file)

start_time = time.time()

for item in json_data:
    data = item
    documents = []

    # Combine golden and noisy contexts into one list of documents
    data['noisy_context'].extend(data['distracting_context'])
    for passage in data['golden_context'] + data['noisy_context']:
        combined_text = ' '.join(passage)
        documents.append(combined_text)

    # Create a TF-IDF vectorizer and fit it to the documents
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)

    # Transform the question into the same TF-IDF space
    question_tfidf = vectorizer.transform([data['question']])

    # Compute cosine similarity between the question and each passage
    cosine_similarities = np.dot(question_tfidf, tfidf_matrix.T).toarray()[0]

    # Rank passages based on their similarity to the question
    ranked_indices = np.argsort(cosine_similarities)[::-1]

    """
    Create graph with nodes and edges representing the passages
    """
    G = nx.DiGraph()

    # Add nodes and edges for each passage, labeled by their relevance order
    for i, idx in enumerate(ranked_indices):
        passage = data['golden_context'][idx] if idx < len(data['golden_context']) else data['noisy_context'][
            idx - len(data['golden_context'])]
        label_prefix = chr(65 + i)  # 'A' for first passage, 'B' for second, etc.
        for j in range(1, len(passage)):
            node_label = f'{label_prefix}{j}'
            G.add_node(node_label, content=passage[j])
            if j > 1:
                prev_node_label = f'{label_prefix}{j - 1}'
                G.add_edge(prev_node_label, node_label)

    """
    Preprocess text and compute topic similarity between chunks
    """

    chunks = []
    chunk_labels = []

    # Extract chunks and labels from the ranked passages
    for idx in ranked_indices:
        passage = data['golden_context'][idx] if idx < len(data['golden_context']) else data['noisy_context'][
            idx - len(data['golden_context'])]
        chunks.extend(passage[1:])
        chunk_labels.extend([passage[0]] * (len(passage) - 1))

    chunk_counter = 1
    chunk_node_mapping = []

    # Generate node labels for each chunk
    for i, idx in enumerate(ranked_indices):
        passage = data['golden_context'][idx] if idx < len(data['golden_context']) else data['noisy_context'][
            idx - len(data['golden_context'])]
        label_prefix = chr(65 + i)  # 'A' for first passage, 'B' for second, etc.
        for j in range(1, len(passage)):
            node_label = f'{label_prefix}{j}'
            chunk_node_mapping.append(node_label)

        chunk_counter += 1

    # Preprocess the chunks and compute the similarity matrix
    preprocessed_chunks = [preprocess_text(chunk) for chunk in chunks]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(preprocessed_chunks)
    similarity_matrix = cosine_similarity(tfidf_matrix)

    # Set similarity to infinity for identical chunks
    for i, label_i in enumerate(chunk_labels):
        for j, label_j in enumerate(chunk_labels):
            if label_i == label_j:
                similarity_matrix[i, j] = float('inf')

    """
    Find the most similar chunk pair for each chunk in the most similar passage
    """

    """
        Find the most similar chunk pair for each chunk in the most similar passage
    """

    dead_nodes = []
    num_passages_to_process = len(ranked_indices)
    only_one_pair = True  # Set this to True or False based on your requirement

    # Process each passage and identify the most similar chunks
    for passage_index in range(num_passages_to_process):
        dead_nodes = process_passage(passage_index, dead_nodes, only_one_pair=only_one_pair)
    print(dead_nodes)

    graph_dict[item['question']] = G

    node_contents = []
    node_labels = []

    # Extract content and labels for each node in the graph
    for node in G.nodes(data=True):
        node_label = node[0]
        node_content = node[1]['content']
        node_contents.append(node_content)
        node_labels.append(node_label)

    # Define the question and supporting facts context
    question = item['question']
    supporting_facts_context = item['supporting_facts_context']

    # Select the retriever method
    # retriever = 'bge_2_hop'
    retriever = 'bge'

    # TF-IDF retrieval
    if retriever == 'TF-IDF':
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(node_contents)
        question_tfidf = vectorizer.transform([question])
        cosine_similarities = cosine_similarity(question_tfidf, tfidf_matrix).flatten()
        top_indices = np.argsort(cosine_similarities)[-5:][::-1]

    # BM25 retrieval
    elif retriever == 'BM25':
        tokenized_contents = [doc.split() for doc in node_contents]
        bm25 = BM25Okapi(tokenized_contents)
        tokenized_question = question.split()
        bm25_scores = bm25.get_scores(tokenized_question)
        top_indices = np.argsort(bm25_scores)[-5:][::-1]

    # BGE retrieval
    elif retriever == 'bge':
        question_embedding = get_bge_embedding(question).flatten()
        node_embeddings = np.array([get_bge_embedding(content).flatten() for content in node_contents])
        cosine_similarities = cosine_similarity([question_embedding], node_embeddings).flatten()
        top_indices = np.argsort(cosine_similarities)[-5:][::-1]

    # BGE instruction-based retrieval
    elif retriever == 'bge_instruction':
        question_embedding = model_instruction.encode_queries([question]).flatten()
        node_embeddings = np.array([model_instruction.encode([content]).flatten() for content in node_contents])
        cosine_similarities = cosine_similarity([question_embedding], node_embeddings).flatten()
        top_indices = np.argsort(cosine_similarities)[-5:][::-1]
        hop2_indices = []

    # BGE with 2-hop retrieval
    elif retriever == 'bge_2_hop':
        question_embedding = model_instruction.encode_queries([question]).flatten()
        node_embeddings = np.array([model_instruction.encode([content]).flatten() for content in node_contents])
        cosine_similarities = cosine_similarity([question_embedding], node_embeddings).flatten()
        top_indices = np.argsort(cosine_similarities)[-5:][::-1]
        hop2_indices = []
        hop2_similarities = []

        for idx in top_indices:
            new_query = question + " " + node_contents[idx]
            new_query_embedding = model_instruction.encode_queries([new_query]).flatten()
            new_cosine_similarities = cosine_similarity([new_query_embedding], node_embeddings).flatten()
            new_top_indices = np.argsort(new_cosine_similarities)[::-1]

            neighbors = set()
            neighbors.update(set(G.successors(node_labels[idx])))
            neighbors.update(set(G.predecessors(node_labels[idx])))

            # Filter out indices that are already in the top indices or their neighbors
            filtered_indices = [new_idx for new_idx in new_top_indices if
                                new_idx not in top_indices and node_labels[new_idx] not in neighbors]
            filtered_indices = filtered_indices[:1]
            hop2_indices.extend(filtered_indices)
            hop2_similarities.extend(new_cosine_similarities[filtered_indices])

            node_label = node_labels[idx]
            # Add edges for the filtered indices
            if len(filtered_indices) > 0:
                G.add_edge(node_label, node_labels[filtered_indices[0]])
            if len(filtered_indices) > 1:
                G.add_edge(node_label, node_labels[filtered_indices[1]])
            if len(filtered_indices) > 2:
                G.add_edge(node_label, node_labels[filtered_indices[2]])

    encoding_name = 'cl100k_base'
    print('top_indices:\n', top_indices)
    print('hop2_indices:\n', hop2_indices)
    print("Most similar nodes:")

    # Display the most similar nodes and their similarity scores
    for idx in top_indices:
        print(f"Node {node_labels[idx]}: {node_contents[idx]}")
        if retriever == 'TF-IDF':
            print(f"Similarity Score: {cosine_similarities[idx]:.4f}")
        elif retriever == 'BM25':
            print(f"BM25 Score: {bm25_scores[idx]:.4f}")
        elif retriever == 'bge' or retriever == 'bge_instruction':
            print(f"Cosine Similarity: {cosine_similarities[idx]:.4f}")
        print()

    # Calculate token and node counts, excluding out-edges
    token_in_one_data = 0
    node_in_one_data = 0
    counted_nodes = set()

    # Process the top retrieved nodes and their edges
    for idx in top_indices:
        node_label = node_labels[idx]
        print(f"\nFind in edge and out edge for Top retrieved node {node_label}:")
        counted_nodes.add(node_label)
        token_in_one_data += num_tokens_from_string(node_contents[idx], encoding_name)

        # Process in-edges
        in_edges = list(G.in_edges(node_label))
        print("  In edges:")
        for edge in in_edges:
            in_node_label = edge[0]
            if in_node_label not in counted_nodes:
                in_node_content = node_contents[node_labels.index(in_node_label)]
                token_in_one_data += num_tokens_from_string(in_node_content, encoding_name)
                counted_nodes.add(in_node_label)
            print(f"    {in_node_label} -> {node_label}")

        # Don't process out-edges anymore
        out_edges = list(G.out_edges(node_label))
        print("  Don't process out edges anymore")
        for edge in out_edges:
            out_node_label = edge[1]
            if out_node_label in [node_labels[i] for i in hop2_indices] and out_node_label not in counted_nodes:
                out_node_content = node_contents[node_labels.index(out_node_label)]
                token_in_one_data += num_tokens_from_string(out_node_content, encoding_name)
                counted_nodes.add(out_node_label)
            print(f"    {node_label} -> {out_node_label}")

    # Process 2-hop indices
    if retriever == 'bge_2_hop':
        for idx in hop2_indices:
            node_label = node_labels[idx]
            if node_label not in counted_nodes:
                token_in_one_data += num_tokens_from_string(node_contents[idx], encoding_name)
                counted_nodes.add(node_label)

    node_in_one_data = len(counted_nodes)
    print(f"Nodes in this data: {node_in_one_data}\n")
    print(f"Token in this data: {token_in_one_data}\n")
    total_token += token_in_one_data
    sum_node += node_in_one_data
    print(f"Sum Retrieved Nodes num: {sum_node}\n")
    print(f"Sum Retrieved Token num: {total_token}\n")
    print()

    # Calculate total token count in the graph
    for node in G.nodes(data=True):
        node_content = node[1]['content']
        total_doc_token += num_tokens_from_string(node_content, encoding_name)

    print(f"Total token count in the graph: {total_doc_token}")

    # Retrieve data from the graph
    answer = data['answer']
    retrieved_data = retrieve_nodes_with_edges(G, top_indices, node_contents, node_labels, question, answer,
                                               supporting_facts_context, hop2_indices)
    all_saved_graphs.append(retrieved_data)

    # Calculate and print the total processing time
    end_time = time.time()
    print("Total time", end_time - start_time)

    # Save the retrieved graph data to a file
    output_file = "retrieved_on_ipg_using_bge.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_saved_graphs, f, ensure_ascii=False, indent=4)

    print(f"Data saved to {output_file}")

