import json
import tiktoken
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from FlagEmbedding import FlagModel
import concurrent.futures
import re
import os
import torch

# Initialize the model for encoding text embeddings
model = FlagModel("../bge-base-en-v1.5", use_fp16=True)
tokenizer = tiktoken.get_encoding('cl100k_base')


def fuzzy_match(query, texts):
    """
    Perform fuzzy matching between a query and a list of texts.

    Parameters:
    - query: The text to match against other texts.
    - texts: A list of texts to compare with the query.

    Returns:
    - List of indices sorted by matching scores.
    """
    scores = [(i, fuzz.partial_ratio(query, text)) for i, text in enumerate(texts)]
    sorted_indices = [i for i, score in sorted(scores, key=lambda x: x[1], reverse=True)]
    return sorted_indices


def tfidf_match(question, texts):
    """
    Calculate TF-IDF based cosine similarity between a question and a list of texts.

    Parameters:
    - question: The question text.
    - texts: A list of texts to compare with the question.

    Returns:
    - List of indices sorted by similarity scores.
    """
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([question] + texts)
    similarity_matrix = cosine_similarity(vectors[0], vectors[1:])[0]
    sorted_indices = np.argsort(similarity_matrix)[::-1]
    return sorted_indices


def clean_text_for_matching(text):
    """
    Clean text by removing extra spaces for better matching.

    Parameters:
    - text: The input text to clean.

    Returns:
    - Cleaned text.
    """
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    text = text.strip()  # Remove leading and trailing spaces
    return text


def check_facts_in_chunks(supporting_facts, chunks):
    """
    Check if supporting facts are present in the chunks.

    Parameters:
    - supporting_facts: List of supporting fact texts.
    - chunks: List of chunk texts.

    Returns:
    - Dictionary indicating presence of each fact in the chunks.
    """
    facts_in_chunks = {}
    cleaned_chunks = [chunk for chunk in chunks]

    for fact in supporting_facts:
        cleaned_fact = clean_text_for_matching(fact)
        facts_in_chunks[fact] = any(cleaned_fact in chunk for chunk in cleaned_chunks)

    return facts_in_chunks


def get_bge_embedding(text):
    """
    Get the BGE (Bidirectional Graph Embedding) for a given text.

    Parameters:
    - text: The input text.

    Returns:
    - The embedding vector of the input text.
    """
    text = text.replace("\n", " ").strip()
    embedding = model.encode([text])
    return embedding


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """
    Calculate the number of tokens in a given string using a specified encoding.

    Parameters:
    - string: The input text string.
    - encoding_name: The encoding type.

    Returns:
    - Number of tokens in the string.
    """
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


# Define the file path for the input JSON file
file_path = '../Eval_Data_Preparation/Hotpot/Hotpot_eval_chunked.json'

# Load the JSON data from the specified file
with open(file_path, 'r', encoding='utf-8') as file:
    json_data = json.load(file)

if __name__ == '__main__':
    sum_token = 0
    retrieved_results = []
    updated_json_data = []
    j = 0

    # Process each item in the JSON data
    for item in json_data:
        data = item
        print(j)
        j += 1
        chunks = []

        # Combine passages from golden, noisy, and distracting contexts
        passages = data['golden_context'] + data['noisy_context'] + data['distracting_context']
        for passage in passages:
            for chunk in passage[1:]:
                chunks.append(chunk)

        question = data['question']
        answer = data['answer']
        supporting_facts = data['supporting_facts_context']

        # Encode the question and chunks
        question_embedding = get_bge_embedding(question)[0]
        chunk_embeddings = [get_bge_embedding(chunk)[0] for chunk in chunks]

        # Calculate cosine similarities
        similarities = cosine_similarity([question_embedding], chunk_embeddings)[0]

        # Get the top-25 similar chunks
        top_15_indices = np.argsort(similarities)[-25:][::-1]
        top_15_chunks = [chunks[i] for i in top_15_indices]

        # Check if supporting facts appear in the top-15 chunks
        # facts_check = check_facts_in_chunks(supporting_facts, top_15_chunks)

        # Print check results in the specified format
        # for i, (fact, is_present) in enumerate(facts_check.items(), start=1):
        #     if is_present:
        #         print(f"Fact {i}: Found")
        #     else:
        #         print(f"Fact {i}: Not Found - {fact}")

        # Update the original data to only include top-15 chunks
        golden_contexts = []
        noisy_context = []
        distracting_contexts = []

        for passage in passages:
            title = passage[0]
            new_passage = []
            for ori_chunk in passage[1:]:
                if any(ori_chunk == top_chunk for top_chunk in top_15_chunks):
                    new_passage.append(ori_chunk)

            # Add the passage to the appropriate context if chunks are retained
            if len(new_passage) > 0:
                is_golden = False
                is_noisy = False

                # Check if the passage belongs to the golden context
                for context in data['golden_context']:
                    if title == context[0]:
                        golden_contexts.append([title] + new_passage)
                        is_golden = True
                        break

                # If not golden, check if it belongs to the noisy context
                if not is_golden:
                    for context in data['noisy_context']:
                        if title == context[0]:
                            noisy_context.append([title] + new_passage)
                            is_noisy = True
                            break

                # If neither golden nor noisy, classify as distracting context
                if not is_golden and not is_noisy:
                    distracting_contexts.append([title] + new_passage)

        # Update data with filtered contexts
        data['golden_context'] = golden_contexts
        data['noisy_context'] = noisy_context
        data['distracting_context'] = distracting_contexts
        updated_json_data.append(data)

    # Write the updated data to a new JSON file
    output_path = '../Eval_Data_Preparation/Hotpot/Hotpot_eval_compressed_chunks.json'
    with open(output_path, 'w', encoding='utf-8') as outfile:
        json.dump(updated_json_data, outfile, ensure_ascii=False, indent=4)
    print('done')
