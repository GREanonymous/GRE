import json
import tiktoken
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from FlagEmbedding import FlagModel
import concurrent.futures
import multiprocessing

model = FlagModel("bge-base-en-v1.5", use_fp16=True)
tokenizer = tiktoken.get_encoding('cl100k_base')

def get_bge_embedding(text):
    text = text.replace("\n", " ").strip()
    embedding = model.encode([text])
    return embedding

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

file_path = '../Eval_Data_Preparation/Hotpot/Hotpot_eval_25_chunks.json'

with open(file_path, 'r', encoding='utf-8') as file:
    json_data = json.load(file)
count =0

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    retrieved_results_dict = {5: [], 10: [], 15: [], 20: []}

    for item in json_data:
        print('item', count)
        count = count + 1
        data = item
        chunks = []
        for passage in data['golden_context'] + data['noisy_context'] + data['distracting_context']:
            for chunk in passage[1:]:
                chunks.append(chunk)

        question = data['question']
        answer = data['answer']
        # supporting_facts = data['supporting_facts_context']

        # Encode the question and chunks
        question_embedding = get_bge_embedding(question)[0]
        chunk_embeddings = [get_bge_embedding(chunk)[0] for chunk in chunks]

        # Calculate cosine similarities
        similarities = cosine_similarity([question_embedding], chunk_embeddings)[0]

        # Loop over different top_k values
        for top_k in [5, 10, 15, 20]:
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            top_chunks = [chunks[i] for i in top_indices]

            queries = [
                f"Answer the question based on the given content. Question is: {question}. Following are documents content."]
            for i, idx in enumerate(top_indices):
                queries.append(f"Chunk[{i}]:\n{chunks[idx]}\n")
            queries.append("Answer the question based on the given contents.")
            queries.append(f"Question is: {question}")
            queries.append(f"Answer in less than 6 words. Your output format is **Answer**:[Answer]")

            result = {
                'query': queries,
                'answer': answer
            }
            retrieved_results_dict[top_k].append(result)

            # # Check supporting facts in top_chunks
            # for fact_counter, fact in enumerate(supporting_facts, 1):
            #     found = False
            #     for i, chunk in enumerate(top_chunks):
            #         if fact in chunk:
            #             print(f"Supporting fact {fact_counter} found in chunk[{i}]: {fact}")
            #             found = True
            #             break
            #     if not found:
            #         print(f"Supporting fact {fact_counter} NOT found: {fact}")

    # Save results to separate JSON files
    for top_k, results in retrieved_results_dict.items():
        with open(f'../LLM_Evaluation/Seq_eval_data/Hotpot/Top{top_k}.json', 'w', encoding='utf-8') as outfile:
            json.dump(results, outfile, ensure_ascii=False, indent=4)
    print('done')