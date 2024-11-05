import json
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
import spacy
import nltk
from typing import List, Any
from Text_spliter import SpacyTextSplitter
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Add a custom path to the nltk data directory
nltk.data.path.append("nltk_data")

# Load the English language model from SpaCy
nlp = spacy.load("en_core_web_sm")


def clean_text(text: str) -> str:
    """
    Clean the given text by removing unwanted characters and sections.

    Parameters:
    - text: The input text string to be cleaned.

    Returns:
    - A cleaned text string with unwanted sections and special characters removed.
    """
    # Remove specific characters like '*' and '—'
    text = re.sub(r'[*—]', '', text)

    # Remove specific reference sections such as "References" or "External links"
    for ref in ["References", "External links", 'Alternate titles', 'See also']:
        text = re.sub(r'==\s*' + re.escape(ref) + r'\s*==', '', text, flags=re.IGNORECASE)

    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)

    # Collapse multiple spaces into a single space and strip leading/trailing spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def process_context(context: List[str], item_index: int, context_index: int, chunk_size: int, separator: str,
                    chunk_overlap: int) -> List[str]:
    """
    Process and split a given context into smaller text chunks using SpaCy.

    Parameters:
    - context: A list containing the title and passage as strings.
    - item_index: The index of the current item in the data list.
    - context_index: The index of the current context within the item.
    - chunk_size: The size of each text chunk.
    - separator: The separator used to split the text.
    - chunk_overlap: The overlap size between consecutive chunks.

    Returns:
    - A list of text chunks, including the title followed by the chunks.
    """
    try:
        # Extract title and passage from the context
        title, passage = context
        combined_text = f"{title}. {passage}"  # Combine title and passage for potential use
        cleaned_text = passage  # Use only the passage for splitting

        # Initialize the text splitter with the given parameters
        text_splitter = SpacyTextSplitter(
            separator=' ',
            pipeline='en_core_web_sm',
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            min_chunk_size=64
        )

        # Split the cleaned text into chunks
        chunks = text_splitter.split_text(cleaned_text)

        print(f"Processing item {item_index}, context {context_index}: {title}")

        # Return a list with the title followed by the chunks
        return [title] + chunks
    except Exception as e:
        # Handle exceptions and return an error message
        print(f"Error processing item {item_index}, context {context_index}: {e}")
        return [f"Error: {e}"]


# File paths for input and output
input_file_path = 'Hotpot/Hotpot_eval.json'
output_file_path = 'Hotpot/Hotpot_eval_chunked.json'

# Load the input data from the JSON file
with open(input_file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

new_data = []

# Use ProcessPoolExecutor for parallel processing with up to 32 workers
with ProcessPoolExecutor(max_workers=32) as executor:
    futures = []
    for item_index, item in enumerate(data):
        # Create a new item with question, answer, and context placeholders
        new_item = {
            'question': item['question'],
            'answer': item['answer'],
            'supporting_facts_context': item['supporting_facts_context'],
            'golden_context': [],
            'noisy_context': [],
            'distracting_context': []
        }

        # Submit tasks for processing each context type
        for context_type in ['golden_context', 'distracting_context', 'noisy_context']:
            for context_index, context in enumerate(item[context_type]):
                future = executor.submit(process_context, context, item_index, context_index, 256, " ", 64)
                futures.append((future, context_type, new_item))

        # Add the new item to the list
        new_data.append(new_item)

    # Collect results as they are completed
    for future in as_completed([f[0] for f in futures]):
        try:
            # Retrieve the result from the future
            result = future.result()
            # Find the corresponding context type and new item in the futures list
            context_type = next(f[1] for f in futures if f[0] == future)
            new_item = next(f[2] for f in futures if f[0] == future)
            # Append the result to the appropriate context list in the new item
            new_item[context_type].append(result)
        except Exception as e:
            # Handle exceptions during future processing
            print(f"Error processing future: {e}")

# Save the processed data to the output JSON file
with open(output_file_path, 'w', encoding='utf-8') as file:
    json.dump(new_data, file, ensure_ascii=False, indent=4)

print('done')
