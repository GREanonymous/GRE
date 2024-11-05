import json
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import re
from sklearn.metrics import precision_score, recall_score, f1_score
from collections import Counter

# Define input path and initialize variables
input_path = "result.json"
data = []
f1_scores = []  # List to store F1 scores
em_scores = []  # List to store Exact Match (EM) scores

# Load the data from JSON file
with open(input_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Initialize ROUGE scorer with stemming
scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

# Initialize BLEU scorer smoothing function
smoothing_function = SmoothingFunction().method4

# Function to calculate F1 score between prediction and reference
def calculate_f1(prediction, reference):
    """
    Calculate F1 score between prediction and reference sentences.

    Parameters:
    - prediction: Predicted answer string.
    - reference: Reference answer string.

    Returns:
    - Tuple containing (precision, recall, f1).
    """
    prediction_tokens = prediction.lower().split()
    reference_tokens = reference.lower().split()

    # Count occurrences of each token in both prediction and reference
    prediction_counter = Counter(prediction_tokens)
    reference_counter = Counter(reference_tokens)

    # Calculate the number of common tokens
    common = prediction_counter & reference_counter
    num_same = sum(common.values())

    # If there are no common tokens, return F1 as 0
    if num_same == 0:
        return 0.0, 0.0, 0.0

    # Calculate Precision, Recall, and F1 score
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(reference_tokens)
    f1 = (2 * precision * recall) / (precision + recall)

    return precision, recall, f1

# Function to extract the answer from a response string
def extract_answer(response):
    """
    Extract answer text from response using regular expressions.

    Parameters:
    - response: The full response string.

    Returns:
    - Extracted answer text.
    """
    # Try different patterns to find the answer in the response
    match = re.search(r'\*\*Answer\*\*: ?(.+)', response)
    if match:
        answer = match.group(1).strip(' "')
    else:
        match = re.search(r'Answer: ?(.+)', response)
        if match:
            answer = match.group(1).strip(' "')
        else:
            match = re.search(r'Answer": ?(.+)', response)
            if match:
                answer = match.group(1).strip(' "')
            else:
                match = re.search(r'Answer\*\*: ?(.+)', response)
                if match:
                    answer = match.group(1).strip(' "')
                else:
                    match = re.search(r'So the answer is?(.+)', response)
                    if match:
                        answer = match.group(1).strip(' "')
                    else:
                        answer = response

    # Clean up unwanted characters from extracted answer
    cleaned_answer = answer.replace('\'', '').replace('.', '').replace('[', '').replace(']', '').replace('Answer', '').replace('*', '').replace(':', '').replace('Answer is:', '').strip()
    return cleaned_answer

# Function to calculate character-level ROUGE-L score
def calculate_char_level_rouge(prediction, reference):
    """
    Calculate character-level ROUGE-L score between prediction and reference.

    Parameters:
    - prediction: Predicted answer string.
    - reference: Reference answer string.

    Returns:
    - Character-level ROUGE-L fmeasure score.
    """
    prediction_chars = ' '.join(list(prediction.replace(' ', '')))
    reference_chars = ' '.join(list(reference.replace(' ', '')))
    char_scores = scorer.score(prediction_chars, reference_chars)
    return char_scores['rougeL'].fmeasure

# Loop through each sample in the data to calculate ROUGE, BLEU, Recall, and F1 scores
results = []
for item in data:
    response = item['response']
    label = item['label']
    try:
        # Calculate word-level ROUGE-L score
        word_rouge_scores = scorer.score(response, label)
        word_rouge_l_score = word_rouge_scores['rougeL'].fmeasure

        # Calculate character-level ROUGE-L score
        char_rouge_l_score = calculate_char_level_rouge(response, label)

        # Calculate BLEU score
        bleu_score = sentence_bleu([label], response, smoothing_function=smoothing_function)

        # Extract answer from response and label
        response_answer = extract_answer(response)
        label_answer = label

        # Calculate Precision, Recall, and F1 score for the answer
        precision, recall, f1 = calculate_f1(response_answer, label_answer)

        # Calculate Exact Match (EM) score
        em_score = 1 if response_answer.strip().lower() == label_answer.strip().lower() else 0

        # Calculate word-level answer ROUGE-L score
        word_answer_rl = scorer.score(response_answer, label_answer)
        word_answer_rl = word_answer_rl['rougeL'].fmeasure

        # Calculate character-level answer ROUGE-L score
        char_answer_rl = calculate_char_level_rouge(response_answer, label_answer)

        # Calculate answer BLEU score
        answer_bleu_score = sentence_bleu([label_answer], response_answer, smoothing_function=smoothing_function)

        # Filter for shorter responses for printing/debugging
        if len(response_answer) < 100:
            print('response:\n', response_answer)
            print('label:\n', label_answer)
            f1_scores.append(f1)
            em_scores.append(em_score)

            # Append results for each sample
            results.append({
                'response': response,
                'label': label,
                'word_rougeL': word_rouge_l_score,
                'char_rougeL': char_rouge_l_score,
                'bleu': bleu_score,
                'word_answer_rl': word_answer_rl,
                'char_answer_rl': char_answer_rl,
                'answer_bleu_score': answer_bleu_score
            })
    except Exception:
        continue  # Skip samples that cause exceptions

# Calculate average scores across all samples
average_word_rouge_l = sum(result['word_rougeL'] for result in results) / len(results)
average_char_rouge_l = sum(result['char_rougeL'] for result in results) / len(results)
average_bleu = sum(result['bleu'] for result in results) / len(results)
average_word_answer_rl = sum(result['word_answer_rl'] for result in results) / len(results)
average_char_answer_rl = sum(result['char_answer_rl'] for result in results) / len(results)
average_answer_bleu = sum(result['answer_bleu_score'] for result in results) / len(results)
average_f1 = sum(f1_scores) / len(f1_scores)
average_em = sum(em_scores) / (len(em_scores)-1)  # Calculate average EM score

# Print average scores for ROUGE-L, BLEU, Recall, F1, and EM
print(f"Average CoT Reasoning Word-level ROUGE-L: {average_word_rouge_l:.4f}")
print(f"Average CoT Reasoning Char-level ROUGE-L: {average_char_rouge_l:.4f}")
print(f"Average CoT Reasoning BLEU: {average_bleu:.4f}")
print(f"Average Answer Word-level ROUGE-L: {average_word_answer_rl:.4f}")
print(f"Average Answer Char-level ROUGE-L: {average_char_answer_rl:.4f}")
print(f"Average Answer BLEU: {average_answer_bleu:.4f}")
print(f"Average F1 Score: {average_f1:.4f}")
print(f"Average Exact Match (EM) Score: {average_em:.4f}")
print(len(results))  # Print the number of valid results processed
