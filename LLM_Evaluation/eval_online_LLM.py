import os
import json
import time
import requests
from tqdm import tqdm
from openai import OpenAI
import concurrent.futures

def deepseek_chat(query, api_key):
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
    )
    setting = [{
        "role": "user",
        "content": query
    }]
    try:
        completion = client.chat.completions.create(
            model="deepseek-chat",
            messages=setting,
            temperature=0.,
            max_tokens=20
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


def moonshotai_chat(query, api_key):
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.moonshot.cn/v1"
    )
    setting = [{
        "role": "user",
        "content": query
    }]
    try:
        completion = client.chat.completions.create(
            model="moonshot-v1-8k",
            messages=setting,
            temperature=0.0,
            max_tokens=20
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



def openai_chat(query, api_key, model="gpt-3.5-turbo"):
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.ohmygpt.com/v1",
    )
    setting = [{
        "role": "user",
        "content": query
    }]
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=setting,
            temperature=0.,
            max_tokens=20
        )
        result = completion.choices[0].message.content
        return result
    except Exception as e:
        error_message = str(e)
        print(f"Request failed: {error_message}. Retrying in 10 seconds...")
        time.sleep(1)
        return None


# gpt4o_mini分析函数
def OpenAI_chat(prompt, api_key, max_retries=3):
    url = "https://cn2us02.opapi.win/v1/chat/completions"
    payload = json.dumps({
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    })
    headers = {
        'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
        'Content-Type': 'application/json',
        "Authorization": 'Bearer ' + api_key,
    }

    retries = 0
    while retries < max_retries:
        try:
            response = requests.post(url, headers=headers, data=payload)
            response.raise_for_status()  # Raise an HTTPError for bad responses
            res = response.json()

            # Ensure the response contains the 'choices' key
            if 'choices' in res and len(res['choices']) > 0:
                res_content = res['choices'][0]['message']['content']
                return res_content
            else:
                error_message = f"Unexpected response structure: {res}"
                print(error_message)
                return None

        except requests.exceptions.RequestException as e:
            error_message = f"Request failed: {str(e)}"
            print(f"{error_message}. Retrying in 10 seconds... (Attempt {retries + 1}/{max_retries})")
            time.sleep(3)
            retries += 1

        except json.JSONDecodeError as e:
            error_message = f"Failed to decode JSON response: {str(e)}"
            print(error_message)
            return None

    # If the maximum number of retries is reached without success
    print(f"Maximum retries reached. Failed to process prompt: {prompt}")
    return None

def process_query(item, api_type, api_key):
    query = item.get("query")
    if isinstance(query, list):
        query = ", ".join(query)
    label = item.get("answer")
    if query:
        if api_type == 'Deepseek':
            api_key = 'your-key-here'
            response = deepseek_chat(query, api_key)
        elif api_type == 'MoonshotAI':
            api_key = 'your-key-here'
            response = moonshotai_chat(query, api_key)
        elif api_type == 'GPT':
            api_key = 'your-key-here'
            response = OpenAI_chat(query, api_key)
        else:
            raise ValueError("Unsupported API type")

        result_item = {
            "query": query,
            "response": response,
            "label": label
        }
        return result_item
    else:
        print(f"No query found for item {i + 1}")
        return None


def main(api_type, dataset, structure, topk, api_key):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    print(f'API Type: {api_type}')

    data_path = 'Full_eval/2WikiMQA_full_doc.json'
    output_file_path = 'Full_eval/2WikiMQA_full_result.json'

    with open(data_path, 'r') as f:
        data = json.load(f)

    results = []

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_query, item, api_type, api_key) for item in data]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=f"Processing queries with {api_type}"):
            result = future.result()
            if result:
                results.append(result)

    if results:
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        with open(output_file_path, 'a') as outfile:
            json.dump(results, outfile, ensure_ascii=False, indent=4)

    print("All results saved to:", output_file_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process some parameters.")
    parser.add_argument('--api_type', type=str, required=True, help="API type (MoonshotAI, Deepseek, GPT)")
    parser.add_argument('--dataset', type=str, required=True, help="Datasets", default="Hotpot")
    parser.add_argument('--structure', type=str, required=True, help="Graph, Raptor, Sequential")
    parser.add_argument('--topk', type=str, required=True, help="Topk 10, 15, 20, 25")
    parser.add_argument('--api_key', type=str, required=True, help="API key for the selected API")

    args = parser.parse_args()

    main(args.api_type, args.dataset, args.structure, args.topk, args.api_key)
