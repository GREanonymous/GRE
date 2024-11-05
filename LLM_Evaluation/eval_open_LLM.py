import os
import json
from tqdm import tqdm
from swift.llm import get_model_tokenizer, get_template, inference, ModelType, get_default_template_type


def main(agent, model_type, dataset, structure, topk, ckpt_dir=None):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    print(f'Agent: {agent}')
    print(f'Model Type: {model_type}')

    agent_to_model = {
        'Mixtral':ModelType.mistral_7b_instruct_v2,
        'Llama3': ModelType.llama3_8b_instruct,
        'llama3_1': ModelType.llama3_1_8b_instruct,
        'Qwen2': ModelType.qwen2_7b_instruct,
        'Yi': ModelType.yi_1_5_9b_chat,
    }
    if agent not in agent_to_model:
        raise ValueError("Unsupported agent type")
    model_type_enum = agent_to_model[agent]

    # Get template type based on model type
    template_type = get_default_template_type(model_type_enum)

    # Load the model and tokenizer
    model_kwargs = {'device_map': 'auto'}
    if model_type == 'Pretrained':
        model, tokenizer = get_model_tokenizer(model_type_enum, model_kwargs=model_kwargs)
        print('Using default model')
    elif model_type == 'Finetuned':
        if not ckpt_dir:
            raise ValueError("Checkpoint directory must be provided for finetuned models")
        model, tokenizer = get_model_tokenizer(model_type_enum, model_kwargs=model_kwargs, model_id_or_path=ckpt_dir)
        print('Using fine-tuned model')
    else:
        raise ValueError("Unsupported model type")

    print(f'Template type: {template_type}')
    template = get_template(template_type, tokenizer)

    # Construct paths
    data_path = f'{structure}_eval_data/{dataset}/Top{topk}.json'
    output_file_path = f'{structure}_eval_result/{dataset}/{agent}/{model_type}/Top{topk}_sa.json'
    print(data_path)

    # Load data from JSON file
    with open(data_path, 'r') as f:
        data = json.load(f)

    results = []

    # Perform inference on each query
    for i, item in enumerate(tqdm(data, desc="Processing queries")):
        query = item.get("query")
        if isinstance(query, list):
            query = ", ".join(query)
        label = item.get("answer")
        if query:
            response, _ = inference(model, template, query)
            result_item = {
                "query": query,
                "response": response,
                "label": label
            }
            print(f'Response: {response}\n')
            print(f'Label: {label}\n')
            results.append(result_item)
        else:
            print(f"No query found for item {i + 1}")

    # Save results to JSON file
    if results:
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        with open(output_file_path, 'w') as outfile:
            json.dump(results, outfile, ensure_ascii=False, indent=4)

    print(f"All results saved to: {output_file_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Parameters.")
    parser.add_argument('--agent', type=str, required=True, help="Agent type (Llama3, Qwen2, Mixtral)")
    parser.add_argument('--type', type=str, required=True, help="Model type (Pretrained, Finetuned)")
    parser.add_argument('--dataset', type=str, required=True, help="Dataset name", default="Hotpot")
    parser.add_argument('--structure', type=str, required=True, help="Data structure type (Graph, Raptor, Sequential)")
    parser.add_argument('--topk', type=str, required=True, help="Topk value (10, 15, 20, 25)")
    parser.add_argument('--ckpt_dir', type=str, help="Path to the checkpoint directory for finetuned models")

    args = parser.parse_args()

    main(args.agent, args.type, args.dataset, args.structure, args.topk, args.ckpt_dir)
