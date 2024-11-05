import json
import re
from typing import List, Any


# Define a function to build a query based on node information from the data
def build_node_as_query(data, filter=True):
    """
    Build a query by concatenating the question with the contents of selected nodes.

    Parameters:
    - data: The input data containing nodes, question, and answer.
    - filter: Boolean indicating whether to remove duplicate content.

    Returns:
    - A dictionary containing the constructed query and the answer.
    """
    nodes = data.get('nodes', [])
    question = data.get('question', '')
    answer = data.get('answer', '')

    # Initialize query with question text and context description
    queries = [
        f"Answer the question based on the given content. \nQuestion is: **{question}**. "
        f"Following are documents content."
    ]
    seen_content = set()  # Set to keep track of unique content
    content_counter = 1  # Counter to number content sections

    # Iterate through nodes and add unique content to the query
    for idx, node in enumerate(nodes):
        node_content = node.get('content', '')

        # Add content if it is unique or filtering is disabled
        if not filter or node_content not in seen_content:
            queries.append(f"content [{content_counter}]: {node_content}")
            seen_content.add(node_content)
            content_counter += 1  # Increment content counter

    # Add content from in-edges of each node
    for node in nodes:
        for edge in node.get('in_edges', []):
            in_node_content = edge.get('content', '')

            # Add content if it is unique or filtering is disabled
            if not filter or in_node_content not in seen_content:
                queries.append(f"content [{content_counter}]: {in_node_content}")
                seen_content.add(in_node_content)
                content_counter += 1  # Increment content counter

    # Optionally add content from out-edges 
    # This section can be enabled if out-edge content is also required
    # for node in nodes:
    #     for edge in node.get('out_edges', []):
    #         out_node_content = edge.get('content', '')
    #         if not filter or out_node_content not in seen_content:
    #             queries.append(f"Content[{content_counter}]: {out_node_content}")
    #             seen_content.add(out_node_content)
    #             content_counter += 1

    # Append instructions for answering the question based on the content
    queries.append("Answer the question based on the given contents.\n")
    queries.append(f"Question is: **{question}**\n")
    queries.append(f"Answer in less than 6 words. Your output format is **Answer**: Answer")

    return {'query': queries, 'answer': answer}


# Function to build a query with a more detailed chain of reasoning
def build_node_prompt_as_query(data):
    """
    Build a query prompt based on node information and include chains of reasoning.

    Parameters:
    - data: The input data containing nodes, question, and answer.

    Returns:
    - A dictionary containing the constructed query prompt and the answer.
    """
    nodes = data.get('nodes', [])
    question = data.get('question', '')
    answer = data.get('answer', '')

    # Initialize query with question text and context description
    queries = [
        f"Answer the question based on the given content. "
        f"Question is: {question}. "
        f"Following are documents content."
    ]
    seen_content = set()  # Set to keep track of unique content
    chains = []  # List to store chains of reasoning

    # Add content for each node and track unique content
    for idx, node in enumerate(nodes):
        node_label = node.get('node_label', f"A{idx + 1}")
        node_content = node.get('content', '')

        if node_content not in seen_content:
            queries.append(f"Content[{node_label}]: {node_content}")
            seen_content.add(node_content)

        # Process in-edges and add unique content
        for edge in node.get('in_edges', []):
            in_node_label = edge.get('node_label', '')
            in_node_content = edge.get('content', '')

            if in_node_content not in seen_content:
                queries.append(f"Content[{in_node_label}]: {in_node_content}")
                seen_content.add(in_node_content)

        # Process out-edges and add unique content
        for edge in node.get('out_edges', []):
            out_node_label = edge.get('node_label', '')
            out_node_content = edge.get('content', '')

            if out_node_content not in seen_content:
                queries.append(f"Content[{out_node_label}]: {out_node_content}")
                seen_content.add(out_node_content)

    # Record chains of reasoning for each node based on its edges
    for idx, node in enumerate(nodes):
        node_label = node.get('node_label', f"A{idx + 1}")

        has_in_edge = bool(node.get('in_edges', []))
        has_out_edge = bool(node.get('out_edges', []))

        # Process in-edges and build chains of reasoning
        for edge in node.get('in_edges', []):
            in_node_label = edge.get('node_label', '')

            # Add chain with out-edges if available
            if node.get('out_edges', []):
                for out_edge in node.get('out_edges', []):
                    out_node_label = out_edge.get('node_label', '')
                    chains.append(f"{in_node_label} -> {node_label} -> {out_node_label}")
            else:
                chains.append(f"{in_node_label} -> {node_label}")

        # Consider out-edges if no in-edges are present
        if not has_in_edge:
            for out_edge in node.get('out_edges', []):
                out_node_label = out_edge.get('node_label', '')
                chains.append(f"{node_label} -> {out_node_label}")

        # Add standalone nodes as single chains if they have no edges
        if not has_in_edge and not has_out_edge:
            chains.append(f"{node_label}")

    # Add chains to the query
    queries.append("Here are the supporting chains of reasoning that may help you get to the answer.")
    for chain in chains:
        queries.append(chain)

    queries.append("Answer the question based on the given contents and supporting chains.")
    queries.append(f"Question is: {question}")
    queries.append(f"Answer in less than 6 words. Your output format is **Answer**:[Answer]")

    return {'query': queries, 'answer': answer}


# Function to build detailed chains with content
def build_chains(data):
    """
    Build chains of reasoning from nodes and their edges, including node contents.

    Parameters:
    - data: The input data containing nodes, question, and answer.

    Returns:
    - A dictionary containing question, chains of reasoning, and answer.
    """
    nodes = data.get('nodes', [])
    question = data.get('question', '')
    answer = data.get('answer', '')
    queries = []
    seen_content = set()  # Track unique content
    content_dict = {}  # Map of node labels to their content

    # Step 1: Collect content and build content_dict
    for idx, node in enumerate(nodes):
        node_label = node.get('node_label', f"A{idx + 1}")
        node_content = node.get('content', '')

        if node_content not in seen_content:
            seen_content.add(node_content)
            content_dict[node_label] = node_content
            print(f"Added node content: {node_label} -> {node_content}")

        # Collect content from in-edges and out-edges
        for edge in node.get('in_edges', []):
            in_node_label = edge.get('node_label', '')
            in_node_content = edge.get('content', '')
            if in_node_content not in seen_content:
                seen_content.add(in_node_content)
                content_dict[in_node_label] = in_node_content

        for edge in node.get('out_edges', []):
            out_node_label = edge.get('node_label', '')
            out_node_content = edge.get('content', '')
            if out_node_content not in seen_content:
                seen_content.add(out_node_content)
                content_dict[out_node_label] = out_node_content

    # Step 2: Build chains of reasoning
    chains = {}
    for idx, node in enumerate(nodes):
        node_label = node.get('node_label', f"A{idx + 1}")

        # Handle in-edges and out-edges to build chains
        for edge in node.get('in_edges', []):
            in_node_label = edge.get('node_label', '')

            if node.get('out_edges', []):
                for out_edge in node.get('out_edges', []):
                    out_node_label = out_edge.get('node_label', '')
                    chain = f"{content_dict[in_node_label]} -> {content_dict[node_label]} -> {content_dict[out_node_label]}"
                    label_chain = f"{in_node_label} -> {node_label} -> {out_node_label}"
                    chains.setdefault(node_label, []).append((label_chain, chain))

    # Add chains to the queries
    for node_label, node_chains in chains.items():
        queries.append(f"Chains for node {node_label}:")
        for label_chain, chain in node_chains:
            queries.append(f"{label_chain}: {chain}")

    return {'question': question, 'chains': chains, 'answer': answer}


# Load data and process using build_node_as_query
with open(
        "retrieved_on_ipg_using_bge.json",
        'r', encoding='utf-8') as f:
    data_list = json.load(f)

output_path = '../../LLM_Evaluation/IPG_eval_data/Hotpot/Top5.json'
processed_data = [build_node_as_query(data) for data in data_list]

# Save the processed data
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(processed_data, f, ensure_ascii=False, indent=4)

print('Format done.')
