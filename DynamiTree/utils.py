import os
from openai import AsyncOpenAI
import openai
import json
import pickle
import torch
import asyncio
from tqdm import tqdm
from collections import deque
from sentence_transformers import SentenceTransformer
from .summary_prompt import SUMMARIZE_FACTS
from typing import List, Tuple, Dict, Optional

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TOGETHER_API_KEY"] = ""
os.environ["TOGETHER_BASE_URL"] = ""
os.environ["OPENAI_API_KEY"] = ""
openai.api_key = ""
client = openai.OpenAI(api_key="")


class BPlusTreeNode:
    def __init__(self, leaf=True):
        self.leaf = leaf
        self.keys = []
        self.children = []
        self.facts = []  
        self.scores = []  
        self.summary = ""

class BPlusTree:
    def __init__(self, max_degree):
        self.root = BPlusTreeNode()
        self.max_degree = max_degree
        self.embedding_model = SentenceTransformer("multi-qa-mpnet-base-cos-v1")

    def insert(self, fact):
        embedding = self.embedding_model.encode(fact, convert_to_tensor=True)
        new_node = self._insert_non_full(self.root, embedding, fact)
        if new_node:
            new_root = BPlusTreeNode(leaf=False)
            new_root.keys = [new_node.keys[0]]
            new_root.children = [self.root, new_node]
            self.root = new_root

    def _insert_non_full(self, node, embedding, fact):
        if node.leaf:
            position = self._find_insert_position(node.keys, embedding)
            node.keys.insert(position, embedding)
            node.facts.insert(position, fact)
            node.scores.insert(position, 0.0)  
            if len(node.keys) > self.max_degree:
                return self._split(node)
        else:
            child_index = self._find_insert_position(node.keys, embedding)
            new_node = self._insert_non_full(node.children[child_index], embedding, fact)
            if new_node:
                node.keys.append(new_node.keys[0])
                node.children.append(new_node)
                sorted_pairs = sorted(zip(node.keys, node.children), key=lambda x: torch.norm(x[0], p=2).item())
                node.keys, node.children = map(list, zip(*sorted_pairs))
                if len(node.keys) > self.max_degree:
                    return self._split(node)
        return None

    def _split(self, node):
        mid_index = len(node.keys) // 2
        new_node = BPlusTreeNode(leaf=node.leaf)
        new_node.keys = node.keys[mid_index:]
        node.keys = node.keys[:mid_index]

        if node.leaf:
            new_node.facts = node.facts[mid_index:]
            node.facts = node.facts[:mid_index]
            new_node.scores = node.scores[mid_index:]
            node.scores = node.scores[:mid_index]
        else:
            new_node.children = node.children[mid_index:]
            node.children = node.children[:mid_index]
        return new_node
    
    def summarize_facts(facts, document_type):
        prompt = SUMMARIZE_FACTS.format(document_type=document_type, facts="\n".join(facts))
        response = openai_async_inference([[{"role": "user", "content": prompt}]])[0]
        return response.get("summary", "")

    def _find_insert_position(self, keys, embedding):
        if not keys:
            return 0
        similarities = [torch.dot(embedding, key).item() for key in keys]
        return torch.argmax(torch.tensor(similarities)).item()


    def generate_summaries(self, document_type="QMSum"):
        
        queue = deque([self.root])
        while queue:
            node = queue.popleft()
            if not node.leaf:
                print("Entered Function")
                facts = [child.summary or " ".join(child.facts) for child in node.children]
                if facts:
                    node.summary = summarize_facts(facts, document_type)  # Store summary properly
                    print(node.summary)
                queue.extend(node.children)

    def print_tree_structure(self):
        """ Debugging function to print the tree structure level-wise """
        queue = deque([(self.root, 0)])
        while queue:
            node, level = queue.popleft()
            node_type = "Leaf" if node.leaf else "Intermediate"
            print(f"Level {level}: {node_type} - {len(node.keys)} keys")
            queue.extend([(child, level + 1) for child in node.children])



    def retrieve_summaries(self, node):
        """ Recursively retrieve summaries from all intermediate nodes. """
        summaries = []
        queue = deque([node])

        while queue:
            current_node = queue.popleft()
            if not current_node.leaf and current_node.summary:
                summaries.append(current_node.summary)
            queue.extend(current_node.children)

        return summaries


    def run_initialization_with_summaries(session_paths, output_path, document_type="QMSum"):
        tree = BPlusTree(max_degree=4)
        for session_path in session_paths:
            with open(session_path, "r") as f:
                session_texts = json.load(f)
            for content in session_texts:
                tree.insert(content["content"])
        tree.generate_summaries(document_type=document_type)
        tree.visualize(output_path, "bptree_with_summaries.txt")
        save_tree(tree, "bptree_with_summaries", output_path)


    def retrieve_top_k_facts(self, query: str, k: int) -> List[Tuple[float, str]]:
        query_embedding = self.embedding_model.encode(query, convert_to_tensor=True)
        all_similarities = self._retrieve_k_recursive(self.root, query_embedding)
        all_similarities.sort(reverse=True, key=lambda x: x[0])
        return all_similarities[:k]

    def _retrieve_k_recursive(self, node, query_embedding) -> List[Tuple[float, str]]:
        query_embedding_norm = query_embedding / torch.norm(query_embedding, p=2)
        if node.leaf:
            similarities = [
                (torch.dot(query_embedding_norm, (key / torch.norm(key, p=2))).item(), fact)
                for key, fact in zip(node.keys, node.facts)
            ]
            return similarities
        else:
            facts = []
            for child in node.children:
                facts.extend(self._retrieve_k_recursive(child, query_embedding))
            return facts

    def visualize(self, output_path, filename):
        lines = []
        queue = deque([(self.root, 0)])
        while queue:
            node, level = queue.popleft()
            if node.leaf:
                for fact in node.facts:
                    lines.append(f"{'│   ' * level}├── {fact}")
            else:
                lines.append(f"{'│   ' * level}└── Summary: {node.summary}")
                queue.extend([(child, level + 1) for child in node.children])
        with open(os.path.join(output_path, filename), "w") as f:
            f.write("\n".join(lines))

def save_tree(bptree, name, cache_path):
    try:
        with open(os.path.join(cache_path, f"{name}.pkl"), "wb") as f:
            pickle.dump(bptree, f)
    except Exception as e:
        print(f"Error saving tree: {e}")

def load_tree(filepath):
    try:
        with open(filepath, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Error loading tree from {filepath}: {e}")
        return None


def parse_json(input_str):
        prompt_template = """
        You will be given a string that represents a JSON object but can not be correctly parsed. Your task is to fix the JSON object so that it can be correctly parsed. Here is the string:

        {input_str}

        Please only generate the JSON object without any additional information.
        """
        message = [{
            "role": "user",
            "content": prompt_template.format(input_str=input_str)
        }]
        response = openai_async_inference([message], tqdm_description="Parsing JSON", model_name="gpt-4o-mini", if_parse_json=False)
        return response[0]


def openai_async_inference(messages, tqdm_description="Calling OpenAI API", model_name="gpt-4o-mini", if_parse_json=True):
    """get response with async openai api"""

    if 'gpt' in model_name:
        client = AsyncOpenAI(
            api_key=os.environ.get('OPENAI_API_KEY'),
            organization=os.environ.get('OPENAI_ORG_KEY'),
        )
    else:
        client = AsyncOpenAI(
            api_key=os.environ.get('TOGETHER_API_KEY'),
            base_url=os.environ.get('TOGETHER_BASE_URL'),
        )
    
    async def get_response(msg, index):
        completion = await client.chat.completions.create(
            model=model_name,
            messages=msg,
            response_format={"type": "json_object"}
        )
        return completion.choices[0].message.content, index
    
    async def get_all_responses(msgs):
        tasks = [get_response(msg, i) for i, msg in enumerate(msgs)]

        results = [None] * len(msgs) 
        with tqdm(total=len(tasks), desc=tqdm_description, position=1, leave=False) as pbar:
            for future in asyncio.as_completed(tasks):
                result = await future
                results[result[1]] = result[0] 
                pbar.update(1)

        return results
    
    loop = asyncio.get_event_loop()
    completions = loop.run_until_complete(get_all_responses(messages))

    final_outputs = []
    for completion in completions:
        try:
            if completion.startswith("```json"):
                completion = completion[7:-3]
            final_outputs.append(json.loads(completion.strip()))
        except json.JSONDecodeError:
            if if_parse_json:
                final_outputs.append(parse_json(completion.strip()))
            else:
                final_outputs.append(completion.strip())
    return final_outputs

def summarize_facts(facts, document_type):
        prompt = SUMMARIZE_FACTS.format(facts="\n".join(facts))
        response = openai_async_inference([[{"role": "user", "content": prompt}]])[0]
        return response.get("summary", "")


def get_all_leaf_nodes(bptree):

    leaf_nodes = []
    queue = deque([bptree.root])

    while queue:
        node = queue.popleft()
        if node.leaf:
            leaf_nodes.append(node)
        else:
            queue.extend(node.children)

    return leaf_nodes


def retrieve_top_k_leaf_nodes(bptree, query, k):

    query_embedding = bptree.embedding_model.encode(query, convert_to_tensor=True)
    leaf_nodes = get_all_leaf_nodes(bptree)

    similarities = []
    for leaf in leaf_nodes:
        if not leaf.facts:
            continue
        leaf_embeddings = torch.stack([key for key in leaf.keys])
        avg_embedding = torch.mean(leaf_embeddings, dim=0)
        similarity = torch.dot(query_embedding, avg_embedding).item()
        similarities.append((similarity, leaf))

    
    similarities.sort(reverse=True, key=lambda x: x[0])
    top_k_leaves = [leaf for _, leaf in similarities[:k]]

    return top_k_leaves
