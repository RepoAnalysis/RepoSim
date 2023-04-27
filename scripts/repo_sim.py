import os
import ast
import pickle
import tarfile
import requests
import argparse

import torch
import pandas as pd

from pathlib import Path
from itertools import combinations
from ast import AsyncFunctionDef, FunctionDef, ClassDef, Module

from unixcoder import UniXcoder


API_HEADERS = {"Accept": "application/vnd.github+json"}
if os.environ.get("GITHUB_TOKEN") is None:
    print(
        "[-] Consider setting GITHUB_TOKEN environment variable to avoid hitting rate limits"
    )
    print(
        "For more info, see: https://docs.github.com/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token"
    )
else:
    API_HEADERS["Authorization"] = f"Bearer {os.environ['GITHUB_TOKEN']}"
    print("[+] Using GITHUB_TOKEN for authentication")


def extract_code_and_docs(text: str):
    """Extract code and documentation from a Python file.

    Args:
        text (str): Source code of a Python file

    Returns:
        tuple: A tuple of two sets, the first is the code set, and the second is the docs set,
            each set contains unique code string or docstring, respectively.
    """
    root = ast.parse(text)
    def_nodes = [
        node
        for node in ast.walk(root)
        if isinstance(node, (AsyncFunctionDef, FunctionDef, ClassDef, Module))
    ]

    code_set = set()
    docs_set = set()
    for node in def_nodes:
        docs = ast.get_docstring(node)
        node_without_docs = node
        if docs is not None:
            docs_set.add(docs)
            # Remove docstrings from the node
            node_without_docs.body = node_without_docs.body[1:]
        if isinstance(node, (AsyncFunctionDef, FunctionDef)):
            code_set.add(ast.unparse(node_without_docs))

    return code_set, docs_set


def get_topics(repo_name):
    api_url = f"https://api.github.com/repos/{repo_name}"
    print(f"[+] Getting topics for {repo_name}")
    try:
        response = requests.get(api_url, headers=API_HEADERS)
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        print(f"[-] Failed to get topics for {repo_name}: {e}")
        return []

    metadata = response.json()
    topics = metadata.get("topics", [])

    return topics


def download_and_extract(repo_list):
    temp_tar_path = cwd / "repo.tar"

    repo_info = {}
    for repo_name in repo_list:
        repo_info[repo_name] = {
            "funcs": set(),
            "docs": set(),
            "topic": get_topics(repo_name),
        }

        download_url = f"https://api.github.com/repos/{repo_name}/tarball"
        print(f"[+] Downloading and extracting tags from {repo_name}")
        try:
            response = requests.get(download_url, headers=API_HEADERS, stream=True)
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            print(f"[-] Failed to download {repo_name}: {e}")
            continue

        with open(temp_tar_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=4096):
                f.write(chunk)

        print(f"[+] Extracting {repo_name} info")
        with tarfile.open(temp_tar_path, "r") as tar:
            for member in tar.getmembers():
                if member.isfile() and member.name.endswith(".py"):
                    file_content = tar.extractfile(member).read().decode("utf-8")
                    try:
                        code_set, docs_set = extract_code_and_docs(file_content)
                    except SyntaxError as e:
                        print(f"[-] SyntaxError in {member.name}: {e}, skipping")
                        continue
                    repo_info[repo_name]["funcs"].update(code_set)
                    repo_info[repo_name]["docs"].update(docs_set)

    temp_tar_path.unlink(missing_ok=True)

    return repo_info


def get_embeddings(text):
    tokens_ids = model.tokenize([text], max_length=512, mode="<encoder-only>")
    source_ids = torch.tensor(tokens_ids).to(device)
    _, embeddings = model(source_ids)

    return embeddings


def calculate_mean_embeddings(text_set):
    if not text_set:
        return None

    with torch.no_grad():
        embeddings_list = torch.concat([get_embeddings(text) for text in text_set])

        mean_embeddings = torch.mean(embeddings_list, axis=0)

    return mean_embeddings.cpu()


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "-i",
    "--input",
    nargs="+",
    help="Input repositories, at least 2 repos are required",
    required=True,
)
parser.add_argument("-o", "--output", help="Output directory", required=True)
parser.add_argument(
    "-e",
    "--eval",
    help="Evaluate cosine similarities between all repository combinations",
    action="store_true",
)
args = parser.parse_args()
if len(args.input) < 2:
    print("[-] At least 2 repositories are required as inputs.")
    exit(1)

cwd = Path(__file__).parent
output_dir = Path(args.output)
output_dir.mkdir(exist_ok=True)


device = "cuda" if torch.cuda.is_available() else "cpu"

model = UniXcoder("Lazyhope/unixcoder-nine-advtest")
model.to(device)

REPOS = args.input

# Prepare repos
repo_info = download_and_extract(REPOS)
if len(repo_info) < 2:
    print("[-] Failed to extract info for at least 2 repos")
    exit(1)

# Generate embeddings
for repo_name, repo_dict in repo_info.items():
    print(f"[+] Generating embeddings for {repo_name}")
    if repo_dict.get("code_embeddings") is None:
        repo_dict["code_embeddings"] = calculate_mean_embeddings(repo_dict["funcs"])
    if repo_dict.get("doc_embeddings") is None:
        repo_dict["doc_embeddings"] = calculate_mean_embeddings(repo_dict["docs"])

with open(output_dir / "repo_info.pkl", "wb") as f:
    pickle.dump(repo_info, f)

# Evaluation
if args.eval:
    cossim = torch.nn.CosineSimilarity(dim=0)
    results = []
    for repo1, repo2 in combinations(repo_info.keys(), 2):
        code_embeddings1 = repo_info[repo1]["code_embeddings"]
        code_embeddings2 = repo_info[repo2]["code_embeddings"]
        if code_embeddings1 is None or code_embeddings2 is None:
            code_similarity = None
        else:
            code_similarity = (
                cossim(code_embeddings1, code_embeddings2).cpu().detach().numpy().item()
            )

        doc_embeddings1 = repo_info[repo1]["doc_embeddings"]
        doc_embeddings2 = repo_info[repo2]["doc_embeddings"]
        if doc_embeddings1 is None or doc_embeddings2 is None:
            doc_similarity = None
        else:
            doc_similarity = (
                cossim(doc_embeddings1, doc_embeddings2).cpu().detach().numpy().item()
            )

        results.append((repo1, repo2, code_similarity, doc_similarity))

    df = pd.DataFrame(results, columns=["repo1", "repo2", "code_sim", "doc_sim"])
    df["avg_sim"] = df[["code_sim", "doc_sim"]].mean(axis=1, skipna=True)
    df.to_csv(output_dir / "eval_res.csv", index=False)
    print(f"[+] Evaluation results saved to {output_dir}")
