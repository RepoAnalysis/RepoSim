import json
import torch
import argparse
import subprocess

import pandas as pd

from pathlib import Path
from itertools import combinations

from sentence_transformers import SentenceTransformer

from unixcoder import UniXcoder


def download_repos(repos):
    for repo in repos:
        repo_url = f"https://github.com/{repo}.git"
        try:
            subprocess.run(
                f"git clone {repo_url} {repo}", cwd=repo_path, shell=True, check=True
            )
            print(f"[+] {repo} cloned")
        except subprocess.CalledProcessError as e:
            print(f"[-] Failed to clone {repo}: {e}")


def get_repo_info(repos):
    for repo in repos:
        try:
            subprocess.run(
                f"inspect4py -i {repo_path / repo} -o {repo_info_path / repo} -sc",
                cwd=cwd,
                shell=True,
                check=True,
            )
            print(f"[+] {repo} info extracted")
        except subprocess.CalledProcessError as e:
            print(f"[-] Failed to extract info from {repo}:\n{e}")


def funcs_to_lists(funcs, func_codes, docs):
    for func_name, func_info in funcs.items():
        if func_info.get("source_code") is not None:
            func_codes.append(func_info["source_code"])
        if func_info.get("doc") is None:
            continue
        for key in ["full", "long_description", "short_description"]:
            if func_info["doc"].get(key) is not None:
                docs.append(f"{func_name} {func_info['doc'].get(key)}")
                break


def file_to_lists(filename):
    func_codes = []
    docs = []
    with open(filename, "r") as f:
        dic = json.load(f)
    dic.pop("readme_files", None)
    for dir_name, files in dic.items():
        for file in files:
            if file.get("functions") is not None:
                funcs_to_lists(file["functions"], func_codes, docs)
            if file.get("classes") is not None:
                for class_name, class_info in file["classes"].items():
                    if class_info.get("methods") is not None:
                        funcs_to_lists(class_info["methods"], func_codes, docs)

    return func_codes, docs


def get_code_embeddings(code):
    tokens_ids = code_model.tokenize([code], max_length=512, mode="<encoder-only>")
    source_ids = torch.tensor(tokens_ids).to(device)
    _, embeddings = code_model(source_ids)

    return embeddings


def get_repo_embeddings(lst, input_type):
    if not lst:
        return None
    with torch.no_grad():
        if input_type == "code":
            embeddings_list = torch.concat([get_code_embeddings(code) for code in lst])
        elif input_type == "doc":
            embeddings_list = doc_model.encode(lst, convert_to_tensor=True)

        mean_embeddings = torch.mean(embeddings_list, axis=0)

    return mean_embeddings


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "-i",
    "--input",
    nargs="+",
    help="Input repositories, at least 2 repos are required",
    required=True,
)
parser.add_argument("-o", "--output", help="Output csv file path", required=True)
args = parser.parse_args()
if len(args.input) < 2:
    print("[-] At least 2 repos are required as inputs")
    exit(1)

cwd = Path(__file__).parent
repo_path = cwd / "repos/"
repo_path.mkdir(exist_ok=True)
repo_info_path = cwd / "repo_infos/"
repo_info_path.mkdir(exist_ok=True)


device = "cuda" if torch.cuda.is_available() else "cpu"

doc_model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2", device=device)
code_model = UniXcoder("Lazyhope/unixcoder-nine-advtest")
code_model.to(device)

# REPOS = ["keon/algorithms", "prabhupant/python-ds", "TheAlgorithms/Python"]
REPOS = args.input

# Prepare repos
download_repos(REPOS)
get_repo_info(REPOS)

repo_info = {}
for repo in REPOS:
    repo_info[repo] = {}
    file_path = Path.joinpath(repo_info_path, repo, "directory_info.json")
    function_list, docstring_list = file_to_lists(file_path)
    repo_info[repo]["docs"] = docstring_list
    repo_info[repo]["funcs"] = function_list

# Generate embeddings
for repo_name, repo_dict in repo_info.items():
    print(f"[+] Generating embeddings for {repo_name}")
    if repo_dict.get("code_embeddings") is None:
        repo_dict["code_embeddings"] = get_repo_embeddings(
            repo_dict["funcs"], input_type="code"
        )
    if repo_dict.get("doc_embeddings") is None:
        repo_dict["doc_embeddings"] = get_repo_embeddings(
            repo_dict["docs"], input_type="doc"
        )

# Evaluation
cossim = torch.nn.CosineSimilarity(dim=0, eps=1e-8)
res = []
for repo1, repo2 in combinations(REPOS, 2):
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

    res.append((repo1, repo2, code_similarity, doc_similarity))

df = pd.DataFrame(res, columns=["repo1", "repo2", "code_sim", "doc_sim"])
df["avg_sim"] = df[["code_sim", "doc_sim"]].mean(axis=1, skipna=True)
df.to_csv(Path(__file__).with_name(args.output), index=False)
print(f"[+] Evaluation results saved to {args.output}")
