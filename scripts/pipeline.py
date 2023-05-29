import ast
import tarfile
from ast import AsyncFunctionDef, ClassDef, FunctionDef, Module

import numpy as np
import requests
import torch
from tqdm.auto import tqdm
from transformers import Pipeline


def extract_code_and_docs(text: str):
    """Extract code and documentation from a Python file.

    Args:
        text (str): Source code of a Python file

    Returns:
        tuple: A tuple of two sets, the first is the code set, and the second is the docs set,
            each set contains unique code string or docstring, respectively.
    """
    code_set = set()
    docs_set = set()
    root = ast.parse(text)
    for node in ast.walk(root):
        if not isinstance(node, (AsyncFunctionDef, FunctionDef, ClassDef, Module)):
            continue
        docs = ast.get_docstring(node)
        node_without_docs = node
        if docs is not None:
            docs_set.add(docs)
            # Remove docstrings from the node
            node_without_docs.body = node_without_docs.body[1:]
        if isinstance(node, (AsyncFunctionDef, FunctionDef)):
            code_set.add(ast.unparse(node_without_docs))

    return code_set, docs_set


def get_metadata(repo_name, headers=None):
    api_url = f"https://api.github.com/repos/{repo_name}"
    tqdm.write(f"[+] Getting metadata for {repo_name}")
    try:
        response = requests.get(api_url, headers=headers)
        response.raise_for_status()

        return response.json()
    except requests.exceptions.HTTPError as e:
        tqdm.write(f"[-] Failed to retrieve metadata from {repo_name}: {e}")
        return {}


def download_and_extract(repos, headers=None):
    extracted_infos = []
    for repo_name in tqdm(repos, disable=len(repos) <= 1):
        # Get metadata
        metadata = get_metadata(repo_name, headers=headers)
        repo_info = {
            "name": repo_name,
            "funcs": set(),
            "docs": set(),
            "topics": [],
            "license": "",
            "stars": metadata.get("stargazers_count"),
        }
        if metadata.get("topics"):
            repo_info["topics"] = metadata["topics"]
        if metadata.get("license"):
            repo_info["license"] = metadata["license"]["spdx_id"]

        # Download repo tarball bytes
        download_url = f"https://api.github.com/repos/{repo_name}/tarball"
        tqdm.write(f"[+] Downloading {repo_name}")
        try:
            response = requests.get(download_url, headers=headers, stream=True)
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            tqdm.write(f"[-] Failed to download {repo_name}: {e}")
            continue

        # Extract python files and parse them
        tqdm.write(f"[+] Extracting {repo_name} info")
        with tarfile.open(fileobj=response.raw, mode="r|gz") as tar:
            for member in tar:
                if (member.name.endswith(".py") and member.isfile()) is False:
                    continue
                try:
                    file_content = tar.extractfile(member).read().decode("utf-8")
                    code_set, docs_set = extract_code_and_docs(file_content)

                    repo_info["funcs"].update(code_set)
                    repo_info["docs"].update(docs_set)
                except UnicodeDecodeError as e:
                    tqdm.write(
                        f"[-] UnicodeDecodeError in {member.name}, skipping: \n{e}"
                    )
                except SyntaxError as e:
                    tqdm.write(f"[-] SyntaxError in {member.name}, skipping: \n{e}")

        extracted_infos.append(repo_info)

    return extracted_infos


class RepoEmbeddingPipeline(Pipeline):
    def __init__(self, github_token=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.API_HEADERS = {"Accept": "application/vnd.github+json"}
        if not github_token:
            print(
                "[*] Consider setting GitHub token to avoid hitting rate limits. \n"
                "For more info, see: "
                "https://docs.github.com/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token"
            )
        else:
            self.set_github_token(github_token)

    def set_github_token(self, github_token):
        self.API_HEADERS["Authorization"] = f"Bearer {github_token}"
        print("[+] GitHub token set")

    def _sanitize_parameters(self, **kwargs):
        _forward_kwargs = {}
        if "max_length" in kwargs:
            _forward_kwargs["max_length"] = kwargs["max_length"]
        if "st_progress" in kwargs:
            _forward_kwargs["st_progress"] = kwargs["st_progress"]

        return {}, _forward_kwargs, {}

    def preprocess(self, inputs):
        if isinstance(inputs, str):
            inputs = [inputs]

        extracted_infos = download_and_extract(inputs, headers=self.API_HEADERS)

        return extracted_infos

    def encode(self, text, max_length):
        """
        Generates an embedding for a input string.

        Parameters:

        * `text`- The input string to be embedded.
        * `max_length`- The maximum total source sequence length after tokenization.
        """
        assert max_length < 1024

        tokenizer = self.tokenizer

        tokens = (
            [tokenizer.cls_token, "<encoder-only>", tokenizer.sep_token]
            + tokenizer.tokenize(text)[: max_length - 4]
            + [tokenizer.sep_token]
        )
        tokens_id = tokenizer.convert_tokens_to_ids(tokens)
        source_ids = torch.tensor([tokens_id]).to(self.device)

        token_embeddings = self.model(source_ids)[0]
        sentence_embeddings = token_embeddings.mean(dim=1)

        return sentence_embeddings

    def _forward(self, extracted_infos, max_length=512, st_progress=None):
        repo_dataset = []
        num_texts = sum(len(x["funcs"]) + len(x["docs"]) for x in extracted_infos)
        with tqdm(total=num_texts) as pbar:
            for repo_info in extracted_infos:
                repo_name = repo_info["name"]
                entry = {
                    "name": repo_name,
                    "topics": repo_info["topics"],
                    "license": repo_info["license"],
                    "stars": repo_info["stars"],
                }

                pbar.set_description(f"Processing {repo_name}")

                tqdm.write(f"[*] Generating embeddings for {repo_name}")

                code_embeddings = []
                for func in repo_info["funcs"]:
                    code_embeddings.append(
                        [func, self.encode(func, max_length).squeeze().tolist()]
                    )

                    pbar.update(1)
                    if st_progress:
                        st_progress.progress(pbar.n / pbar.total)

                entry["code_embeddings"] = code_embeddings
                entry["mean_code_embedding"] = (
                    np.mean([x[1] for x in code_embeddings], axis=0).tolist()
                    if code_embeddings
                    else None
                )

                doc_embeddings = []
                for doc in repo_info["docs"]:
                    doc_embeddings.append(
                        [doc, self.encode(doc, max_length).squeeze().tolist()]
                    )

                    pbar.update(1)
                    if st_progress:
                        st_progress.progress(pbar.n / pbar.total)

                entry["doc_embeddings"] = doc_embeddings
                entry["mean_doc_embedding"] = (
                    np.mean([x[1] for x in doc_embeddings], axis=0).tolist()
                    if doc_embeddings
                    else None
                )

                repo_dataset.append(entry)

        return repo_dataset

    def postprocess(self, repo_dataset):
        return repo_dataset
