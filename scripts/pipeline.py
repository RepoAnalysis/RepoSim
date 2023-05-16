import ast
import tarfile
from ast import AsyncFunctionDef, ClassDef, FunctionDef, Module
from io import BytesIO

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


def get_topics(repo_name, headers=None):
    api_url = f"https://api.github.com/repos/{repo_name}"
    print(f"[+] Getting topics for {repo_name}")
    try:
        response = requests.get(api_url, headers=headers)
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        print(f"[-] Failed to get topics for {repo_name}: {e}")
        return []

    metadata = response.json()
    topics = metadata.get("topics", [])
    if topics:
        print(f"[+] Topics found for {repo_name}: {topics}")

    return topics


def download_and_extract(repos, headers=None):
    extracted_info = {}
    for repo_name in repos:
        extracted_info[repo_name] = {
            "funcs": set(),
            "docs": set(),
            "topics": get_topics(repo_name, headers=headers),
        }

        download_url = f"https://api.github.com/repos/{repo_name}/tarball"
        print(f"[+] Extracting functions and docstrings from {repo_name}")
        try:
            response = requests.get(download_url, headers=headers, stream=True)
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            print(f"[-] Failed to download {repo_name}: {e}")
            continue

        repo_bytes = BytesIO(response.raw.read())
        print(f"[+] Extracting {repo_name} info")
        with tarfile.open(fileobj=repo_bytes) as tar:
            for member in tar.getmembers():
                if member.isfile() and member.name.endswith(".py"):
                    file_content = tar.extractfile(member).read().decode("utf-8")
                    try:
                        code_set, docs_set = extract_code_and_docs(file_content)
                    except SyntaxError as e:
                        print(f"[-] SyntaxError in {member.name}: {e}, skipping")
                        continue
                    extracted_info[repo_name]["funcs"].update(code_set)
                    extracted_info[repo_name]["docs"].update(docs_set)

    return extracted_info


class RepoEmbeddingPipeline(Pipeline):
    def __init__(self, github_token=None, st_messager=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Streamlit single element container created by st.empty()
        self.st_messager = st_messager

        self.API_HEADERS = {"Accept": "application/vnd.github+json"}
        if not github_token:
            message = (
                "[*] Consider setting GitHub token to avoid hitting rate limits. \n"
                "For more info, see: "
                "https://docs.github.com/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token"
            )
            print(message)
            if self.st_messager:
                self.st_messager.info(message)
        else:
            self.set_github_token(github_token)

    def set_github_token(self, github_token):
        self.API_HEADERS["Authorization"] = f"Bearer {github_token}"
        message = "[+] GitHub token set"
        print(message)
        if self.st_messager:
            self.st_messager.success(message)

    def _sanitize_parameters(self, **kwargs):
        _forward_kwargs = {}
        if "max_length" in kwargs:
            _forward_kwargs["max_length"] = kwargs["max_length"]
        if "st_progress" in kwargs:
            _forward_kwargs["st_progress"] = kwargs["st_progress"]

        return {}, _forward_kwargs, {}

    def preprocess(self, inputs):
        if isinstance(inputs, str):
            inputs = (inputs,)

        if self.st_messager:
            self.st_messager.info("[*] Downloading and extracting repos...")
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
        repo_dataset = {}
        num_texts = sum(
            len(x["funcs"]) + len(x["docs"]) for x in extracted_infos.values()
        )
        with tqdm(total=num_texts) as pbar:
            for repo_name, repo_info in extracted_infos.items():
                pbar.set_description(f"Processing {repo_name}")
                entry = {"topics": repo_info.get("topics")}

                message = f"[*] Generating embeddings for {repo_name}"
                tqdm.write(message)
                if self.st_messager:
                    self.st_messager.info(message)

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

                repo_dataset[repo_name] = entry

        return repo_dataset

    def postprocess(self, repo_dataset):
        return repo_dataset
