# RepoSim

## About

This script takes a list of repositories as input and performs the following operations on each repository:

1. Extracts all function source codes and docstrings using `inspect4py`.
2. Calculates the embeddings for each code and docstring using a fine-tuned `UniXCoder` model and the sentence transformers model `paraphrase-multilingual-mpnet-base-v2`.
3. Averages all docstring embeddings and code embeddings to represent the docstring semantics and code semantics of the repository.

Then the script computes the cosine similarity of the docstring embeddings and code embeddings for each pair of repositories, and calculates the average of these two similarity scores. Results are stored as a csv file in the specified output path.

## Prerequisites & Installation

* Python 3.9+

* pip

    In your virtual environment, run:

    ```sh
    pip install -r requirements.txt
    ```

    to install the required packages.

    (The above command will install cpu-only version of the `pytorch` package. Please refer to [PyTorch's official website](https://pytorch.org/get-started/locally/) for instructions on how to install other versions of `pytorch` on your machine.)

## Usage

```sh
python repo_sim.py --input <repo1> <repo2> ... --output <output_path>
```

For example:

```sh
python repo_sim.py --input keon/algorithms prabhupant/python-ds --output ./res.csv
```

The input is a list of GitHub repository names (at least 2) in the format of `<owner>/<repo>` (e.g. `keon/algorithms`). The output of the script is a csv file containing five columns: `repo1`, `repo2`, `code_sim`, `doc_sim`, and `avg_sim`, representing two repositories and their similarity scores in terms of function source code and docstrings, and the average of the two scores.

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Acknowledgments

* [UniXCoder](https://arxiv.org/abs/2203.03850)
* [Sentence Transformers](https://www.sbert.net/)
