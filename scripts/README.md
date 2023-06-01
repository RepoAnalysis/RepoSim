# RepoSim

## About

This script takes a list of repositories as input and performs the following operations on each repository:

1. Extracts all function source codes and docstrings from the github repository.
2. Calculates the embeddings for each code and docstring using a `UniXCoder` model fine-tuned on the nl-code-search task.
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

## Usage

```sh
python repo_sim.py --input <repo1> <repo2> ... --output <output_dir> [--eval]
```

For example:

```sh
python repo_sim.py --input keon/algorithms prabhupant/python-ds --output ./
```

The input is a list of GitHub repository names (at least 2) in the format of `<owner>/<repo>` (e.g. `keon/algorithms`). The output of the script is a python pickle file `<output_dir>/output.pkl` which stores a list of dictionaries containing all repositories' information, including name, topics, license, stars, extracted function/docstring list and their corresponding embeddings, as well as the mean code/docstrings embedding. This file can be used for later experiments such as semantic similarity search/comparison.

When `--eval` is specified, the script will also save a csv file with five columns: `repo1`, `repo2`, `code_sim`, `doc_sim`, and `avg_sim`, representing two repositories and their similarity scores in terms of function source code and docstrings, and the average of the two scores. This file will compare each pair of repositories in the input list and save the results at `<output_dir>/eval_res.csv`.

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Acknowledgments

* [UniXCoder](https://arxiv.org/abs/2203.03850)
* [Sentence Transformers](https://www.sbert.net/)
* [awesome-python](https://github.com/vinta/awesome-python/)
