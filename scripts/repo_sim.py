import argparse
import os
import pickle
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from numpy import dot
from numpy.linalg import norm
from transformers import pipeline


def cossim(a, b):
    if np.isnan(np.min(a)) or np.isnan(np.min(b)):
        return np.nan
    return dot(a, b) / (norm(a) * norm(b))


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-i",
        "--input",
        nargs="+",
        help="Input repositories",
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

    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)

    model = pipeline(
        model="Lazyhope/RepoSim",
        trust_remote_code=True,
        device_map="auto",
        github_token=os.environ.get("GITHUB_TOKEN"),
    )

    REPOS = args.input

    repo_dataset = model(tuple(REPOS))
    with open(output_dir / "repo_dataset.pkl", "wb") as f:
        pickle.dump(repo_dataset, f)

    if not args.eval:
        return

    if len(REPOS) < 2:
        print("[-] At least 2 repositories are required for evaluation.")
        return

    # Evaluation
    rows_list = []
    for repo1, repo2 in combinations(repo_dataset.keys(), 2):
        info1, info2 = repo_dataset[repo1], repo_dataset[repo2]
        rows_list.append(
            {
                "repo1": repo1,
                "repo2": repo2,
                "topics1": info1["topics"],
                "topics2": info2["topics"],
                "code_sim": cossim(
                    info1["mean_code_embedding"], info2["mean_code_embedding"]
                ),
                "doc_sim": cossim(
                    info1["mean_doc_embedding"], info2["mean_doc_embedding"]
                ),
            }
        )

    df = pd.DataFrame(rows_list)
    df["avg_sim"] = df[["code_sim", "doc_sim"]].mean(axis=1, skipna=True)
    df.to_csv(output_dir / "eval_res.csv", index=False)
    print(f"[+] Evaluation results saved to {output_dir}")


if __name__ == "__main__":
    main()
