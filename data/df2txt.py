# This script converts the PoolC dataset to the format required by the UniXCoder 
# clone detection fine-tuning script
import json
import pathlib

import pandas as pd
from tqdm import tqdm

from datasets import load_dataset
from sklearn.model_selection import train_test_split


# Create a directory called "dataset" in the same directory as the current script
dataset_dir = pathlib.Path(__file__).parent / "dataset"
dataset_dir.mkdir(exist_ok=True)

def df2txt(train, val, test=None):

    code_cols = [train["code1"], train["code2"], val["code1"], val["code2"]]
    if test is not None:
        code_cols.extend([test["code1"], test["code2"]])
    arr = pd.concat(code_cols, axis=0, ignore_index=True).unique()
    func_idx = {arr[i]: str(i) for i in range(arr.size)}

    print("Generating data.jsonl")
    with open(dataset_dir / "data.jsonl", "w") as f:
        for func, idx in tqdm(func_idx.items()):
            line = json.dumps({"func": func, "idx": idx}) + "\n"
            f.write(line)

    print("Generating train.txt")
    with open(dataset_dir / "train.txt", "w") as f:
        for idx, row in tqdm(train.iterrows(), total=train.shape[0]):
            f.write(
                f"{func_idx[row['code1']]} {func_idx[row['code2']]} {row['similar']}\n"
            )

    print("Generating valid.txt")
    with open(dataset_dir / "valid.txt", "w") as f:
        for idx, row in tqdm(val.iterrows(), total=val.shape[0]):
            f.write(
                f"{func_idx[row['code1']]} {func_idx[row['code2']]} {row['similar']}\n"
            )

    if test is None:
        return
    print("Generating test.txt")
    with open(dataset_dir / "test.txt", "w") as f:
        for idx, row in tqdm(test.iterrows(), total=test.shape[0]):
            f.write(
                f"{func_idx[row['code1']]} {func_idx[row['code2']]} {row['similar']}\n"
            )


# Load the first fold of the PoolC 5fold dataset
ds = load_dataset("PoolC/1-fold-clone-detection-600k-5fold")
train_test = ds["train"].to_pandas()
val = ds["val"].to_pandas()

# Split the train set into train and test
train, test = train_test_split(
    train_test, test_size=0.2, random_state=42, stratify=train_test["similar"]
)

# Convert the dataframes to txt files for fine-tuning
df2txt(train, val, test)
