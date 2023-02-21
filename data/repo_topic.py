# This script is used to generate a dictionary of repo-topic pairs from the repo_topic.json file
# Repo-topic Source: https://github.com/vinta/awesome-python/blob/master/README.md
import json
import argparse
from pathlib import Path


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "-t",
    "--topics",
    nargs="+",
    help="Select topics to get corresponding repos",
    default=None,
)

topics = parser.parse_args().topics

data = {}
with open(Path(__file__).with_name("repo_topic.json")) as f:
    data = json.load(f)
    if topics is None:
        topics = list(data.keys())
print(f"Selected topics:\n{topics}\n")

repo_topic = {}
for topic in topics:
    if topic not in data:
        print(f"[-] Topic '{topic}' not found")
        exit(1)
    repo_topic.update({repo: topic for repo in data[topic]})

print(f"Repo-topic pairs:\n{repo_topic}")
