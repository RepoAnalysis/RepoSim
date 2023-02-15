# This script is used to generate a dictionary of repo-topic pairs from the repo_topic.json file
# Repo-topic Source: https://github.com/vinta/awesome-python/blob/master/README.md
import json
from pathlib import Path

with open(Path(__file__).with_name("repo_topic.json")) as f:
    data = json.load(f)

# Specify topics
topics = [
    "Algorithms",
    "Audio",
    "OAuth",
    "Cryptography",
    "Downloader",
    "PDF",
    "Markdown",
]

repo_topic = {repo: topic for topic in topics for repo in data[topic]}

print(f"Selected topics:\n{topics}")
print(f"Repo-topic pairs:\n{repo_topic}")
