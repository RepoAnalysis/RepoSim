# RepoSim

An approach to detect semantically similar python repositories using pre-trained language models.

## About

This repository contains the notebooks and scripts conducted for our approach to detect semantically similar python repositories using pre-trained language models.

Currently our best performing model is [UniXCoder](https://github.com/microsoft/CodeBERT/tree/master/UniXcoder/downstream-tasks/code-search#1-advtest-dataset) fine-tuned on code search task with AdvTest dataset. For evaluations of different language models on repository similarity comparison, please refer to this Jupyter notebook: [notebooks/BiEncoder/Embeddings_evaluation.ipynb](notebooks/BiEncoder/Embeddings_evaluation.ipynb)

## Directory Structure

```bash
RepoSim
├── LICENSE
├── README.md
├── data
│   ├── df2txt.py  # Convert PoolC dataset for clone detection fine-tuning script
│   ├── repo_topic.json # Topic-Repos mapping
│   └── repo_topic.py  # Script to select repos from topics
├── notebooks
│   ├── BiEncoder
│   │   ├── Embeddings_evaluation.ipynb  # Evaluations for comparing different language models
│   │   ├── RepoSim.ipynb  # Our approach's implementation
│   │   └── UnixCoder_C4_Evaluation.ipynb
│   └── CrossEncoder
│       ├── Clone_Detection_C4_Evaluation.ipynb
│       ├── HungarianAlgorithm.ipynb  # Cross-encoder approaches for repo similarity comparison
│       └── keonalgorithms-TheAlgorithmsPython.csv  # Evaluation results by ungarianAlgorithm.ipynb
└── scripts
    ├── LICENSE
    ├── PlayGround.ipynb  # For experimenting with repo embeddings
    ├── README.md
    ├── pipeline.py  # Our approach's implementation as a HuggingFace pipeline
    ├── repo_sim.py
    └── requirements.txt
```

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Acknowledgments

* [GraphCodeBERT](https://arxiv.org/abs/2009.08366)
* [UniXCoder](https://arxiv.org/abs/2203.03850)
* [AdvTest](https://arxiv.org/abs/1909.09436)
* [Sentence Transformers](https://www.sbert.net/)
* [awesome-python](https://github.com/vinta/awesome-python/)
* [Original work of the customized GraphCodeBERT model by @snoop2head](https://github.com/sangHa0411/CloneDetection)
* [Python clone dataset from dacon](https://dacon.io/competitions/official/235900/overview/description)
* [Python clone dataset shared by PoolC](https://huggingface.co/PoolC)
