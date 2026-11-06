# PyTorch Deep Learning

End-to-end PyTorch deep learning engineering: workflows, neural network classification, computer vision, custom datasets, modular architectures, experiment tracking, and paper replication.

Based on the [Learn PyTorch for Deep Learning](https://www.learnpytorch.io/) curriculum by Daniel Bourke.

---

## Repository Structure

```text
pytorch-deep-learning/
├── 00_fundamentals/
│   ├── 00_pytorch_fundamentals.ipynb            # Walkthrough & personal experiments
│   └── 00_pytorch_fundamentals_exercises.ipynb  # Solutions to the 10 chapter exercises
│
├── mini_tasks/
│   ├── README.md                                # Master matrix & level hierarchy
│   └── 01_raisin_binary_classification/         # Level 1 Capstone: End-to-end Tabular Classification
│
└── .gitignore
```

---

## Applied Mini-Tasks Index

Independent projects applied to real-world datasets across the learning roadmap:

- **[Mini-Tasks Directory](mini_tasks/)**: Master index and difficulty progression roadmap.
- **[Task 01: Raisin Binary Classification](mini_tasks/01_raisin_binary_classification/)**: Complete tabular binary classification pipeline using custom `Dataset`, `DataLoader`, `nn.BCEWithLogitsLoss`, model checkpointing, and evaluation metrics (Accuracy: $86.67\%$, F1: $86.21\%$).
