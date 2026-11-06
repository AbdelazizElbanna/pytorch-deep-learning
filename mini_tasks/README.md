# Applied Mini-Tasks

A curated collection of end-to-end, independent deep learning mini-tasks applied to real-world datasets. These tasks bridge theoretical concepts and exercise templates from Daniel Bourke's [PyTorch Deep Learning](https://www.learnpytorch.io/) curriculum into production-grade pipelines.

---

## Master Task Matrix

| ID | Task Name | Domain / Modality | Prerequisite Chapter | Level | Key PyTorch Mechanics & Modules | Dataset Source |
| :---: | :--- | :---: | :---: | :---: | :--- | :---: |
| **01** | [Raisin Binary Classification](01_raisin_binary_classification/) | Tabular / Vision Features | Chapter 02 (Classification) + Chapter 01 (Workflow) | **Level 1** | Custom `Dataset`, `DataLoader`, `nn.Module`, `nn.BCEWithLogitsLoss`, `train_step`, `test_step`, Model Checkpointing, Metrics & Confusion Matrix | [UCI Raisin Dataset](https://archive.ics.uci.edu/dataset/850/raisin) |
| *02* | *Planned: Regression Workflow* | *Tabular / Numerical* | *Chapter 01 (Workflow)* | *Level 1* | *`nn.L1Loss` / `MSELoss`, Linear Regression, Learning Rate Schedulers* | *TBD* |
| *03* | *Planned: Vision CNN Benchmark* | *Computer Vision* | *Chapter 03 (Computer Vision)* | *Level 2* | *`nn.Conv2d`, `nn.MaxPool2d`, Spatial Flattening, Multi-class Classification* | *TBD* |
| *04* | *Planned: Custom Image Pipeline* | *Computer Vision* | *Chapter 04 (Custom Datasets)* | *Level 3* | *`ImageFolder`, Data Augmentation (`transforms.v2`), Custom Data Loaders* | *TBD* |
| *05* | *Planned: Modularized Training Pipeline* | *Engineering / MLOps* | *Chapter 05 (Going Modular)* | *Level 3* | *Python script modularization (`data_setup.py`, `model_builder.py`, `train.py`)* | *TBD* |
| *06* | *Planned: Transfer Learning Feature Extractor* | *Computer Vision* | *Chapter 06 (Transfer Learning)* | *Level 4* | *Pretrained weights, Frozen feature extractors, Classifier head replacement* | *TBD* |

---

## Progression & Level Hierarchy

Each mini-task is categorized into a clear tier of difficulty, ensuring a structured progression for revision and study:

### Level 1: Foundations & Tabular Workflows (Chapters 01 - 02)
- **Focus**: Mastering the end-to-end PyTorch lifecycle on structured tabular and numerical data.
- **Key Concepts**:
  - Converting Pandas DataFrames and NumPy arrays into PyTorch Tensors.
  - Subclassing `torch.utils.data.Dataset` and configuring `DataLoader` batching.
  - Designing Multilayer Perceptrons (MLPs) using `nn.Sequential` or subclassed `nn.Module`.
  - Loss formulation: Binary Cross-Entropy with Logits (`nn.BCEWithLogitsLoss`) vs Softmax Multi-class Cross-Entropy (`nn.CrossEntropyLoss`).
  - Standard training and testing loops (`train_step`, `test_step`).
  - Model checkpointing: Saving and restoring `model.state_dict()` based on validation metrics.
  - Evaluation diagnostics: Precision, Recall, F1-Score, and Seaborn Confusion Matrix.

### Level 2: Computer Vision & Convolutional Neural Networks (Chapter 03)
- **Focus**: Spatial pattern recognition and 2D convolution mechanics.
- **Key Concepts**:
  - Multi-dimensional tensor shapes for vision: $[B, C, H, W]$.
  - Convolutional layers (`nn.Conv2d`), pooling layers (`nn.MaxPool2d`), and spatial dimension tracking.
  - Multi-class classification on image grids.

### Level 3: Custom Datasets, Augmentation & Modularity (Chapters 04 - 05)
- **Focus**: Handling uncurated raw image folders and refactoring exploratory notebooks into reusable Python modules.
- **Key Concepts**:
  - Custom dataset loading from disk directories (`torchvision.datasets.ImageFolder`).
  - Data preprocessing and augmentation pipelines (`torchvision.transforms.v2`).
  - Transitioning from Jupyter notebooks to modular Python codebases (`data_setup.py`, `engine.py`, `model_builder.py`, `utils.py`).

### Level 4: Transfer Learning & Experiment Tracking (Chapters 06 - 07)
- **Focus**: Leveraging state-of-the-art pretrained architectures and tracking training runs.
- **Key Concepts**:
  - Loading pretrained backbones from `torchvision.models` (e.g., EfficientNet, ResNet).
  - Freezing feature extraction layers and training custom classifier heads.
  - Tracking loss curves and hyperparameter comparisons via TensorBoard.

### Level 5: Architecture Replication & Production Deployment (Chapters 08 - 09)
- **Focus**: Replicating research papers from scratch (e.g., Vision Transformers - ViT) and deploying models to interactive interfaces.
- **Key Concepts**:
  - Multi-head self-attention and transformer encoder blocks.
  - Model export (TorchScript, ONNX).
  - Building interactive deployment apps using Gradio or FastAPI.

---

## Standard Mini-Task Structure

All mini-tasks follow a clean, reproducible folder hierarchy:

```text
mini_tasks/
├── [task_id]_[descriptive_name]/
│   ├── data/
│   │   └── raw/              # Raw data files (.csv, .xlsx, .txt)
│   ├── [pipeline_notebook].ipynb  # Self-contained exploratory & modeling notebook
│   └── best_model.pth        # Saved model checkpoint (gitignored)
```

---

## Spotlight: Task 01 — Raisin Binary Classification

- **Directory**: [`01_raisin_binary_classification/`](01_raisin_binary_classification/)
- **Dataset**: 900 grain instances of *Kecimen* and *Besni* raisin varieties with 7 geometric/morphological features extracted via computer vision (Area, Perimeter, MajorAxisLength, MinorAxisLength, Eccentricity, ConvexArea, Extent).
- **Architecture**:
  - Custom `RaisinDataset` class wrapping standardized PyTorch tensors.
  - `RaisinClassifier`: Linear(7 $\to$ 16) $\to$ ReLU $\to$ Linear(16 $\to$ 8) $\to$ ReLU $\to$ Linear(8 $\to$ 1).
  - Optimizer: `Adam(lr=0.001, weight_decay=1e-4)`.
  - Loss: `nn.BCEWithLogitsLoss()`.
- **Validation Performance**:
  - **Accuracy**: $86.67\%$
  - **Precision**: $85.23\%$
  - **Recall**: $87.21\%$
  - **F1-Score**: $86.21\%$
  - **Test Loss**: $0.3234$
