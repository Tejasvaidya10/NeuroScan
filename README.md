# NeuroScan

A deep learning application that classifies brain tumors from MRI scans using transfer learning with ResNet50. It provides a web interface built with Streamlit for uploading scans and receiving instant predictions.

## Features

- **Transfer Learning**: Uses ResNet50 pretrained on ImageNet for high accuracy with limited training data
- **4-Class Classification**: Glioma, Meningioma, Pituitary Tumor, and No Tumor
- **Web Interface**: Streamlit app with real-time predictions and confidence scores
- **Class Imbalance Handling**: Automatic class weight balancing during training
- **Proper Evaluation**: Classification report, confusion matrix, and per-class metrics on a held-out test set
- **Privacy Focused**: Runs entirely locally — no data leaves your machine

## Project Structure

```
Brain-Tumor-Classification/
├── app/
│   └── main.py              # Streamlit web application
├── src/
│   ├── model.py              # ResNet50 model architecture
│   ├── train.py              # Training pipeline with evaluation
│   └── utils.py              # Image preprocessing and model loading
├── models/                    # Saved models and evaluation reports
├── dataset/
│   ├── Training/              # Training images (split 80/20 for train/val)
│   │   ├── glioma_tumor/
│   │   ├── meningioma_tumor/
│   │   ├── no_tumor/
│   │   └── pituitary_tumor/
│   └── Testing/               # Held-out test set (used only for final evaluation)
│       ├── glioma_tumor/
│       ├── meningioma_tumor/
│       ├── no_tumor/
│       └── pituitary_tumor/
├── requirements.txt
└── README.md
```

## Setup

### Prerequisites

- Python 3.8+

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Tejasvaidya10/NeuroScan.git
   cd NeuroScan
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # macOS/Linux
   venv\Scripts\activate     # Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download the [Brain Tumor Classification (MRI)](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri) dataset and place it in the `dataset/` directory with `Training/` and `Testing/` subdirectories.

## Usage

### Training

```bash
python src/train.py --dataset dataset --epochs 50 --batch_size 32
```

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset` | `dataset` | Path to dataset directory |
| `--epochs` | `50` | Number of training epochs |
| `--batch_size` | `32` | Batch size |

After training completes, the following files are saved to `models/`:
- `brain_tumor_model.h5` — trained model weights
- `class_labels.json` — class label mapping
- `training_history.png` — accuracy/loss plots
- `evaluation_report.txt` — classification report and confusion matrix

### Web Application

```bash
streamlit run app/main.py
```

Opens at `http://localhost:8501`. Upload an MRI scan and click "Analyze Scan" to get predictions.

## Model Architecture

- **Backbone**: ResNet50 (pretrained on ImageNet, two-phase fine-tuning)
- **Head**: GlobalAveragePooling2D → BatchNormalization → Dense(256, ReLU) → Dropout(0.5) → Dense(4, Softmax)
- **Input Size**: 224 x 224 x 3
- **Optimizer**: Adam
- **Loss**: Categorical Crossentropy

## Dataset

The [Brain Tumor Classification (MRI)](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri) dataset contains 3,264 MRI images across 4 classes:

| Class | Training | Testing |
|-------|----------|---------|
| Glioma | 826 | 100 |
| Meningioma | 822 | 115 |
| No Tumor | 395 | 105 |
| Pituitary | 827 | 74 |

The training set is further split 80/20 into train and validation subsets. The testing set is held out entirely and used only for final evaluation.

## License

MIT
