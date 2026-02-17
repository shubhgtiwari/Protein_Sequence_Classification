# Protein Family Classification Using Data Mining Techniques

## Overview

A comprehensive multi-class classification pipeline that classifies protein amino acid sequences into three protein families (Kinase, Transferase, GPCRs) using a combination of traditional machine learning, deep learning, unsupervised analysis, and advanced feature selection techniques. The project processes 22,815 protein sequences from UniProtKB/Swiss-Prot, engineers 1,427 biologically meaningful features per sequence using BioPython, and benchmarks four classifiers including a Bidirectional LSTM.

---

## Dataset

**Source:** UniProtKB/Swiss-Prot curated protein database (FASTA format)

| Protein Family | Raw Sequences | After Downsampling |
|----------------|--------------|-------------------|
| Kinase | 15,647 | 3,162 |
| Transferase | 4,006 | 4,006 |
| GPCRs | 3,162 | 3,162 |
| **Total** | **22,815** | **10,330** |

**Downsampling Strategy:** Kinase sequences (majority class) are randomly downsampled to match the smallest class (GPCRs = 3,162) before any feature extraction, preventing class-level bias in feature computation.

**Data Validation:**
- Sequences validated against the 20 standard amino acids (ACDEFGHIKLMNPQRSTVWY)
- Non-standard residues cleaned via character filtering
- Minimum sequence length filter (10 or more residues)
- FASTA parsing via BioPython's `SeqIO`

---

## Feature Engineering (1,427 Features)

Each protein sequence is transformed into a 1,427-dimensional feature vector using multiple biologically motivated feature extraction strategies:

| Feature Category | Count | Description | Biological Rationale |
|-----------------|-------|-------------|---------------------|
| Sequence Length | 1 | Number of amino acid residues | Fundamental property affecting protein structure and function |
| Amino Acid Composition | 20 | Relative frequency of each standard amino acid | Captures global residue distribution |
| Biophysical Properties | 4 | Molecular Weight (MW), Aromaticity, Instability Index, GRAVY (hydrophobicity) | Calculated via BioPython `ProteinAnalysis` -- critical physicochemical properties |
| Dipeptide Composition | 400 | Relative frequency of all 20x20 amino acid pairs | Captures local sequence patterns |
| Top k-mer (Tripeptide) | ~1,000 | Frequency of the 1,000 most common 3-mers across all sequences | Functionally important motifs at the tripeptide level |
| Shannon Entropy | 1 | Information-theoretic measure of sequence variability | Reflects conservation and complexity of the sequence |

**Total features per sequence:** 1,427 (verified from notebook: `Final dataset shape: (10330, 1427)`)

### Feature Extraction Pipeline

```
FASTA Files --> SeqIO.parse --> Validate/Clean --> extract_features()
                                                       |
                    +----------------------------------+----------------------------------+
                    |              |                    |                    |              |
              Length (1)     AA Comp (20)         Biophysical (4)     Dipeptide (400)  Top 3-mers (~1000)
                    |              |                    |                    |              |
                    +----------------------------------+----------------------------------+
                                                       |
                                            Shannon Entropy (1)
                                                       |
                                              Feature Vector (1,427)
```

---

## Data Preprocessing

### Class Balancing

After feature extraction, SMOTE (Synthetic Minority Oversampling Technique) is applied to the already-downsampled dataset:

| Stage | Shape |
|-------|-------|
| After feature extraction | 10,330 x 1,427 |
| After dropping label column | 10,330 x 1,426 |
| After SMOTE resampling | **12,018 x 1,426** |

### Standardization and Splitting

- **StandardScaler** applied to all features before SMOTE
- **LabelEncoder** converts class names to integers (0, 1, 2)
- **80/20 stratified train-test split** preserves class proportions
- **Label binarization** for multi-class ROC-AUC computation

---

## Unsupervised Analysis

Before classification, unsupervised techniques are applied for data understanding:

| Technique | Purpose |
|-----------|---------|
| **PCA** (Principal Component Analysis) | Dimensionality reduction and variance analysis |
| **t-SNE** | Non-linear embedding for cluster structure visualization |
| **K-Means Clustering** | Validate whether natural clusters align with protein families |
| **DBSCAN** | Density-based clustering to detect arbitrary-shaped clusters and outliers |

---

## Feature Selection

Two feature selection methods are applied:

| Method | Description |
|--------|-------------|
| **RFE** (Recursive Feature Elimination) | Iteratively removes least important features using Logistic Regression as estimator |
| **Mutual Information** (`mutual_info_classif`) | Non-parametric information-theoretic scoring of each feature's relevance to the target |

---

## Classification Models

### Traditional ML Models (with Hyperparameter Tuning)

All three models are tuned via `RandomizedSearchCV` with `StratifiedKFold(n_splits=3)`:

| Model | Hyperparameter Search Space | Best CV Accuracy |
|-------|---------------------------|-----------------|
| **Random Forest** | n_estimators: [100,200,300,500], max_depth: [None,10,20,30], min_samples_split: [2,5,10], class_weight: [balanced, balanced_subsample, None] | **0.9750** |
| **XGBoost** | n_estimators: [100,200,300], learning_rate: [0.01,0.1,0.2], max_depth: [3,6,9], subsample: [0.8,0.9,1.0], colsample_bytree: [0.8,0.9,1.0] | Pending notebook re-run |
| **SVM (RBF)** | C: [0.1,1,10,100], gamma: [scale,auto,0.1,0.01], kernel: rbf | Pending notebook re-run |

**Random Forest Best Parameters:** n_estimators=200, min_samples_split=2, max_depth=None, class_weight=None

### Deep Learning Model

| Model | Architecture | Details |
|-------|-------------|---------|
| **Bidirectional LSTM** | Embedding + Bidirectional LSTM + Dense | Protein sequences tokenized via Keras `Tokenizer`, padded with `pad_sequences`, processed through embedding layer and Bidirectional LSTM layers with Dense output |

### Evaluation

All models are evaluated using:
- Classification Report (Per-class Precision, Recall, F1-Score)
- Confusion Matrix with `ConfusionMatrixDisplay`
- ROC Curves (per-class and macro average via `label_binarize`)
- ROC-AUC Score

---

## Tech Stack

| Category | Technologies |
|----------|-------------|
| **Machine Learning** | Scikit-learn (Random Forest, SVM, Logistic Regression, PCA, K-Means, DBSCAN, t-SNE, RFE, StandardScaler, StratifiedKFold, RandomizedSearchCV) |
| **Deep Learning** | TensorFlow/Keras (Sequential, Bidirectional LSTM, Embedding, Dense, Masking, Tokenizer, pad_sequences) |
| **Gradient Boosting** | XGBoost |
| **Imbalanced Learning** | imbalanced-learn (SMOTE) |
| **Bioinformatics** | BioPython (SeqIO for FASTA parsing, ProteinAnalysis for biophysical features) |
| **Data** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Environment** | Google Colab |

---

## Skills Demonstrated

| Role Perspective | Skills Showcased |
|-----------------|-----------------|
| **Data Scientist** | Multi-class classification, hyperparameter tuning (RandomizedSearchCV), cross-validation (StratifiedKFold), evaluation metrics (ROC-AUC, F1, Precision, Recall), class imbalance handling (SMOTE + downsampling), feature importance |
| **ML Engineer** | End-to-end pipeline (data loading, feature extraction, preprocessing, model training, evaluation), model comparison/benchmarking, deep learning (Bidirectional LSTM), gradient boosting (XGBoost) |
| **Data Engineer** | FASTA file parsing with BioPython, data validation and cleaning pipeline, feature vector construction from raw biological sequences, scalable ETL-style data processing |
| **Data Analyst** | Unsupervised exploratory analysis (PCA, t-SNE, K-Means, DBSCAN), class distribution visualization, confusion matrix analysis, ROC curve interpretation |

---

## Project Structure

```
Protein_Sequence_Classification/
    Protein_Classification.ipynb    # Complete analysis notebook (2,894 lines, 1.1 MB)
    README.md
    LICENSE
```

## Getting Started

```bash
# Clone the repository
git clone https://github.com/ShubhGTiwari/Protein_Sequence_Classification.git
cd Protein_Sequence_Classification

# Install dependencies
pip install tensorflow scikit-learn xgboost imbalanced-learn biopython pandas numpy matplotlib seaborn

# Run the notebook (designed for Google Colab with mounted Drive for FASTA files)
jupyter notebook Protein_Classification.ipynb
```

**Note:** The FASTA data files (swissprot_kinase.fasta, swissprot_transferase.fasta, swissprot_gpcrs.fasta) are loaded from Google Drive paths in the notebook. Update the `DATASETS` dictionary at the top of the notebook to point to your local file paths.

## License

Apache 2.0
