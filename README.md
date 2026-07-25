# Persian Text Embedding Comparison

<p align="center">
  <strong>Comparative Analysis of TF-IDF, Word2Vec, and Gemma-2B for Persian Text Similarity</strong>
</p>

<p align="center">
  <a href="#overview">Overview</a> •
  <a href="#methodology">Methodology</a> •
  <a href="#results">Results</a> •
  <a href="#installation">Installation</a> •
  <a href="#usage">Usage</a> •
  <a href="#limitations">Limitations</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/NumPy-Scientific%20Computing-013243?logo=numpy&logoColor=white" alt="NumPy">
  <img src="https://img.shields.io/badge/Scikit--learn-Machine%20Learning-F7931E?logo=scikit-learn&logoColor=white" alt="Scikit-learn">
  <img src="https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Hugging%20Face-Transformers-FFD21E?logo=huggingface&logoColor=black" alt="Hugging Face">
</p>

---

## Overview

This project investigates how different text representation techniques encode Persian text into numerical vector spaces and how these representations affect semantic similarity analysis.

Three approaches are compared:

* **TF-IDF** — a classical statistical representation
* **Word2Vec** — a distributional word embedding approach
* **Gemma-2B** — a transformer-based contextual representation

The resulting vectors are compared using **Cosine Similarity**. The project also combines the generated representations and applies **Truncated Singular Value Decomposition (SVD)** for dimensionality reduction.

The project was developed in Python as a **Linear Algebra project**, with a focus on applying vector spaces, matrix representations, similarity measures, and dimensionality reduction to a Natural Language Processing problem.

---

## Research Question
 How do classical, distributional, and transformer-based text representations differ when Persian texts are represented as vectors and compared using cosine similarity?

---

## Objectives

This project aims to:

* Represent Persian text using multiple embedding techniques.
* Compare statistical, distributional, and transformer-based representations.
* Measure pairwise similarity between text representations.
* Apply dimensionality reduction using Truncated SVD.
* Demonstrate practical applications of linear algebra in Natural Language Processing.
* Provide a reproducible experimental workflow for comparing text representations.

---

## Methodology

### 1. Persian Text Preprocessing

The input texts are processed using Persian NLP tools from `Hazm`.

The preprocessing pipeline includes:

1. Tokenization
2. Stopword removal
3. Filtering non-alphabetic tokens

Each text is transformed into a sequence of tokens before being passed to the embedding methods.

---

### 2. TF-IDF

TF-IDF represents each text using the importance of its words within the document collection.

The implementation uses a maximum vocabulary size of 100 features.

TF-IDF produces a sparse vector representation in which each dimension corresponds to a vocabulary term.

---

### 3. Word2Vec

Word2Vec learns dense word representations based on word co-occurrence patterns.

The project trains a Word2Vec model using the Persian text dataset with:

| Parameter          | Value |
| ------------------ | ----: |
| Vector size        |   100 |
| Context window     |     5 |
| Minimum word count |     1 |
| Workers            |     4 |

A text-level representation is obtained by averaging the Word2Vec vectors of the words contained in that text.

---

### 4. Gemma-2B

The project uses the pretrained `google/gemma-2b` model to obtain transformer-based representations.

The processing pipeline is:

```text
Persian Text
     │
     ▼
Tokenization
     │
     ▼
Transformer Model
     │
     ▼
Hidden States
     │
     ▼
Mean Pooling
     │
     ▼
Fixed-Length Vector Representation
```

This allows the project to compare a modern transformer-based representation with classical and distributional approaches.

---

### 5. Cosine Similarity

The similarity between two text vectors is measured using cosine similarity.

For two vectors `A` and `B`:

```text
cosine_similarity(A, B) = (A · B) / (||A|| × ||B||)
```

Cosine similarity measures the angle between two vectors in the representation space.

A value closer to `1` generally indicates greater directional similarity, while a value closer to `0` indicates lower similarity.

---

### 6. Singular Value Decomposition

The generated representations are combined into a larger feature matrix and reduced using Truncated SVD.

SVD decomposes a matrix into three matrices:

```text
A = UΣVᵀ
```

In this project, SVD is used to:

* Reduce the dimensionality of the combined feature matrix.
* Produce a lower-dimensional representation.
* Analyze the variance captured by the retained components.

The reduced representation is saved as:

```text
output/reduced_matrix_svd.csv
output/reduced_matrix_svd.npy
```

---

## Experimental Pipeline

```text
┌──────────────────────┐
│ Persian Text Dataset │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Text Preprocessing   │
│ Tokenization         │
│ Stopword Removal     │
└──────────┬───────────┘
           │
     ┌─────┼─────┐
     ▼     ▼     ▼
  TF-IDF Word2Vec Gemma
     │     │     │
     └─────┼─────┘
           │
           ▼
┌───────────────────────┐
│ Vector Representations│
└──────────┬────────────┘
           │
           ▼
┌───────────────────────┐
│   Cosine Similarity   │
└──────────┬────────────┘
           │
           ▼
┌───────────────────────┐
│    Combined Matrix    │
└──────────┬────────────┘
           │
           ▼
┌───────────────────────┐
│     Truncated SVD     │
└──────────┬────────────┘
           │
           ▼
┌───────────────────────┐
│ Reduced Representation│
└──────────┬────────────┘
```

---

## Dataset

The dataset is stored in:

```text
data/persian_texts.txt
```

Each non-empty line represents one Persian text sample.

The dataset is designed for experimental comparison of text representations rather than for training a large-scale language model.

---

## Results

The `generate_results.py` script is used to generate additional analysis outputs from the same experimental workflow.

The generated comparison table contains:

* Average Similarity
* Standard Deviation
* Similarity Range

The generated visualizations include:

```text
output/figures/
├── average_similarity.png
├── standard_deviation.png
├── similarity_range.png
└── svd_explained_variance.png
```

The comparison table is saved as:

```text
output/comparison_results.csv
```

The SVD representations are saved as:

```text
output/reduced_matrix_svd.csv
output/reduced_matrix_svd.npy
```

---

## Project Structure

```text
Persian-Text-Embedding-Comparison-TFIDF-Word2Vec-Gemma/
│
├── data/
│   └── persian_texts.txt
│
├── Main.py
│
├── generate_results.py
│
└── README.md
```

The main implementation is contained in `Main.py`.

The separate `generate_results.py` script is used to generate additional comparison outputs and visualizations without modifying the original main implementation.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/ArianShayestehfard/Persian-Text-Embedding-Comparison-TFIDF-Word2Vec-Gemma.git
```

Navigate to the project directory:

```bash
cd Persian-Text-Embedding-Comparison-TFIDF-Word2Vec-Gemma
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the virtual environment on Windows:

```powershell
.venv\Scripts\activate
```

Install the required dependencies:

```bash
pip install numpy pandas scikit-learn gensim hazm transformers torch matplotlib
```

---

## Usage

### Run the Main Implementation

```bash
python Main.py
```

### Generate Additional Results and Visualizations

```bash
python generate_results.py
```

The generated files will be stored in the `output/` directory.

---

## Reproducibility

The experiment can be reproduced by:

1. Cloning the repository.
2. Installing the required Python dependencies.
3. Using the provided Persian text dataset.
4. Running `Main.py`.
5. Running `generate_results.py` to generate additional analysis outputs.

Because transformer models and numerical libraries may behave differently across hardware and software environments, small variations in results may occur.

---

## Limitations

This project is an experimental comparative study and should not be interpreted as a universal benchmark.

The main limitations include:

* The dataset is relatively small.
* Word2Vec is trained on the project dataset rather than a large Persian corpus.
* The transformer model is a general-purpose pretrained model.
* The original experimental implementation evaluates similarity on a limited number of text samples.
* Different preprocessing strategies may affect the performance of all three methods.
* Results may vary depending on the installed versions of Python libraries and available hardware.

---

## Technologies

* Python
* NumPy
* Pandas
* Scikit-learn
* Gensim
* Hazm
* PyTorch
* Hugging Face Transformers
* Matplotlib

---

## Academic Context

This project demonstrates how fundamental concepts from Linear Algebra can be applied to a modern Natural Language Processing problem.

The main mathematical concepts used include:

* Vector spaces
* Matrix representations
* Dot products
* Vector norms
* Cosine similarity
* Matrix concatenation
* Singular Value Decomposition
* Dimensionality reduction

The project connects these mathematical concepts with practical text representation methods used in modern NLP systems.

---

## References

* Gemma Team — Gemma: Open Models Based on Gemini Research and Technology.
* Hazm — Persian Natural Language Processing Toolkit.
* Scikit-learn — Machine Learning and Scientific Computing Tools for Python.
* Gensim — Topic Modeling and Vector Space Modeling Toolkit.

---

## Author

**Arian Shayestehfard**

Computer Engineering Student

GitHub: [@ArianShayestehfard](https://github.com/ArianShayestehfard)

---

## License

This project is intended for educational and research purposes.
