# LLMs4Subjects - Bilingual Subject Tagging Solution

Welcome to the **LLMs4Subjects** repository! This project presents a **bilingual subject tagging** solution for the Semeval 2025 task 5. The goal is to leverage large language models (LLMs) to recommend relevant GND (Gemeinsame Normdatei) subjects for **technical records** from Leibniz University’s TIBKAT collection.

## Table of Contents

- [Overview](./Overview.md)
- [Repository Structure](./Repository_Structure.md)
- [Approach Details](./Approach_Details.md)
  - [Soft Retrieval + Negative Sampling](#soft-retrieval--negative-sampling)
  - [Advanced MLP and Attention](#advanced-mlp-and-attention)
  - [Margin-Based Loss](#margin-based-loss)
  - [FAISS-Based Inference](#faiss-based-inference)
- [Acknowledgments](./Acknowledgments.md)

## Overview

This solution addresses the **LLMs4Subjects** shared task, aiming to:

1. **Train** powerful bilingual models (English and German) to embed titles and abstracts of technical records.
2. **Retrieve** top-50 relevant GND subjects for each record, leveraging both **semantic embeddings** and **FAISS** indexing for efficient similarity search.
3. **Support** bilingual processing (English and German), aligning to the GND taxonomy for subject tagging.

The first two files (`English_train.py` and `German_train.py`) **train** the respective models, while the remaining two files (`English_test.py` and `German_test.py`) **test** these models on new data and **generate** top-50 subject recommendations.

## Repository Structure

- **database_english_creation.py**: Creates the CSV data that is used for the vector database purpose for English Subjects.
- **database_german_creation.py**: Creates the CSV data that is used for the vector database purpose for German Subjects.
- **train_dev_dataset_creation.py**: Creates the train and development CSV files for English and German, preparing datasets for training and evaluation.
- **English_train.py**: Trains a subject retrieval model on **English** TIBKAT data.
- **German_train.py**: Trains a subject retrieval model on **German** TIBKAT data, enabling the system to handle German technical records effectively.
- **English_test.py**: Utilizes the trained English model to retrieve and rank relevant GND subjects for new English records.
- **German_test.py**: Utilizes the trained German model to retrieve and rank relevant GND subjects for new German records.

## Approach Details

### Soft Retrieval + Negative Sampling

The approach employs **soft retrieval** combined with **negative sampling** to train robust and discriminative embeddings. This method ensures that the model not only learns to associate technical records with their correct subjects but also distinguishes them from irrelevant subjects.

**Process:**

1. **Anchor**: Represents the embedding of a technical record’s **title + abstract**, transformed by the `FocusedEmbTransform` model.
2. **Positive**: The average embedding of the gold-standard GND subjects annotated for the record.
3. **Negatives**: A set of randomly sampled subject embeddings from the entire subject corpus.

**Mathematical Representation:** For each training sample *i*:

- Let **M_i** be the embedding of the title + abstract.
- Let **O_i** be the average embedding of the gold-standard GND subjects.
- Let **S_j** for *j* = 1 to *K* be *K* randomly sampled negative subject embeddings.

The goal is to minimize the distance between **M_i** and **O_i**, while maximizing the distance between **M_i** and each **S_j**.

### Advanced MLP and Attention

A custom neural network module, `FocusedEmbTransform`, is designed to enhance the embedding representations through non-linear transformations and attention mechanisms.

**Components:**

1. **Multi-layer MLP**: Introduces non-linearity and increases the capacity of the model to capture complex relationships.
   * **Architecture**:
     - The hidden layer **H** is computed using the ReLU activation function applied to the linear transformation of the input embedding **x**.
     - The output **y** is obtained by applying another linear transformation to **H**.
     - Where:
       - **x** is the input embedding.
       - **W1** and **W2** are weight matrices.
       - **b1** and **b2** are bias vectors.
       - **y** is the transformed embedding.
2. **Per-Dimension Attention**: A learnable attention vector that re-weights each dimension of the transformed embedding, allowing the model to focus on the most relevant features.
   * **Mathematical Representation**:
     - The transformed embedding **y** is element-wise multiplied by the attention vector **α** to produce **y′**.
     - Where:
       - **α** is the attention vector with learnable parameters.
       - The multiplication is performed element-wise.

**Implementation Details:**

- The `FocusedEmbTransform` class encapsulates the MLP and attention mechanism.
- The attention weights are initialized to ones and are learned during training.

### Margin-Based Loss

To effectively train the embeddings, a **margin-based triplet loss** is utilized. This loss function encourages the model to position the anchor closer to the positive and farther from the negatives within a specified margin.

**Formula:**

The loss **L_i** for the *i*-th sample is calculated by summing the maximum of zero and the margin plus the distance between the anchor **a_i** and positive **p_i** minus the distance between the anchor **a_i** and each negative **n_ij** for all *j* from 1 to *K*.

Where:

- **a_i** is the transformed anchor embedding.
- **p_i** is the positive embedding.
- **n_ij** is the *j*-th negative embedding.
- **dist(a, b)** represents the cosine distance between embeddings **a** and **b**.
- **Margin** is a hyperparameter (e.g., 0.2) that defines the minimum desired difference between positive and negative distances.

**Intuition:**

- **Positive Pair**: The model should minimize the distance between the anchor and positive embeddings.
- **Negative Pairs**: The model should maximize the distance between the anchor and each negative embedding, ensuring that irrelevant subjects are sufficiently distinct.

### FAISS-Based Inference

During the testing phase, **FAISS** (Facebook AI Similarity Search) is employed to perform efficient similarity searches over the large subject corpus. FAISS enables rapid retrieval of the top-K most similar subjects based on the embedded representations.

**Process:**

1. **Embedding Subjects**: Each subject in the corpus is embedded using the trained `FocusedEmbTransform` model.
2. **Building FAISS Index**: All subject embeddings are added to a FAISS index for efficient similarity search.
3. **Query Embedding**: For each new technical record, its title and abstract are embedded and transformed using the trained model.
4. **Similarity Search**: The FAISS index is queried to retrieve the top-50 subjects that are most similar to the query embedding.
5. **Result Ranking**: Retrieved subjects are ranked based on their similarity scores (cosine similarity) and saved as recommendations.

**Mathematical Representation:** Given a query embedding **q**, FAISS retrieves the top-K subjects **s_k** for *k* = 1 to *K* that minimize the distance **dist(q, s_k)**.

Where:

- **q** is the transformed embedding of the query (title + abstract).
- **s_k** are the subject embeddings in the FAISS index.
- **dist** is typically the cosine distance.

**Advantages:**

- **Scalability**: FAISS is optimized for handling large-scale datasets with millions of vectors.
- **Speed**: Enables real-time retrieval of similar subjects, making it suitable for operational workflows.
- **Flexibility**: Supports various distance metrics and indexing strategies to balance speed and accuracy.

## Acknowledgments

This solution is built for the **LLMs4Subjects** Semeval 2025 shared task, hosted by Leibniz University’s Technical Library (TIBKAT). We extend our thanks to the organizers for providing the dataset and guiding the community toward innovative solutions in **LLM-based subject tagging**.