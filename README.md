# Fine-Tuning-GPT-2-for-Sentiment-Analysis
Fine-Tuning of GPT-2 for Sentiment Analysis using IMDB Dataset.csv

# Fine-Tuning GPT-2 for Sentiment Analysis

This repository contains the code and instructions to fine-tune a pre-trained GPT-2 model for sentiment analysis using the IMDB dataset. The goal of this project is to classify movie reviews into three categories: **positive**, **negative**, and **neutral**.

## Table of Contents
1. [Overview](#overview)
2. [Dataset](#dataset)
3. [Dependencies](#dependencies)
4. [Setup](#setup)
5. [Fine-Tuning Process](#fine-tuning-process)
6. [Training](#training)
7. [Evaluation](#evaluation)
8. [Results](#results)
9. [Acknowledgments](#acknowledgments)

---

## Overview

In this project, we fine-tune the GPT-2 language model for sentiment analysis. The model is trained on the IMDB dataset, which consists of movie reviews labeled as positive or negative. We extend the dataset by adding neutral reviews to make it more robust. The fine-tuned model can then classify new reviews into one of three sentiment categories: **positive**, **negative**, or **neutral**.

---

## Dataset

The dataset used in this project is the **IMDB Movie Reviews Dataset**, which contains 50,000 reviews labeled as either positive or negative. The dataset is split into:

- **Training set**: 80% of the data
- **Validation set**: 10% of the data
- **Test set**: 10% of the data

Additionally, we introduce a **neutral** sentiment class to make the model more versatile.

### Dataset Source:
- Original dataset: [IMDB Dataset](https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

---

## Dependencies

To run this project, you need the following dependencies:

- Python >= 3.7
- PyTorch >= 1.7
- Transformers (Hugging Face) >= 4.0
- Datasets (Hugging Face) >= 1.0
- Pandas
- Scikit-learn

---


## Fine-Tuning Process


1. Data Preprocessing
The dataset is preprocessed as follows:

The dataset is split into training (80%), validation (10%), and test (10%) sets.
Neutral reviews are added to the dataset to balance the sentiment classes.
Each review is tokenized using the GPT-2 tokenizer, with padding and truncation applied to ensure consistent input lengths.


2. Model Selection
We use the GPT-2 model from Hugging Face's transformers library. The model is fine-tuned for sequence classification with three output classes: positive , negative , and neutral .


3. Tokenization
The GPT-2 tokenizer is used to tokenize the text data. The tokenizer is configured to pad sequences to a maximum length of 512 tokens and truncate longer sequences.


4. Training Configuration
The model is fine-tuned using the following hyperparameters:

Learning rate : 2e-5
Batch size : 8 (per device)
Number of epochs : 3
Weight decay : 0.01
Evaluation strategy : After each epoch
Save strategy : After each epoch
You can install the required libraries using the following command:

5. Evaluation
After training, the model is evaluated on the test set. The evaluation metrics include:

Accuracy
Precision
Recall
F1-score


```bash
pip install transformers datasets torch pandas scikit-learn
git clone https://github.com/yourusername/fine-tune-gpt2-sentiment-analysis.git
cd fine-tune-gpt2-sentiment-analysis
cp /path/to/IMDB\ Dataset.csv data/
pip install -r requirements.txt
python train.py
python evaluate.py
