## Machine Learning Project: Music Lyrics Sentiment Analysis

A learning project exploring sentiment/emotion classification on song lyrics using BERT, with experiment tracking via Weights & Biases (W&B).

## Overview

This project fine-tunes a pre-trained BERT model (bert-base-uncased) to classify song lyrics into one of six emotions: sadness, joy, love, anger, fear, surprise. It covers the full pipeline from raw text cleaning to model training, using Hugging Face's transformers and datasets libraries.

The dataset used is the Kaggle Spotify 500K Songs with Lyrics and Audio Features dataset, sampled down for experimentation.

The model is trained on the ARC High-Performance Computing Cluster at University of Calgary

## What This Project Covers


Data loading & sampling: Reading the lyrics dataset and mapping text emotion labels to integers
Class distribution analysis: Visualizing how balanced (or imbalanced) the emotion classes are
Text preprocessing:

Expanding contractions (e.g. "I'd" → "I would") using contractions
Removing song structure markers like [Verse 1], [Chorus]
Stripping punctuation
Removing the most frequent and rarest words from the corpus



Train/test splitting with scikit-learn
Tokenization using the BERT tokenizer, converted into a Hugging Face Dataset/DatasetDict
Model setup: Loading bert-base-uncased with a classification head swapped in for 6 emotion labels (AutoModelForSequenceClassification)
GPU device handling
Training & experiment tracking: Using Hugging Face's Trainer with TrainingArguments, logged live to W&B (learning rate, batch size, optimizer, scheduler, etc.)


## Tech Stack


Python, Pandas, NumPy
Hugging Face transformers and datasets
PyTorch
scikit-learn (train/test split)
contractions (text preprocessing)
Weights & Biases (wandb) for experiment tracking
Matplotlib (visualization)


## Notes / Learnings


This is part of my ongoing learning in NLP and applied machine learning, particularly around fine-tuning transformer models for text classification.
Class imbalance in the emotion labels was identified early on. Please see the linked articles in the notebook for approaches considered (e.g. BERT-specific class imbalance handling, imbalanced-learn).
A separate version of this project exists with stop words removed during preprocessing, to compare the effect on model performance.

Additionally, no features engineering has been developed since the goal is to learn how to preprocess text data to specifically train for NLP tasks...


## Status

Work in progress (model training and evaluation results to be added as the project develops.)

## How to Run


Install dependencies: transformers, datasets, torch, wandb, contractions, scikit-learn, pandas, numpy, matplotlib, evaluate
Set up a W&B account and run wandb login to enable experiment tracking
Update the dataset path to point to your local copy of the Spotify lyrics dataset
Run the notebook cells in order
