# Named-Entity-Recognition-NER

This project implements Named Entity Recognition (NER) using deep learning models in TensorFlow/Keras. It demonstrates how to preprocess CoNLL-style datasets, build vocabularies, encode sequences, and train models for sequence labeling tasks such as identifying entities in text.

# 📌 Features
Data loading from CoNLL format (train.txt, valid.txt, test.txt).

Vocabulary and label mapping with <PAD> and <UNK> tokens.

Sequence encoding with fixed maximum length.

# Two models implemented:

## Basic Embedding + Dense Layer

## Bidirectional LSTM (BiLSTM)

# Training with callbacks:
  EarlyStopping
  ReduceLROnPlateau
  Evaluation using seqeval metrics (Precision, Recall, F1-score).
  Visualization of training performance with Matplotlib.
  Inference example on a custom sentence (["John", "lives", "in", "New", "York"]).

# 📂 Project Structure
Code
├── train.txt
├── valid.txt
├── test.txt
├── ner_model.py   # Main script with data loading, training, evaluation
├── README.md      # Project documentation

# ⚙️ Requirements
Install dependencies before running the project:

bash
pip install tensorflow numpy matplotlib seqeval

# 🚀 Usage
1. Prepare Data
Ensure your dataset files (train.txt, valid.txt, test.txt) are in CoNLL format:

Code
John    B-PER
lives   O
in      O
New     B-LOC
York    I-LOC

Mary    B-PER
works   O
at      O
Google  B-ORG

# 2. Train Models
Run the script to train both models:

bash
python ner_model.py
Model 1: Embedding + Dense

Model 2: BiLSTM + Dense

Training history plots will be displayed.

# 3. Evaluate Models
Evaluation metrics (Precision, Recall, F1-score) are printed using seqeval.

# Example output:

Code
Precision: 0.92
Recall: 0.90
F1-score: 0.91

Detailed Report:
              precision    recall  f1-score   support
LOC              0.89      0.87      0.88       120
ORG              0.91      0.92      0.91       150
PER              0.94      0.93      0.93       200

# 4. Inference
Test on a custom sentence:

python
sentence = ["John", "lives", "in", "New", "York"]
Output:

Code
John --> B-PER
lives --> O
in --> O
New --> B-LOC
York --> I-LOC

# 📊 Results
Model 1 (Embedding + Dense): Baseline performance.

Model 2 (BiLSTM): Improved contextual understanding and higher F1-score.

🔮 Future Improvements
Add CRF layer for better sequence modeling.

Use pre-trained embeddings (e.g., GloVe, FastText).

Experiment with transformer-based models (BERT, RoBERTa).
