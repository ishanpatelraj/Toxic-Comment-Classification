# Toxic Comment Classification Model

This repository contains a **Deep Learning NLP model** designed to classify toxic comments from social media platforms into different categories. The project aims to assist social media teams in identifying harmful content effectively and automating moderation processes.

## Features
- **Multi-class Classification**: Categories include toxic, obscene, threat, identity hate, and more.
- **Deep Learning Architecture**: Built with TensorFlow, Keras, and transformers for accurate text classification.
- **High Accuracy**: Achieved a classification accuracy of **91%** across various toxic comment categories.
- **End-to-End Pipeline**: Includes data preprocessing, training, evaluation, and deployment-ready model.

## Technologies Used
- **Python**
- **TensorFlow** and **Keras**
- **Transformers**
- **Scikit-learn**
- **Pandas** and **Numpy**
- **Natural Language Processing (NLP)** techniques

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/YourUsername/ToxicCommentClassification.git
   ```
2. Navigate to the project directory:
   ```bash
   cd ToxicCommentClassification
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset
The model was trained and tested on a public toxic comment dataset. Ensure you have the dataset in the required format before running the scripts.

**Expected format**:
- Columns: `id`, `comment_text`, `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`
- Labels: Binary (0/1) for each category.

## Usage
1. Preprocess the data:
   ```bash
   python preprocess.py
   ```
2. Train the model:
   ```bash
   python train.py
   ```
3. Evaluate the model:
   ```bash
   python evaluate.py
   ```
4. Make predictions:
   ```bash
   python predict.py --input "This is a sample comment"
   ```

## Model Architecture
The model uses:
- **Embedding Layer**: For word representation.
- **Bidirectional LSTM**: For capturing contextual information.
- **Dense Layers with Softmax Activation**: For multi-class output.

## Results
- **Accuracy**: 91%
- **F1 Score**: 0.89 (macro average)
- **Precision/Recall**: Optimized for minimizing false negatives in toxic categories.
