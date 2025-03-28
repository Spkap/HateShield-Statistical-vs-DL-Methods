# Hate Speech Detection: Statistical vs. Deep Learning Approaches

## Overview

This project explores the performance of various statistical and deep learning-based embedding methods for hate speech detection. The analysis is conducted using two Jupyter notebooks: `NLP/Statistical models.ipynb` and `NLP/DEEP Learning Models.ipynb`. The focus is on understanding the strengths and weaknesses of each approach in terms of accuracy, precision, recall, and F1-score.

## Dataset

- **Source**: `cardiffnlp/tweet_eval` hate-speech dataset from Hugging Face
- **Composition**: Training, validation, and test sets of tweets labeled for hate speech
- **Labels**: Binary classification (0: non-hate speech, 1: hate speech)

## Statistical Models

### Preprocessing and Dataset Setup

- **Libraries Used**: NLTK, scikit-learn, Gensim, pandas, numpy, Matplotlib, seaborn
- **Resources**: NLTK stopwords, lemmatization, GloVe embeddings (Twitter GloVe 25-dimensional)
- **Dataset**: `cardiffnlp/tweet_eval` hate-speech dataset

### Text Preprocessing

- **Functionality**:
  - Handles invalid input (returns empty string for non-string/blank text)
  - Converts to lowercase for uniformity
  - Cleans Twitter elements (replaces URLs with "URL", mentions with "USER", removes hashtags)
  - Removes special characters, numbers, and non-alphabetic tokens
  - Removes stopwords while retaining important negations ("not", "no", "never")
  - Applies lemmatization to reduce words to base forms
- **Safe Preprocessing**: Ensures error handling and adds a `processed_text` column.

### Embedding Methods

1. **Bag of Words (BoW)**

   - Uses `CountVectorizer` with a max of 5000 features
   - Creates sparse document-term matrices for training and testing datasets

2. **TF-IDF**

   - Employs `TfidfVectorizer` with a max of 5000 features
   - Weights terms by frequency and inverse document frequency

3. **Word2Vec**

   - Trains on tokenized tweets with vector size 100
   - Parameters: window size 5, min word count 2, 4 workers
   - Document vectors created by averaging word vectors

4. **FastText**

   - Similar parameters as Word2Vec (vector size 100, window 5, min count 2)
   - Better handling of out-of-vocabulary words through subword information

5. **GloVe**
   - Utilizes pre-trained Twitter GloVe embeddings (25-dimensional)
   - Document vectors created by averaging word vectors

### Model Training and Evaluation

- **Model**: Logistic Regression (random_state=42)
- **Metrics**:
  - Classification report (precision, recall, F1-score)
  - Accuracy
  - Hate Speech Detection Rate (Recall for class 1)
  - False Positive Rate (1 - Recall for class 0)
  - ROC curves with AUC
  - Confusion matrices

### Results

| Model    | Accuracy | Hate Speech Detection Rate | False Positive Rate | F1 (Hate Speech) |
| -------- | -------- | -------------------------- | ------------------- | ---------------- |
| BoW      | 51.0%    | 90.1%                      | 77.5%               | 60.8%            |
| TF-IDF   | 50.7%    | 88.8%                      | 77.1%               | 60.3%            |
| Word2Vec | 47.6%    | 65.6%                      | 65.5%               | 51.3%            |
| FastText | 49.1%    | 67.3%                      | 64.1%               | 52.7%            |
| GloVe    | 57.8%    | 0.0%                       | 0.0%                | 0.0%             |
| BERT     | 76.3%    | 74.2%                      | 22.1%               | 75.8%            |

### Visual Analysis

- **Confusion Matrices**: Show true vs. predicted labels for each embedding method
- **ROC Curves**: Display False Positive Rate vs. True Positive Rate with AUC scores

## Deep Learning Models

### Setup

- **Libraries**: Transformers, PyTorch, torch.nn, sklearn.metrics
- **Device**: Automatically utilizes GPU (CUDA) if available, otherwise CPU
- **Text Preprocessing**: Uses a custom function to:
  - Convert text to lowercase
  - Remove URLs, mentions, hashtags, and non-alphabetic characters
  - Remove extra spaces

### Dataset Preparation

- **Dataset**: Same TweetEval dataset used for statistical models
- **Custom Dataset Class**: `TextDataset` for tokenized text data with:
  - Tokenization using `tokenizer.encode_plus`
  - Handling of padding, truncation, and attention masks
  - Batching with PyTorch DataLoader

### Model Architecture

- **Base Model**: `BERTClassifier` leveraging pre-trained BERT (`bert-base-uncased`)
- **Components**:
  - BERT base model for contextual representations
  - Dropout layer (rate: 0.3) for regularization
  - Linear classification head for binary classification
- **Forward Pass**:
  - Gets transformer outputs from input_ids and attention_mask
  - Uses [CLS] token representation (pooled output) for classification
  - Applies dropout and classification layer to produce logits

### Training Process

- **Optimizer**: AdamW with learning rate 2e-5
- **Loss Function**: CrossEntropyLoss
- **Training Loop**:
  - Batch processing with progress tracking via tqdm
  - Regular evaluation on validation set
  - Early stopping based on validation F1 score
  - Metrics tracking (accuracy, F1 score) for both training and validation

### Evaluation Metrics

- **Accuracy**: Overall correctness of predictions
- **Precision**: Correct positive predictions / total positive predictions
- **Recall**: Correctly identified positive samples / total actual positive samples
- **F1-Score**: Harmonic mean of precision and recall
- **Classification Report**: Detailed breakdown of metrics by class
- **Confusion Matrix**: Visual representation of true vs. predicted labels

### Results

- **Deep Learning Models**: Generally outperform statistical models in:
  - Higher precision for hate speech detection
  - Better recall with fewer false negatives
  - Improved F1-scores indicating better overall performance
- **Contextual Understanding**: Better capture of nuances and context in language
- **Visualization**:
  - Training/validation metrics over epochs
  - Confusion matrix heatmaps
  - Example predictions compared to ground truth

## Key Insights

1. **Statistical Models**

   - **Pros**:
     - Simplicity and interpretability
     - Quick deployment and training
     - Lower computational cost
     - BoW and TF-IDF show high recall but with false positive tradeoff
   - **Cons**:
     - Higher false positive rates (especially BoW and TF-IDF)
     - Less nuanced understanding of context
     - GloVe implementation showed poor performance for hate speech detection

2. **Deep Learning Models**

   - **Pros**:
     - Better contextual understanding through attention mechanisms
     - Improved precision and recall balance
     - More robust to language variations and nuances
     - Better understanding of semantics beyond keywords
   - **Cons**:
     - Higher computational requirements (GPU recommended)
     - Longer training times
     - Less interpretable "black box" decisions

3. **Use Case Considerations**
   - **Statistical Models**: Suitable for simpler tasks with limited resources or when quick deployment is needed
   - **Deep Learning Models**: Ideal for complex language tasks requiring nuanced understanding where accuracy is critical

## Recommendations

1. **Hybrid Approaches**: Combine statistical and deep learning methods for improved performance:

   - Use statistical models for initial filtering
   - Apply deep learning for refined decisions on edge cases

2. **Resource Allocation**:

   - Consider computational resources when choosing between approaches
   - For resource-constrained environments, optimized TF-IDF models may be preferable
   - For critical applications where accuracy is paramount, invest in deep learning infrastructure

3. **Domain Adaptation**:

   - Fine-tune models on domain-specific data for better results
   - Consider regular retraining as language and hate speech patterns evolve
   - Implement active learning for continuous improvement

4. **Evaluation Strategy**:
   - Focus on balanced metrics (F1-score) rather than just accuracy
   - Consider the specific costs of false positives vs. false negatives

## Dependencies

- Python 3.10+
- PyTorch
- Transformers
- scikit-learn
- NLTK
- Gensim
- pandas
- numpy
- matplotlib
- seaborn
- tqdm
- datasets

## References

### Documentation

- [Transformers Documentation](https://huggingface.co/docs/transformers/index)
- [scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Gensim Documentation](https://radimrehurek.com/gensim/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [NLTK Documentation](https://www.nltk.org/)

### Resources

1. **Text Preprocessing**

   - [Comprehensive Guide to Text Preprocessing](https://aman.ai/primers/ai/preprocessing/)

2. **Word Embeddings**

   - [Understanding Word Vectors and Embeddings](https://aman.ai/primers/ai/word-vectors/)

3. **BERT Architecture**

   - [Deep Dive into BERT](https://aman.ai/primers/ai/bert/)

4. **Sentence Transformers**

   - [From BERT to SBERT: Sentence Embeddings](https://www.pinecone.io/learn/series/nlp/sentence-embeddings/)

5. **Hate Speech Detection**
   - [TweetEval Benchmark](https://github.com/cardiffnlp/tweeteval)
   - [HateCheck: Functional Tests for Hate Speech Detection Models](https://aclanthology.org/2021.acl-long.4.pdf)
