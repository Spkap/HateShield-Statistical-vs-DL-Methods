# Hate Speech Detection: Statistical vs. Deep Learning Approaches

## Overview

This project explores the performance of various statistical and deep learning-based embedding methods for hate speech detection. The analysis is conducted using two Jupyter notebooks: `NLP/Statistical models.ipynb` and `NLP/DEEP Learning Models.ipynb`. The focus is on understanding the strengths and weaknesses of each approach in terms of accuracy, precision, recall, and F1-score.

## Statistical Models

### Preprocessing and Dataset Setup

- **Libraries Used**: NLTK, scikit-learn, Gensim, Matplotlib
- **Resources**: NLTK stopwords, lemmatization, GloVe embeddings
- **Dataset**: `cardiffnlp/tweet_eval` hate-speech dataset

### Text Preprocessing

- **Functionality**: Handles invalid input, converts to lowercase, cleans Twitter elements, removes noise, and applies lemmatization.
- **Safe Preprocessing**: Ensures error handling and adds a `processed_text` column.

### Embedding Methods

1. **Bag of Words (BoW)**

   - Uses `CountVectorizer` with a max of 5000 features.

2. **TF-IDF**

   - Employs `TfidfVectorizer` with a max of 5000 features.

3. **Word2Vec**

   - Trains on tokenized tweets with vector size 100.

4. **FastText**

   - Similar parameters as Word2Vec.

5. **GloVe**
   - Utilizes pre-trained Twitter GloVe embeddings.

### Model Training and Evaluation

- **Model**: Logistic Regression
- **Metrics**: Classification report, accuracy, precision, recall, F1-score

### Results

| Model    | Accuracy | Hate Speech Detection Rate | False Positive Rate | F1 (Hate Speech) |
| -------- | -------- | -------------------------- | ------------------- | ---------------- |
| BoW      | 51.0%    | 90.1%                      | 77.5%               | 60.8%            |
| TF-IDF   | 50.7%    | 88.8%                      | 77.1%               | 60.3%            |
| Word2Vec | 47.6%    | 65.6%                      | 65.5%               | 51.3%            |
| FastText | 49.1%    | 67.3%                      | 64.1%               | 52.7%            |
| GloVe    | 57.8%    | 0.0%                       | 0.0%                | 0.0%             |

## Deep Learning Models

### Setup

- **Libraries**: Transformers, PyTorch
- **Device**: Utilizes GPU if available

### Model Architecture

- **Transformer-based Models**: BERT, RoBERTa, etc.
- **Training**: Fine-tuning on the hate-speech dataset

### Evaluation Metrics

- **Accuracy**: Overall correctness of predictions
- **Precision**: Correct positive predictions
- **Recall**: True positive rate
- **F1-Score**: Balance between precision and recall

### Results

- **Deep Learning Models**: Generally outperform statistical models in terms of precision and recall.
- **Contextual Understanding**: Better capture of nuances and context in language.

## Key Insights

1. **Statistical Models**

   - **Pros**: Simplicity, quick deployment, lower computational cost.
   - **Cons**: Higher false positive rates, less nuanced understanding.

2. **Deep Learning Models**

   - **Pros**: Better contextual understanding, improved precision and recall.
   - **Cons**: Higher computational requirements, longer training times.

3. **Use Case Considerations**
   - **Statistical Models**: Suitable for simpler tasks with limited resources.
   - **Deep Learning Models**: Ideal for complex language tasks requiring nuanced understanding.

## Recommendations

1. **Hybrid Approaches**: Combine statistical and deep learning methods for improved performance.
2. **Resource Allocation**: Consider computational resources when choosing between approaches.
3. **Domain Adaptation**: Fine-tune models on domain-specific data for better results.

## Dependencies

- Python 3.10+
- PyTorch
- Transformers
- scikit-learn
- NLTK
- Gensim
- pandas
- numpy

## References

### Documentation

- [Transformers Documentation](https://huggingface.co/docs/transformers/index)
- [scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Gensim Documentation](https://radimrehurek.com/gensim/)

### Resources

1. **Text Preprocessing**

   - [Comprehensive Guide to Text Preprocessing](https://aman.ai/primers/ai/preprocessing/)

2. **Word Embeddings**

   - [Understanding Word Vectors and Embeddings](https://aman.ai/primers/ai/word-vectors/)

3. **BERT Architecture**

   - [Deep Dive into BERT](https://aman.ai/primers/ai/bert/)

4. **Sentence Transformers**
   - [From BERT to SBERT: Sentence Embeddings](https://www.pinecone.io/learn/series/nlp/sentence-embeddings/)
