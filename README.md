# Fake News Detection Using LSTM

A deep learning-based system for detecting and classifying fake news articles using Long Short-Term Memory (LSTM) neural networks.

## Overview

This project implements a binary classification model to distinguish between true and fake news articles. The model analyzes news article titles and text content to identify patterns associated with misinformation, achieving robust performance on a dataset of over 44,000 news articles.

## Features

- **Text Preprocessing Pipeline**: Comprehensive NLP preprocessing including stemming, stopword removal, and text normalization
- **Deep Learning Architecture**: Bidirectional LSTM network for sequential text analysis
- **Balanced Dataset**: Trained on 23,481 fake news and 21,417 true news articles
- **Real-time Classification**: Capable of classifying new articles based on learned patterns

## Dataset

The project uses two CSV files containing news articles:
- `True.csv`: Verified true news articles
- `Fake.csv`: Identified fake news articles

Each article includes:
- Title
- Text content
- Subject category
- Publication date
- Label (1 for true, 0 for fake)

## Technology Stack

- **Python 3.x**
- **TensorFlow/Keras**: Deep learning framework
- **NLTK**: Natural language processing
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Scikit-learn**: Machine learning utilities

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/fake-news-detection.git
cd fake-news-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install tensorflow pandas nltk numpy scikit-learn
```

## Usage

1. **Prepare the data**:
```python
import pandas as pd

# Load datasets
df1 = pd.read_csv("path/to/True.csv")
df2 = pd.read_csv("path/to/Fake.csv")

# Label the data
df1['labels'] = 1
df2['labels'] = 0

# Combine datasets
df = pd.concat([df1, df2], axis=0, ignore_index=True)
```

2. **Preprocess text**:
```python
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

# Text cleaning and stemming
ps = PorterStemmer()
# Apply preprocessing pipeline
```

3. **Train the model**:
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Bidirectional

# Build and train LSTM model
# (See notebook for complete implementation)
```

## Project Structure

```
fake-news-detection/
│
├── LSTM.ipynb              # Main Jupyter notebook
├── README.md               # Project documentation
├── requirements.txt        # Python dependencies
└── data/
    ├── True.csv           # True news dataset
    └── Fake.csv           # Fake news dataset
```

## Preprocessing Steps

1. **Text Cleaning**: Removal of special characters and non-alphabetic content
2. **Normalization**: Converting text to lowercase
3. **Tokenization**: Splitting text into individual words
4. **Stemming**: Reducing words to their root form using Porter Stemmer
5. **Stopword Removal**: Filtering out common English stopwords
6. **Sequence Padding**: Ensuring uniform input length for LSTM

## Model Architecture

- **Embedding Layer**: Converts text to dense vectors
- **Bidirectional LSTM**: Captures context from both directions
- **Dense Layers**: Fully connected layers for classification
- **Output Layer**: Binary classification (True/Fake)

## Results

The model effectively learns patterns distinguishing fake news from legitimate journalism, analyzing linguistic features, writing style, and content structure.

## Future Improvements

- [ ] Implement Word2Vec embeddings for better semantic representation
- [ ] Add cross-validation for robust performance evaluation
- [ ] Experiment with attention mechanisms
- [ ] Deploy model as web API
- [ ] Create interactive web interface for real-time predictions
- [ ] Expand dataset with more recent news articles

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- Dataset sourced from publicly available fake news datasets
- Built using TensorFlow and NLTK libraries
- Inspired by the need to combat misinformation in digital media

## Contact

For questions or feedback, please open an issue or reach out via patelharsha77@gmail.com.

---

⭐ If you found this project helpful, please consider giving it a star!
