# 🔍 EchoTruth - AI-Powered Fake News Detection System

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)
![Gradio](https://img.shields.io/badge/gradio-4.0+-green.svg)
![Accuracy](https://img.shields.io/badge/accuracy-99%25-brightgreen.svg)

An intelligent machine learning system that detects fake news and misinformation with **99% accuracy** using advanced Natural Language Processing techniques.

## 🌟 User Interface 
![Image Alt](https://github.com/rajatDpatil/EchoTruth---Fake-News-Detection-System/blob/e295a697a2aefa16e5efdef4a084fdecf7d82c0f/UI.jpg)

## 🚀 Features

- **High Accuracy**: 99% accuracy on test dataset
- **Real-time Detection**: Instant analysis of news articles
- **User-friendly Interface**: Beautiful Gradio web app
- **Confidence Scoring**: Shows prediction confidence levels
- **Sample Testing**: Built-in real and fake news samples
- **Mobile Responsive**: Works seamlessly on all devices

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | 99% |
| Precision | 99% |
| Recall | 99% |
| F1-Score | 99% |

## 🛠️ Technology Stack

- **Machine Learning**: Scikit-learn
- **Algorithm**: Logistic Regression with TF-IDF Vectorization
- **Web Interface**: Gradio
- **Data Processing**: Pandas, NumPy
- **Model Serialization**: Joblib

## 📁 Project Structure

```
EchoTruth/
├── model_training.ipynb    # Main ML training notebook
├── app.py                  # Gradio web application
├── requirements.txt        # Python dependencies
├── vectorizer.jb          # Trained TF-IDF vectorizer
├── lr_model.jb            # Trained logistic regression model
├── Fake.csv               # Fake news dataset
├── True.csv               # Real news dataset
└── README.md              # Project documentation
```

## 🚀 Quick Start

### Run in Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1AXpywfz4F2QHAFiRfeJfM89ntk1e1gjS?usp=sharing)

1. Click the Colab badge above
2. Run all cells to train the model
3. Run the Gradio app cell for instant deployment

## 📖 How It Works

1. **Data Preprocessing**: Clean and prepare news article text
2. **Feature Extraction**: Convert text to numerical features using TF-IDF
3. **Model Training**: Train Logistic Regression classifier
4. **Prediction**: Classify news as Real (1) or Fake (0)
5. **Confidence Scoring**: Calculate prediction probability

## 🎯 Usage Examples

### Detecting Fake News
```python
# Example fake news
text = "Scientists discover aliens living in pizza according to unnamed sources"
# Result: ❌ FAKE NEWS (Confidence: 94.2%)
```

### Detecting Real News  
```python
# Example real news
text = "Scientists develop new cancer treatment using immunotherapy techniques"
# Result: ✅ REAL NEWS (Confidence: 96.8%)
```

## 📈 Model Training Details

- **Dataset Size**: 44,898 articles (50% real, 50% fake)
- **Features**: TF-IDF vectorization with 5000 features
- **Algorithm**: Logistic Regression with L2 regularization
- **Cross-validation**: 5-fold cross-validation
- **Training Time**: ~2 minutes on Google Colab

## 🔧 Configuration

You can customize the model by modifying these parameters:

```python
# TF-IDF Parameters
max_features = 5000
stop_words = 'english'
ngram_range = (1, 2)

# Logistic Regression Parameters
C = 1.0
max_iter = 1000
random_state = 42
```

## 📊 Dataset Information

- **Source**: Kaggle Fake News Dataset
- **Real News**: Verified articles from reliable sources
- **Fake News**: Flagged misinformation and satire articles
- **Preprocessing**: Text cleaning, lowercasing, punctuation removal

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Kaggle for the fake news dataset
- Gradio team for the amazing web interface framework
- Scikit-learn developers for the ML tools

## ⚠️ Disclaimer

This tool is designed for educational and research purposes. While it achieves high accuracy, always verify important news with multiple trusted sources before making decisions based on the predictions.

---

**Made with ❤️ and AI**
