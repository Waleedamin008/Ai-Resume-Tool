# 📄 AI Resume Job Predictor

A production-ready machine learning web application that predicts the most suitable job roles from resume text using Natural Language Processing and Machine Learning.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.54.0-red.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.8.0-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 🌟 Features

- **PDF Resume Upload**: Extract text from PDF resumes automatically
- **Text Input Support**: Alternative option to paste resume text directly
- **Top-3 Predictions**: Get the three most likely job roles with confidence scores
- **Modern UI**: Beautiful gradient design with card-style results display
- **Real-time Analysis**: Instant predictions with loading indicators
- **Production Ready**: Error handling, model validation, and graceful degradation
- **Deployment Ready**: Configured for Streamlit Cloud deployment

## 🎯 Use Cases

- **Job Seekers**: Discover which job roles match your resume profile
- **Career Counselors**: Help clients identify suitable career paths
- **Recruiters**: Quickly categorize and match resumes to job openings
- **HR Departments**: Automate initial resume screening

## 🏗️ Project Structure

```
resume-ai-tool/
│
├── data/
│   └── final_merged_dataset2.csv    # Training dataset
│
├── models/
│   ├── model.pkl                    # Trained Logistic Regression model
│   └── vectorizer.pkl               # TF-IDF vectorizer
│
├── src/
│   ├── preprocess.py                # Text cleaning and normalization
│   ├── train.py                     # Model training pipeline
│   ├── predict.py                   # Prediction functions
│   └── pdf_utils.py                 # PDF text extraction
│
├── app.py                           # Streamlit web application
├── requirements.txt                 # Python dependencies
└── README.md                        # Documentation
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd "AI resume Tool"
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare your dataset**
   - Place your CSV file in the `data/` folder
   - Ensure it has columns: `Resume` (text) and `Category` (label)

### Training the Model

Before using the app, train the model on your dataset:

```bash
python src/train.py
```

This will:
- Load and clean the dataset
- Apply TF-IDF vectorization (max_features=10000)
- Train a Logistic Regression classifier
- Display accuracy and classification report
- Save `model.pkl` and `vectorizer.pkl` in the `models/` folder

### Running the Application

Launch the Streamlit web app:

```bash
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`

## 📊 Model Details

### Architecture

- **Preprocessing**: Text cleaning (lowercase, URL removal, special character removal)
- **Vectorization**: TF-IDF with 10,000 max features
- **Algorithm**: Logistic Regression (multi-class classification)
- **Output**: Top-3 job categories with probability scores

### Performance

The model provides:
- **Accuracy Score**: Displayed during training
- **Classification Report**: Precision, recall, and F1-score per category
- **Confidence Scores**: Probability estimates for each prediction

## 💻 Usage Guide

### Option 1: PDF Upload

1. Click on "Choose a PDF file"
2. Select your resume PDF
3. Click "🔮 Predict Job Roles"
4. View your top 3 job matches with confidence scores

### Option 2: Text Input

1. Expand "💡 Or paste your resume text here"
2. Paste your resume content
3. Click "🔮 Predict Job Roles"
4. View your top 3 job matches with confidence scores

### Understanding Results

Each prediction shows:
- **Rank**: 🥇 1st, 🥈 2nd, 🥉 3rd match
- **Job Category**: The predicted job role
- **Confidence Score**: Percentage indicating prediction confidence
- **Progress Bar**: Visual representation of confidence level

## 🔧 API Reference

### predict_top3(resume_text)

Returns top 3 job predictions with confidence scores.

```python
from src.predict import predict_top3

predictions = predict_top3("Your resume text here...")
# Returns: [('Software Developer', 0.85), ('Data Scientist', 0.10), ...]
```

### clean_text(text)

Cleans and normalizes text data.

```python
from src.preprocess import clean_text

cleaned = clean_text("Raw resume text...")
```

### extract_text_from_pdf(pdf_file)

Extracts text from PDF file.

```python
from src.pdf_utils import extract_text_from_pdf

text = extract_text_from_pdf(uploaded_file)
```

## 🌐 Deployment

### Streamlit Cloud

1. **Push to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git push origin main
   ```

2. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub repository
   - Select `app.py` as the main file
   - Click "Deploy"

3. **Pre-deployment checklist**
   - ✅ Train and commit model files (`models/model.pkl`, `models/vectorizer.pkl`)
   - ✅ Ensure `requirements.txt` is complete
   - ✅ Test locally before deployment

### Other Platforms

The app can also be deployed on:
- **Heroku**: Use `setup.sh` and `Procfile`
- **AWS EC2**: Run with systemd service
- **Docker**: Create a Dockerfile for containerization

## 📦 Dataset Requirements

Your CSV file must have:

| Column | Type | Description |
|--------|------|-------------|
| Resume | Text | Full resume content |
| Category | String | Job role label |

Example:
```csv
Resume,Category
"Experienced Python developer with 5 years...",Software Developer
"Data analyst with expertise in SQL and...",Data Analyst
```

## 🛠️ Customization

### Modify Model Parameters

Edit [`src/train.py`](src/train.py):

```python
# Change TF-IDF features
vectorizer = TfidfVectorizer(max_features=5000)  # Default: 10000

# Adjust model parameters
model = LogisticRegression(
    max_iter=2000,      # Default: 1000
    C=1.0,              # Regularization strength
    solver='lbfgs'      # Optimization algorithm
)
```

### Customize UI Theme

Edit CSS in [`app.py`](app.py):

```python
# Change gradient colors
background: linear-gradient(135deg, #your-color-1 0%, #your-color-2 100%);
```

## 🐛 Troubleshooting

### Model Not Found Error

**Error**: `FileNotFoundError: Model file not found`

**Solution**: Train the model first:
```bash
python src/train.py
```

### PDF Extraction Fails

**Error**: `Error extracting text from PDF`

**Solution**: 
- Ensure PDF is not password-protected
- Try using the text input option instead
- Check if PDF contains actual text (not just images)

### Import Errors

**Error**: `ModuleNotFoundError: No module named 'src'`

**Solution**: Run commands from the project root directory:
```bash
cd "AI resume Tool"
python src/train.py
streamlit run app.py
```

## 📈 Future Enhancements

- [ ] Add support for DOCX resumes
- [ ] Implement deep learning models (BERT, RoBERTa)
- [ ] Add resume scoring and improvement suggestions
- [ ] Multi-language support
- [ ] API endpoint for integration
- [ ] User authentication and history tracking
- [ ] Batch processing for multiple resumes

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👨‍💻 Author

Built with ❤️ using Python, Scikit-learn, and Streamlit

## 🙏 Acknowledgments

- **Scikit-learn**: For machine learning algorithms
- **Streamlit**: For the web framework
- **PDFPlumber**: For PDF text extraction
- **Open Source Community**: For inspiration and support

## 📞 Support

For questions or issues:
- Open an issue on GitHub
- Check the troubleshooting section
- Review the documentation

---

**⭐ Star this repository if you find it helpful!**#   A i - R e s u m e - T o o l  
 #   A i - R e s u m e - T o o l  
 #   A i - R e s u m e - T o o l  
 #   A i - R e s u m e - T o o l  
 #   A i - R e s u m e - T o o l  
 #   A i - R e s u m e - T o o l  
 