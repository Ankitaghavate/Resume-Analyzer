# Resume Screening and Job Role Classification

This project focuses on classifying resumes into job categories using Natural Language Processing (NLP) and Machine Learning techniques.

## 📁 Dataset

The dataset contains resumes with the following columns:
- `Category`: The job role (e.g., Data Scientist, Web Developer, Java Developer, Sales, Mechanical, etc.)
- `Resume`: Text summary of the candidate’s resume

## 🧪 Exploratory Data Analysis
- Checked the number of resumes and unique job categories
- Performed basic text analysis on the resume data

## 🔢 Preprocessing

### Label Encoding
- Converted job role categories (text) into numeric labels using Label Encoding to make them suitable for machine learning models.

### Text Vectorization (TF-IDF)
- Applied **TF-IDF (Term Frequency–Inverse Document Frequency)** using `TfidfVectorizer` from `scikit-learn`
- Transformed textual resume data into numerical vectors representing the importance of words

## 🧠 Model Used

### OneVsRestClassifier with K-Nearest Neighbors (KNN)
- Used the `OneVsRestClassifier` wrapper to handle multi-class classification
- Trained the model using **K-Nearest Neighbors (KNN)** algorithm
- Split the data into **training** and **testing** sets

## ✅ Results
- Achieved an impressive **accuracy of around 98%**
- The model successfully predicted job categories based on resume content

## 📊 Features

- **Independent Feature**: Resume text
- **Dependent Feature**: Job category

## 💻 Technologies Used

- Python
- Pandas
- Scikit-learn
- Natural Language Processing (TF-IDF)

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd resume-screening

2. Install dependencies & Run the script:
  
 ```bash
   pip install -r requirements.txt
   python resume_classifier.py
