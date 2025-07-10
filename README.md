# Resume Classifier

## Overview
This project implements a resume classification system that predicts job roles based on resume text. It uses natural language processing (NLP) techniques and a logistic regression model to categorize resumes into various job roles.

## Features
- Cleans and preprocesses resume text using regex and NLTK stopwords
- Converts text to numerical features using TF-IDF vectorization
- Trains a logistic regression model to classify resumes
- Provides predictions with confidence scores for new resume inputs
- Evaluates model performance with classification reports and confusion matrices

## Requirements
- Python 3.x
- Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn, nltk
- Optional: Google Colab for file upload functionality

## Installation
1. Clone the repository:
   ```
   git clone <repository-url>
   ```
2. Install dependencies:
   ```
   pip install pandas numpy matplotlib seaborn scikit-learn nltk
   ```
3. Download NLTK stopwords:
   ```python
   import nltk
   nltk.download('stopwords')
   ```

## Usage
1. Prepare a CSV file (`UpdatedResumeDataSet.csv`) with columns `Resume` (text) and `Category` (job role).
2. Run the script:
   ```bash
   python resume_classifier.py
   ```
3. Input a sample resume text to predict the job role and confidence score.

## Example
```python
sample_resume = """
Experienced in Python, machine learning, and data visualization.
Worked on predictive models, cleaned large datasets, deployed solutions using Flask and cloud.
Familiar with TensorFlow, Pandas, NumPy, and GCP.
"""
role, confidence = predict_role_confidence(sample_resume)
print(f"Predicted Role: {role}, Confidence: {confidence}%")
```

## Output
The script outputs:
- A classification report with precision, recall, and F1-score
    ![Output Screenshot](output%20screenshots/Screenshot%202025-07-10%20134518.png)
- A confusion matrix heatmap
- Predicted job role and confidence for sample resume input


---

## Notes
- The model uses a maximum of 1000 TF-IDF features for efficiency.
- Adjust `max_iter` in `LogisticRegression` if convergence issues occur.
- Ensure the input CSV file is properly formatted to avoid errors.
