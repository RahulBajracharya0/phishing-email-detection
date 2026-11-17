# Phishing Email Detection Using Machine Learning

## Project Overview
This project is a web application that detects whether an email is Safe or Phishing using machine learning techniques.

Users can enter an email subject and body to receive a prediction.

## Technologies Used
- Python
- Flask
- Scikit-learn
- Pandas
- NumPy
- HTML and CSS
- TF-IDF

## Machine Learning Models
The project compares four machine learning classifiers:
- Naive Bayes
- Logistic Regression
- Linear SVM
- Random Forest

A Soft Voting Ensemble is used for the final prediction.

## Running the Project

1. Install the required packages:

   `pip install -r requirements.txt`

2. Start the Flask application:

   `python app.py`

3. Open the application in your browser:

   `http://127.0.0.1:5000`

## Project Features
- Email phishing detection
- Safe and phishing classification
- Model performance dashboard
- Evaluation metrics and visualisations