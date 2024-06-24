# Fake News Detection App

## Overview

The Fake News Detection App is designed to identify whether a news article is real or fake using natural language processing (NLP) and machine learning techniques. The process involves data preprocessing, model training, prediction, and integrating a user-friendly interface for interaction.

## Table of Contents

- [Dependencies](#dependencies)
- [Data Preprocessing](#data-preprocessing)
  - [Dataset Analysis](#dataset-analysis)
  - [Handling Null Values](#handling-null-values)
  - [Text Merging and Cleaning](#text-merging-and-cleaning)
  - [Stemming](#stemming)
- [Vectorization](#vectorization)
- [Model Training and Evaluation](#model-training-and-evaluation)
  - [Train-Test Split](#train-test-split)
  - [Accuracy Scores](#accuracy-scores)
- [Model Saving](#model-saving)
- [Prediction Function](#prediction-function)
- [Frontend Integration](#frontend-integration)
- [Running the App](#running-the-app)
- [License](#license)

## Dependencies

The project requires several Python libraries including:
- numpy
- pandas
- re (regular expressions)
- nltk (for natural language processing, especially stopwords and stemming)
- sklearn (for machine learning tasks like vectorization, model training, and evaluation)
- joblib (for saving and loading the trained model)
- gradio (for building the web interface)

## Data Preprocessing

### Dataset Analysis

The dataset is loaded using pandas. The initial analysis includes checking the shape (number of rows and columns) of the dataset and displaying the first few rows to understand its structure.

### Handling Null Values

Null values in the dataset are replaced with empty strings to ensure that the preprocessing steps can be applied without errors.

### Text Merging and Cleaning

The author name and news title are merged into a single column to form the content of the news. This step helps in consolidating the information for text processing.

### Stemming

Stemming involves reducing words to their root forms. This is done to normalize the text and reduce variations of words to a common base. The text is also cleaned by removing non-alphabetic characters and converting everything to lowercase. Stopwords (common words like "the", "is", "in") are removed as they do not contribute significant meaning to the content.

## Vectorization

The cleaned and stemmed text data is converted into numerical format using TF-IDF (Term Frequency-Inverse Document Frequency) vectorization. This step transforms the text into a format that can be used by machine learning algorithms.

## Model Training and Evaluation

### Train-Test Split

The dataset is split into training and testing sets. This allows for evaluating the performance of the model on unseen data.

### Accuracy Scores

A logistic regression model is trained on the training set. The accuracy of the model is then evaluated on both the training and testing sets to ensure it is performing well and not overfitting.

## Model Saving

The trained model and the TF-IDF vectorizer are saved using joblib. This allows for reloading the model and vectorizer later without having to retrain them.

## Prediction Function

Functions are defined to preprocess new articles and make predictions using the trained model. These functions handle the necessary text cleaning, stemming, vectorization, and prediction.

## Frontend Integration

Gradio is used to build a web interface where users can input news articles and get predictions. The interface allows users to select a news title from a dropdown menu and see whether it is classified as real or fake.

## Running the App

To run the app, execute the script. The app will launch in the browser, providing an interactive way for users to check the authenticity of news articles.
