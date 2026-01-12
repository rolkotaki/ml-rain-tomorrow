# ML Rain Tomorrow
Machine Learning project using Python to predict if it's going to rain tomorrow.

My blog article about this project: [here](https://www.rolkotech.com/articles/machine-learning-will-it-rain-tomorrow)

## Description

In this project I use historical weather data to predict whether it's going to rain tomorrow or not. I use Madrid data as I live here currently.

I go through the important steps of a machine learning project:
* Loading the data
* Cleaning and preprocessing the data
* Feature engineering
* Normalizing the data
* Splitting the data into training and testing sets
* Training the machine learning algorithm(s)
* Evaluating the accuracy and score of the models
* Visualization techniques

I try out three machine learning algorithms:
* Logistic Regression
* Decision Tree
* Gradient Boosting Classifier

Files:
* `predict_rain_initial.py` : models trained with fewer data and more features
* `predict_rain.py` : models trained with more data and without unused features (this is the final and more accurate version)

*(I kept both files because of the blog article.)* 

Third-party libraries used:
* `pandas`
* `numpy`
* `scikit-learn`
* `matplotlib`

Python version used for the development: Python 3.9.6
