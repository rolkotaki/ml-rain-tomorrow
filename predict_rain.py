import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
from sklearn import preprocessing


# plt.show() lines are commented


# To display all column from the dataframe
pd.set_option('display.max_columns', None)


# Loading the historical weather data into a dataframe
df_1 = pd.read_csv('data/Madrid 2006-01-01 to 2008-06-30.csv')
df_2 = pd.read_csv('data/Madrid 2008-07-01 to 2010-12-31.csv')
df_3 = pd.read_csv('data/Madrid 2011-01-01 to 2013-06-30.csv')
df_4 = pd.read_csv('data/Madrid 2013-07-01 to 2015-12-31.csv')
df_5 = pd.read_csv('data/Madrid 2016-01-01 to 2018-06-30.csv')
df_6 = pd.read_csv('data/Madrid 2018-07-01 to 2020-12-31.csv')
df_7 = pd.read_csv('data/Madrid 2021-01-01 to 2023-04-25.csv')
df = pd.concat([df_1, df_2, df_3, df_4, df_5, df_6, df_7], ignore_index=True)

print("Number of rows in the dataframe: {}\n".format(len(df)))


# Dropping unnecessary columns
df.drop(['name', 'datetime', 'temp', 'feelslikemax', 'feelslikemin', 'feelslike', 'precipprob', 'precipcover',
         'preciptype', 'snow', 'snowdepth', 'visibility', 'solarradiation', 'solarenergy', 'uvindex', 'severerisk',
         'sunrise', 'sunset', 'moonphase', 'conditions', 'description', 'stations', 'icon'],
        axis=1, inplace=True)


print("Descriptive statistics of the dataframe:")
print(df.describe(), end='\n\n')


# Checking for columns with null values and doing the necessary cleaning
print("Number of rows containing null values:")
print(df.isna().sum())
df.drop(['windgust'], axis=1, inplace=True)
df.dropna(inplace=True)
print("Number of rows in the dataframe after dropping rows with null values: {}\n".format(len(df)))


# Adding the target column
df['rain_today'] = (df['precip'] > 0.0).astype(np.int8)
df['rain_tomorrow'] = df['rain_today'].shift(-1)
df.iloc[-1, df.columns.get_loc('rain_tomorrow')] = 0  # no rain on 26th April 2023
df['rain_tomorrow'] = df['rain_tomorrow'].astype(np.int8)
print("The added target column and helper column:")
print(df[['rain_today', 'rain_tomorrow']].tail(), end='\n\n')


# Checking the correlation of the features to the target
df_corr = df.corr(method='pearson')[['rain_tomorrow']]
df_corr.sort_values('rain_tomorrow', ascending=False, inplace=True)
print("Correlation of the features to the target:")
print(df_corr, end='\n\n')

# Visualizing the correlation of the columns
plt.bar(x=df_corr.index, height=df_corr['rain_tomorrow'])
plt.xticks(rotation=90)
# plt.show()

# Dropping columns with very low correlation
df.drop(['winddir'], axis=1, inplace=True)


# Normalizing the data
scaler = preprocessing.MinMaxScaler()
scaler.fit(df)
df = pd.DataFrame(scaler.transform(df), index=df.index, columns=df.columns)
print("Data after normalization:\n{}\n".format(df.head()))


# Splitting the data into training and testing sets
X = df.loc[:, df.columns != 'rain_tomorrow']  # features
y = df['rain_tomorrow']                       # target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=123)

print("rain_tomorrow ratio in original dataset: {}".format(y[y==1].count() / y.count()))
print("rain_tomorrow ratio in training dataset: {}".format(y_train[y_train==1].count() / y_train.count()))
print("rain_tomorrow ratio in testing dataset:  {}".format(y_test[y_test==1].count() / y_test.count()), end='\n\n')


# Building the models


# Logistic Regression
print("***** Logistic Regression *****")

log_reg = LogisticRegression(random_state=123)
log_reg.fit(X_train, y_train)

x_pred = log_reg.predict(X_train)
score = accuracy_score(y_train, x_pred)
print("Prediction score for training data: {}".format(score))
y_pred = log_reg.predict(X_test)
score = accuracy_score(y_test, y_pred)
print("Prediction score for test data: {}".format(score))

print(classification_report(y_pred, y_test), end='\n\n')

# Plotting feature importances
importance = log_reg.coef_[0]
feature_importance = pd.Series(importance, index=X_train.columns)
feature_importance.sort_values().plot(kind='bar')
plt.ylabel('Importance')
# plt.show()


# Decision Tree
print("***** Decision Tree *****")

dtc = DecisionTreeClassifier(random_state=123)
param_grid = {'max_depth': [3, 4, 5, 6],
              'min_samples_split': [2, 3, 5, 10],
              'min_samples_leaf': [1, 2, 3, 5],
              # 'criterion': ['gini', 'entropy', 'log_loss']
              }
gs = GridSearchCV(estimator=dtc, param_grid=param_grid)
gs.fit(X_train, y_train)
dtc = gs.best_estimator_
print("Best model: {}".format(dtc))
dtc.fit(X_train, y_train)

x_pred = dtc.predict(X_train)
score = accuracy_score(y_train, x_pred)
print("Prediction score for training data: {}".format(score))
y_pred = dtc.predict(X_test)
score = accuracy_score(y_test, y_pred)
print("Prediction score for test data: {}".format(score))

print(classification_report(y_pred, y_test), end='\n\n')

# Plotting feature importances
importance = dtc.feature_importances_
feature_importance = pd.Series(importance, index=X_train.columns)
feature_importance.sort_values().plot(kind='bar')
plt.ylabel('Importance')
# plt.show()

# Plotting the decision tree
plot_tree(dtc, feature_names=list(X.columns), class_names=True, filled=True, fontsize=7)
# plt.show()


# GradientBoostingClassifier
print("Gradient Boosting Classifier")

gbc = GradientBoostingClassifier()
param_grid = {'n_estimators': [100, 200, 500],
              'max_depth': [3, 4, 5],
              'min_samples_leaf': [7, 10, 12],
              'loss': ['log_loss']}
gs = GridSearchCV(gbc, param_grid=param_grid, n_jobs=4)
gs.fit(X_train, y_train)
gbc = gs.best_estimator_
print("Best model: {}".format(gbc))
gbc.fit(X_train, y_train)

x_pred = gbc.predict(X_train)
score = accuracy_score(y_train, x_pred)
print("Prediction score for training data: {}".format(score))
y_pred = gbc.predict(X_test)
score = accuracy_score(y_test, y_pred)
print("Prediction score for test data: {}".format(score))

print(classification_report(y_pred, y_test))
