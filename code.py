import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
dataset = pd.read_csv("HousePricePrediction.csv")
print(dataset.head(5))

# Data Preprocessing
#Here we categorize the features based on their data types (integer, float, or object) and count the number of features in each category
object_cols = dataset.select_dtypes(include=['object']).columns
print("Categorical variables:", len(object_cols))
int_ = dataset.select_dtypes(include=['int64']).columns
print("Integer variables:", len(int_))
fl_cols = dataset.select_dtypes(include=['float64']).columns
print("Float variables:", len(fl_cols))

#Exploratory Data Analysis
#Exploratory Data Analysis involves examining the dataset in depth to uncover patterns, detect anomalies and understand the underlying structure. Before drawing any conclusions, it’s important to analyze all variables carefully.
#Here we will create a heatmap using the Seaborn library to visualize correlations between features.
numerical_dataset = dataset.select_dtypes(include=['int64', 'float64'])

plt.figure(figsize=(12, 6))
sns.heatmap(numerical_dataset.corr(),
            cmap='BrBG',
            fmt='.2f',
            linewidths=2,
            annot=True)
plt.title("Correlation Heatmap of Numerical Features")
plt.tight_layout()
plt.savefig("correlation_heatmap.png")
print("Heatmap saved as correlation_heatmap.png")

#To examine the categorical features, we will create a bar plot to visualize their distributions
unique_values = []
for col in object_cols:
  unique_values.append(dataset[col].unique().size)
plt.figure(figsize=(10,6))
plt.title('No. Unique values of Categorical Features')
plt.xticks(rotation=90)
sns.barplot(x=object_cols,y=unique_values)

plt.figure(figsize=(18, 36))
plt.title('Categorical Features: Distribution')
plt.xticks(rotation=90)
index = 1

for col in object_cols:
    y = dataset[col].value_counts()
    plt.subplot(11, 4, index)
    plt.xticks(rotation=90)
    sns.barplot(x=list(y.index), y=y)
    index += 1

#Data Cleaning
#Data Cleaning is the way to improvise the data or remove incorrect, corrupted or irrelevant data. As in our dataset there are some columns that are not important and irrelevant for the model training. So we can drop that column before training. There are 2 approaches to dealing with empty/null values

#We can easily delete the column/row (if the feature or record is not much important).
#Filling the empty slots with mean/mode/0/NA/etc. (depending on the dataset requirement).
#As Id Column will not be participating in any prediction. So we can Drop it.
dataset.drop(['Id'],
             axis=1,
             inplace=True)

#Replacing SalePrice empty values with their mean values to make the data distribution symmetric.
dataset['SalePrice'] = dataset['SalePrice'].fillna(
  dataset['SalePrice'].mean())

# Dropping records with null values (as the empty records are very less).
new_dataset = dataset.dropna()

# Checking features which have null values in the new dataframe (if there are still any).
print("Null values in new dataset:")
print(new_dataset.isnull().sum())

#OneHotEncoder - For Label categorical features
#One hot Encoding is the best way to convert categorical data into binary vectors. This maps the values to integer values. By using OneHotEncoder, we can easily convert object data into int. So for that firstly we have to collect all the features which have the object datatype. To do so, we will make a loop.
from sklearn.preprocessing import OneHotEncoder

s = (new_dataset.dtypes == 'object')
object_cols = list(s[s].index)
print("Categorical variables:")
print(object_cols)
print('No. of. categorical features: ', 
      len(object_cols))
OH_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
OH_cols = pd.DataFrame(OH_encoder.fit_transform(new_dataset[object_cols]))
OH_cols.index = new_dataset.index
OH_cols.columns = OH_encoder.get_feature_names_out()
df_final = new_dataset.drop(object_cols, axis=1)
df_final = pd.concat([df_final, OH_cols], axis=1)
print("Final dataset shape: ", df_final.shape)

# Splitting Dataset into Training and Testing
# X and Y splitting (i.e. Y is the SalePrice column and the rest of the other columns are X)
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

X = df_final.drop(['SalePrice'], axis=1)
Y = df_final['SalePrice']

X_train, X_valid, Y_train, Y_valid = train_test_split(
    X, Y, train_size=0.8, test_size=0.2, random_state=0)
print("Training set shape: ", X_train.shape)
print("Validation set shape: ", X_valid.shape)

# Model Training and Accuracy
# As we have to train the model to determine the continuous values, so we will be using these regression models.
#
# SVM-Support Vector Machine
# Random Forest Regressor
# Linear Regressor
# And To calculate loss we will be using the mean_absolute_percentage_error module. It can easily be imported by using sklearn library. The formula for Mean Absolute Error is:
#
# MAE = (1/n) * Σ|y_i - ŷ_i|
#
# 1. SVM - Support vector Machine
# Support vector Machine is a supervised machine learning algorithm primarily used for classification tasks though it can also be used for regression. It works by finding the hyperplane that best divides a dataset into classes. The goal is to maximize the margin between the data points and the hyperplane.
from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import mean_absolute_percentage_error

model_SVR = svm.SVR()
model_SVR.fit(X_train,Y_train)
Y_pred = model_SVR.predict(X_valid)

print(mean_absolute_percentage_error(Y_valid, Y_pred))

# 2. Random Forest Regressor
# Random Forest is an ensemble learning method that constructs multiple decision trees during training and outputs the mean
# prediction of the individual trees. It is used for both classification and regression tasks and is known for its high accuracy and ability to handle large datasets with many features.
from sklearn.ensemble import RandomForestRegressor
model_RFR = RandomForestRegressor()
model_RFR.fit(X_train,Y_train)
Y_pred = model_RFR.predict(X_valid)
print(mean_absolute_percentage_error(Y_valid, Y_pred))

# 3. Linear Regressor
# Linear Regression is a linear approach to modeling the relationship between a dependent variable and one or more
#independent variables. The goal is to find the best-fitting straight line through the data points that minimizes the sum of squared differences between the observed and predicted values.
from sklearn.linear_model import LinearRegression
model_LR = LinearRegression()
model_LR.fit(X_train,Y_train)
Y_pred = model_LR.predict(X_valid)
print(mean_absolute_percentage_error(Y_valid, Y_pred))
