# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

# Load datasets
train_df = pd.read_csv(r'C:\Users\cristi\Desktop\An4_sem1\ML_Lab\lab1\train.csv')
test_df = pd.read_csv(r'C:\Users\cristi\Desktop\An4_sem1\ML_Lab\lab1\test.csv')

# Preview the data
print(train_df.head())

# Step 1: Data Preprocessing
# Handle missing values
# For numerical data, we'll use the mean, and for categorical data, we'll use the most frequent value
num_features = train_df.select_dtypes(include=[np.number]).columns
cat_features = train_df.select_dtypes(include=[object]).columns

# Impute missing values
imputer_num = SimpleImputer(strategy='mean')
imputer_cat = SimpleImputer(strategy='most_frequent')

train_df[num_features] = imputer_num.fit_transform(train_df[num_features])
train_df[cat_features] = imputer_cat.fit_transform(train_df[cat_features])

# Convert categorical features to numerical using Label Encoding
label_encoders = {}
for col in cat_features:
    le = LabelEncoder()
    train_df[col] = le.fit_transform(train_df[col])
    label_encoders[col] = le

# Step 2: Feature Selection
# We will drop features like 'Id' as it is not useful for prediction
X = train_df.drop(['Id', 'SalePrice'], axis=1)  # Features
y = train_df['SalePrice']  # Target variable

# Split the data into train and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train Regression Model
# Using LinearRegression as requested
model = LinearRegression()
model.fit(X_train, y_train)

# Step 4: Evaluate the model
y_pred = model.predict(X_valid)

# Calculate evaluation metrics
rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
mae = mean_absolute_error(y_valid, y_pred)

print(f'Root Mean Squared Error: {rmse}')
print(f'Mean Absolute Error: {mae}')

# Step 5: Plot the results
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_valid, y=y_pred)
plt.plot([min(y_valid), max(y_valid)], [min(y_pred), max(y_pred)], color='red', linestyle='--')  # Adding regression line
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted House Prices (Linear Regression)')
plt.show()

# Step 6: Predict on the test set
test_df[num_features] = imputer_num.transform(test_df[num_features])
test_df[cat_features] = test_df[cat_features].apply(lambda col: label_encoders[col.name].transform(col))

X_test = test_df.drop('Id', axis=1)
test_predictions = model.predict(X_test)

# Output predictions
output = pd.DataFrame({'Id': test_df['Id'], 'SalePrice': test_predictions})
output.to_csv('house_price_predictions.csv', index=False)
