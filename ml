# Experiment 1 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

diabetes = load_diabetes()
df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
df['target'] = diabetes.target

plt.figure(figsize=(8, 5))
sns.scatterplot(x='bmi', y='target', data=df)
plt.title('BMI vs Disease Progression')
plt.show()

plt.figure(figsize=(8, 5))
sns.boxplot(x='bmi', data=df)
plt.title('Boxplot of BMI')
plt.show()

sns.pairplot(df[['bmi', 'bp', 's1', 's5', 'target']])
plt.show()

X = df[['bmi']] 
y = df['target'] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse}")
print(f"R² Score: {r2}")

df.loc[0:5, 'bmi'] = np.nan 
print("\nMissing values before handling:")
print(df.isnull().sum())

df['bmi'].fillna(df['bmi'].median(), inplace=True)
print("\nMissing values after handling:")
print(df.isnull().sum())

Q1 = df['bmi'].quantile(0.25)
Q3 = df['bmi'].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df['bmi'] < (Q1 - 1.5 * IQR)) | (df['bmi'] > (Q3 + 1.5 * IQR))]
print(f"\nNumber of outliers in BMI: {len(outliers)}")

df_cleaned = df[~((df['bmi'] < (Q1 - 1.5 * IQR)) | (df['bmi'] > (Q3 + 1.5 * IQR)))]

print(f"\nNumber of duplicate rows: {df.duplicated().sum()}")

df_cleaned = df_cleaned.drop_duplicates()
print(f"Number of rows after removing duplicates: {len(df_cleaned)}")
