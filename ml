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



# EXP 2 Linear Regression

import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

diabetes = load_diabetes()
df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
df['target'] = diabetes.target

X = df[['bmi']] 
y = df['target'] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R²): {r2}")

plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_test['bmi'], y=y_test, color='blue', label='Actual values')
sns.lineplot(x=X_test['bmi'], y=y_pred, color='red', label='Predicted values')
plt.title('Linear Regression: BMI vs Disease Progression')
plt.xlabel('BMI')
plt.ylabel('Disease Progression')
plt.legend()
plt.show()



# EXP 3 Naive Bayes

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import seaborn as sns
df = sns.load_dataset('titanic')

df = df[['pclass', 'sex', 'embarked', 'age', 'survived']]

df.dropna(inplace=True)

le = LabelEncoder()
df['sex'] = le.fit_transform(df['sex'])  
df['embarked'] = le.fit_transform(df['embarked'])  
df['pclass'] = le.fit_transform(df['pclass'])  

X = df[['pclass', 'sex', 'embarked', 'age']]
y = df['survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = CategoricalNB()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))




# EXP 4 Feature Engineering

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

import seaborn as sns
df = sns.load_dataset('titanic')

df = df[['pclass', 'sex', 'age', 'fare', 'embarked', 'survived']]
df.dropna(inplace=True)

df = pd.get_dummies(df, columns=['sex', 'embarked'], drop_first=True)

X = df.drop('survived', axis=1)
y = df['survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

def evaluate_model(X_train, X_test, y_train, y_test, model_name="Default"):
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel: {model_name}")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(classification_report(y_test, y_pred))

evaluate_model(X_train, X_test, y_train, y_test, model_name="No Feature Engineering")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
evaluate_model(X_train_scaled, X_test_scaled, y_train, y_test, model_name="Scaled Features")

select_kbest = SelectKBest(chi2, k=4)
X_train_selected = select_kbest.fit_transform(X_train, y_train)
X_test_selected = select_kbest.transform(X_test)
evaluate_model(X_train_selected, X_test_selected, y_train, y_test, model_name="Feature Selection (KBest)")

pca = PCA(n_components=3)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
evaluate_model(X_train_pca, X_test_pca, y_train, y_test, model_name="PCA (Dimensionality Reduction)")


print("\nClassification Report:")
print(classification_report(y_test, y_pred))




# EXP 5 DBSCAN

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

centers = [[1, 1], [-1, -1], [1, -1]] 
cluster_std = [0.4, 0.3, 0.2] 

X, y = make_blobs(n_samples=750, centers=centers, cluster_std=cluster_std, random_state=42)
X = StandardScaler().fit_transform(X)

dbscan = DBSCAN(eps=0.3, min_samples=10) 
dbscan.fit(X)

labels = dbscan.labels_

unique_labels = set(labels)
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

for k, col in zip(unique_labels, colors):
    if k == -1:
        col = 'k'
    class_member_mask = (labels == k)
    xy = X[class_member_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=6)

plt.title("DBSCAN Clustering with Noise")
plt.show()

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)

print(f"Estimated number of clusters: {n_clusters}")
print(f"Estimated number of noise points: {n_noise}")




# EXP 6 K-means Clustering

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

X, y = make_blobs(n_samples=500, centers=5, cluster_std=1.0, random_state=42)
X = StandardScaler().fit_transform(X)

wcss = [] 
K_values = range(1, 11) 

for k in K_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)  

plt.figure(figsize=(8, 5))
plt.plot(K_values, wcss, 'bo-', markersize=8)
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of clusters (K)')
plt.ylabel('WCSS (Inertia)')
plt.show()

optimal_k = 5  

kmeans_optimal = KMeans(n_clusters=optimal_k, random_state=42)
y_kmeans = kmeans_optimal.fit_predict(X)

plt.figure(figsize=(8, 5))
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap='viridis', s=50)
plt.scatter(kmeans_optimal.cluster_centers_[:, 0], kmeans_optimal.cluster_centers_[:, 1], s=300, c='red', marker='X', label='Centroids')
plt.title(f'K-Means Clustering with K={optimal_k}')
plt.legend()
plt.show()


