import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# 1. Încărcarea datelor
file_path = r"C:\Users\cristi\Desktop\An4_sem1\ML_Lab\lab6\Country-data.csv"
df = pd.read_csv(file_path)

# 2. Exploratory Data Analysis (EDA)
# Verificăm primele 5 rânduri ale dataset-ului
print(df.head())

# Verificăm dacă există valori lipsă
print("\nValori lipsă:")
print(df.isnull().sum())

# Descrierea statistică a datelor (pentru fiecare feature)
print("\nDescriere statistică:")
print(df.describe())

# Distribuția valorilor pentru variabila țintă (country)
plt.figure(figsize=(8, 6))
sns.countplot(x='country', data=df)
plt.title('Distribuția țării (target) în dataset')
plt.xlabel('Țara')
plt.ylabel('Frecvență')
plt.xticks(rotation=45)
plt.show()

# 3. Pregătirea datelor
# Separam caracteristicile (features) de țară (target)
X = df.drop('country', axis=1)
y = df['country']

# 4. Standardizarea datelor pentru PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. Reducerea dimensionalității folosind PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# 6. Calculăm informațiile pentru fiecare componentă PCA
pca_informations = pd.DataFrame({
    'Components': [f'PC {i+1}' for i in range(X.shape[1])],
    'Eigen values': pca.explained_variance_,
    'Explained variance (%)': pca.explained_variance_ * 100,
    'Explained variance cumulative (%)': [sum(pca.explained_variance_ratio_[:i+1])*100 for i in range(X.shape[1])]
})

# 7. Vizualizare a varianței explicate și varianței explicate cumulative
plt.figure(figsize=(12, 6))

# Barplot pentru varianta explicată
sns.barplot(data=pca_informations, x='Components', y='Explained variance (%)', color='skyblue')

# Linie pentru varianța explicată cumulativă
sns.lineplot(data=pca_informations, x='Components', y='Explained variance cumulative (%)', color='gray', linestyle='--', marker='o')

plt.title('Varianța explicată și varianța cumulativă explicată per componentă PCA')
plt.xlabel('Componente PCA')
plt.ylabel('Varianța explicată (%)')
plt.xticks(rotation=45)
plt.show()

# 8. Evaluarea performanței pentru fiecare număr de componente PCA (succesiv)
accuracy_list = []  # Liste pentru a salva performanța modelului
components_range = range(1, X.shape[1] + 1)  # De la 1 la numărul de caracteristici originale

# Antrenarea modelului pentru fiecare număr de componente PCA
for n_components in components_range:
    # Reducem dimensionalitatea la 'n_components' componente
    X_reduced = X_pca[:, :n_components]
    
    # Împărțim datele în set de antrenament și test
    X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.3, random_state=42)
    
    # Creăm și antrenăm modelul Random Forest
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    # Predicții și evaluare
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_list.append(accuracy)

# 9. Vizualizare performanței în funcție de numărul de componente PCA
plt.figure(figsize=(10, 6))
plt.plot(components_range, accuracy_list, marker='o', color='b', linestyle='-', markersize=6)
plt.title('Performanța modelului în funcție de numărul de componente PCA')
plt.xlabel('Număr de componente PCA')
plt.ylabel('Acuratețea modelului')
plt.grid(True)
plt.show()

# 10. Vizualizare a varianței explicate cumulativă
explained_variance = np.cumsum(pca.explained_variance_ratio_)
plt.figure(figsize=(10, 6))
plt.plot(range(1, X.shape[1] + 1), explained_variance, marker='o', color='r', linestyle='-', markersize=6)
plt.title('Varianța explicată cumulativă de fiecare componentă PCA')
plt.xlabel('Număr de componente PCA')
plt.ylabel('Varianța explicată cumulativă')
plt.grid(True)
plt.show()

# 11. Proiectarea datelor pe primele două componente PCA pentru vizualizare
final_pca = PCA(n_components=2)
final_result = final_pca.fit_transform(X_scaled)

plt.figure(figsize=(12, 4), dpi=100)
sns.scatterplot(x=final_result[:, 0], y=final_result[:, 1], s=60)
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.axhline(y=0, ls='--', c='red')
plt.axvline(x=0, ls='--', c='red')
plt.title('Proiecția datelor pe primele două componente PCA')
plt.show()
