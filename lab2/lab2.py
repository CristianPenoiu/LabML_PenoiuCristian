# Importarea librăriilor necesare
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV

# Încărcarea setului de date
csv_path = r"D:\FAC\ML_lab\lab2\gym_members_exercise_tracking.csv"
df = pd.read_csv(csv_path)

# Informații generale despre date
print(df.info())
print(df.describe())
print(df.head())

# Verificare valori lipsă
print(df.isnull().sum())

# Analiza exploratorie a datelor (vizualizări)
sns.countplot(data=df, x="Gender")
plt.title("Distribuția pe genuri")
plt.show()

sns.histplot(data=df, x='Age', kde=True)
plt.title("Distribuția vârstei")
plt.show()

sns.scatterplot(data=df, x='Weight (kg)', y='BMI', hue='Gender')
plt.title("Greutatea vs BMI")
plt.show()

sns.barplot(data=df, x='Workout_Type', y='Calories_Burned', hue='Gender')
plt.title("Calorii arse în funcție de tipul antrenamentului")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()

# Preprocesarea datelor: etichetare categorii
label_encoder = LabelEncoder()
df['Gender'] = label_encoder.fit_transform(df['Gender'])
df['Workout_Type'] = label_encoder.fit_transform(df['Workout_Type'])
df['Experience_Level'] = label_encoder.fit_transform(df['Experience_Level'])

# Eliminarea coloanelor irelevante (de exemplu Member_ID)
#df = df.drop(columns=['Member_ID'])

# Verificare corelații între variabile
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Matricea de corelații")
plt.show()

# Pregătirea datelor pentru modelare
X = df.drop(columns=['Workout_Type'])  # Variabile independente
y = df['Workout_Type']  # Variabilă dependentă

# Împărțirea setului de date în antrenament și test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Afișare dimensiuni seturi de date
print(f"Dimensiunea setului de antrenament: {X_train.shape}")
print(f"Dimensiunea setului de test: {X_test.shape}")

# Creare model DecisionTreeClassifier
dt_model = DecisionTreeClassifier(random_state=42)

# Definire grilă de hiperparametri pentru GridSearchCV
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 10, 20, 30, 40],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Aplicare GridSearch pentru optimizarea hiperparametrilor
grid_search = GridSearchCV(estimator=dt_model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

# Afișare cei mai buni hiperparametri
print("Cei mai buni hiperparametri: ", grid_search.best_params_)

# Modelul optimizat
best_model = grid_search.best_estimator_

# Predicții pe setul de test
y_pred = best_model.predict(X_test)

# Evaluarea performanței modelului
accuracy = accuracy_score(y_test, y_pred)
print(f'Acuratețea modelului: {accuracy * 100:.2f}%')

# Afișare raport de clasificare
print("Raport de clasificare:")
print(classification_report(y_test, y_pred))

