import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import time

# Citește dataset-ul
file_path = r"C:\Users\cristi\Desktop\An4_sem1\ML_Lab\lab7\gym_members_exercise_tracking.csv"
data = pd.read_csv(file_path)

# Explorarea dataset-ului
print("Primele 5 rânduri ale dataset-ului:\n", data.head())
print("\nInformații despre dataset:\n", data.info())
print("\nDescriere statistică:\n", data.describe())
# Analiza Exploratorie a Datelor (EDA)
plt.figure(figsize=(10, 6))
sns.countplot(x="Workout_Type", data=data)
plt.title("Distribuția tipurilor de antrenament")
plt.xlabel("Workout_Type")
plt.ylabel("Frecvența")
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x="Workout_Type", y="Calories_Burned", data=data)
plt.title("Calories Burned vs Workout Type")
plt.xlabel("Workout_Type")
plt.ylabel("Calories_Burned")
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(x="Age", y="Calories_Burned", hue="Workout_Type", data=data)
plt.title("Calories Burned în funcție de vârstă și tipul antrenamentului")
plt.xlabel("Age")
plt.ylabel("Calories_Burned")
plt.show()

# Pregătirea datelor
label_encoder = LabelEncoder()
data['Gender'] = label_encoder.fit_transform(data['Gender'])
data['Workout_Type'] = label_encoder.fit_transform(data['Workout_Type'])

X = data.drop('Workout_Type', axis=1)
y = data['Workout_Type']

# Împărțirea datelor în seturi de antrenament și testare
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scalează datele pentru Logistic Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model 1: Logistic Regression
logistic_model = LogisticRegression(max_iter=5000, solver='saga')
logistic_model.fit(X_train_scaled, y_train)
logistic_preds = logistic_model.predict(X_test_scaled)
logistic_accuracy = accuracy_score(y_test, logistic_preds)

# Model 2: Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_preds)

# Rezultatele finale
print("\nRezultatele algoritmilor:")
print(f"Logistic Regression - Acuratețe: {logistic_accuracy:.2f}")
print(f"Random Forest - Acuratețe: {rf_accuracy:.2f}")

# ---- Prezicerea tipului de antrenament pentru un sportiv ---- #
print("\nSelectați un rând din dataset pentru predicție.")
selected_index = int(input("Introduceți indexul rândului (între 0 și {}): ".format(len(data) - 1)))

# Extrage datele sportivului selectat
selected_sportiv = X.iloc[[selected_index]]  # Extrage rândul specificat
print("\nCaracteristicile sportivului selectat:")
print(selected_sportiv)

# Scalează datele pentru Logistic Regression
selected_sportiv_scaled = scaler.transform(selected_sportiv)

# Preziceri
logistic_prediction = logistic_model.predict(selected_sportiv_scaled)
rf_prediction = rf_model.predict(selected_sportiv)

# Decodificarea rezultatelor
workout_types = label_encoder.inverse_transform(range(len(data['Workout_Type'].unique())))
print("\nPredicții pentru sportivul selectat:")
print(f"Logistic Regression: {workout_types[logistic_prediction[0]]}")
print(f"Random Forest: {workout_types[rf_prediction[0]]}")
