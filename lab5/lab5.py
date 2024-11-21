import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Încărcarea datelor din fișier
file_path = r"C:\Users\cristi\Desktop\An4_sem1\ML_Lab\lab5\Groceries_dataset.csv"
df = pd.read_csv(file_path)

# Conversia datei în format datetime
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)

# Verificarea datelor
print("Primele 5 rânduri din dataset:")
print(df.head())

# 2. Exploratory Data Analysis (EDA)
# a) Distribuția produselor
product_counts = df['itemDescription'].value_counts()
plt.figure(figsize=(10, 6))
sns.barplot(x=product_counts.values[:10], y=product_counts.index[:10], palette="pastel")
plt.title("Top 10 produse frecvente")
plt.xlabel("Frecvență")
plt.ylabel("Produs")
plt.show()

# b) Numărul de cumpărături pe lună
df['Month'] = df['Date'].dt.month
monthly_counts = df['Month'].value_counts().sort_index()
plt.figure(figsize=(10, 6))
sns.barplot(x=monthly_counts.index, y=monthly_counts.values, palette="muted")
plt.title("Numărul de cumpărături pe lună")
plt.xlabel("Lună")
plt.ylabel("Număr de cumpărături")
plt.show()

# 2. Transformarea datelor în format market basket
basket = df.groupby(['Member_number', 'itemDescription']).size().unstack(fill_value=0)
basket = basket.astype(bool)  # Convertim în boolean pentru algoritmul Apriori

min_support = 0.02  # Prag mai mic pentru a captura mai multe itemset-uri frecvente
frequent_itemsets = apriori(basket, min_support=min_support, use_colnames=True)
frequent_itemsets.sort_values(by="support", ascending=False, inplace=True)

# Adăugare coloană cu numărul de itemsets
frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))

print("Itemset-uri frecvente:")
print(frequent_itemsets.head())

# 5. Generarea regulilor de asociere
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0, num_itemsets=len(frequent_itemsets))
rules = rules.sort_values(by=["lift", "confidence"], ascending=False)

print("Reguli de asociere generate:")
print(rules.head())

# 6. Vizualizare Lift vs Confidence cu Matplotlib (plt.scatter)
plt.figure(figsize=(10, 6))
plt.scatter(rules['confidence'], rules['lift'], alpha=0.7, edgecolors='k', c=rules['support'], cmap='viridis')
plt.colorbar(label='Support')  # Bara de culoare care indică valorile pentru Support
plt.title('Confidence vs Lift pentru regulile generate')
plt.xlabel('Confidence')
plt.ylabel('Lift')
plt.grid(True)
plt.show()