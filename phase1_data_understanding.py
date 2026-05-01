import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

#phase 1: data understanding
df = pd.read_csv('./data/raw/train_data.csv')

print("=" * 60)
print("STEP 1: Basic Info")
print("=" * 60)
print(f"Shape: {df.shape}")
print(f"\nColumns:\n{df.columns.tolist()}")
print(f"\nDtypes:\n{df.dtypes}")
print(f"\nFirst 5 rows:\n{df.head()}")

print("\n" + "=" * 60)
print("STEP 2: Missing Values")
print("=" * 60)
missing = pd.DataFrame({
    "Nulls": df.isnull().sum(),
    "Percent": (df.isnull().mean() * 100).round(2)
}).sort_values("Percent", ascending=False)
print(missing[missing["Nulls"] > 0])

print("\n" + "=" * 60)
print("STEP 3: Target Column — GamePopularity")
print("=" * 60)
print(df['GamePopularity'].value_counts())
print("\nClass distribution (%):")
print((df['GamePopularity'].value_counts(normalize=True) * 100).round(2))

# Plot class distribution
fig, ax = plt.subplots(figsize=(7, 4), facecolor='#F8F7F4')
ax.set_facecolor('#F0EFE8')
counts = df['GamePopularity'].value_counts()
bars = ax.bar(counts.index, counts.values,
              color=['#4C8EDA', '#E8593C', '#1D9E75'], alpha=0.85, edgecolor='white')
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
            str(val), ha='center', fontsize=10, fontweight='500')
ax.set_title('Class Distribution — GamePopularity', fontsize=13,
             fontweight='600', color='#2C2C2A')
ax.set_xlabel('Class', fontsize=10)
ax.set_ylabel('Count', fontsize=10)
plt.tight_layout()
plt.savefig('./plots/phase1_class_distribution.png', dpi=130,
            bbox_inches='tight', facecolor='#F8F7F4')
plt.close()
print("\nSaved: plots/phase1_class_distribution.png")

# Check imbalance warning
ratios = df['GamePopularity'].value_counts(normalize=True)
if ratios.max() > 0.6:
    print("\n  WARNING: Class imbalance detected!")
    print("    Consider using class_weight='balanced' in classifiers.")
else:
    print("\n Classes are reasonably balanced.")

print("\n" + "=" * 60)
print("STEP 4: Feature Types")
print("=" * 60)
num_cols = df.select_dtypes(include=np.number).columns.tolist()
cat_cols = df.select_dtypes(include='object').columns.tolist()
print(f"Numerical columns ({len(num_cols)}): {num_cols}")
print(f"\nCategorical/Text columns ({len(cat_cols)}): {cat_cols}")

print("\n" + "=" * 60)
print("STEP 5: Basic Statistics")
print("=" * 60)
print(df.describe().T.to_string())

print("\n Phase 1 Complete.")
