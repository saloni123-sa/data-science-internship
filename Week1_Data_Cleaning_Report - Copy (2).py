# Week1_Data_Cleaning_Report - Virtual Data Science with Python Internship (Saloni Kumari)
# Run with: python Week1_Data_Cleaning_Report.py   (needs numpy, pandas, matplotlib, seaborn, scikit-learn, scipy)
import os; os.makedirs('figures', exist_ok=True)

import pandas as pd, numpy as np
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.datasets import load_breast_cancer

raw = load_breast_cancer(as_frame=True)
df = raw.frame.copy()
df['diagnosis'] = df['target'].map({0: 'malignant', 1: 'benign'})
df = df.drop(columns='target')
print("Shape:", df.shape)
print(df.iloc[:3, [0, 1, 2, 3, -1]])

clean_truth = df.copy()           # ground truth, used later only to evaluate the imputation
rng = np.random.default_rng(42)
dirty = df.copy()

# (a) random missing values in six columns (~4%)
cols_missing = ['mean radius', 'mean texture', 'mean area', 'mean smoothness', 'worst perimeter', 'worst concavity']
for c in cols_missing:
    idx = rng.choice(dirty.index, size=int(0.04 * len(dirty)), replace=False)
    dirty.loc[idx, c] = np.nan

# (b) impossible / sentinel values
dirty.loc[rng.choice(dirty.index, 4, replace=False), 'mean area'] = 0
dirty.loc[rng.choice(dirty.index, 3, replace=False), 'mean radius'] = -999
dirty.loc[rng.choice(dirty.index, 3, replace=False), 'mean texture'] = -1

# (c) duplicate rows
dirty = pd.concat([dirty, dirty.sample(3, random_state=1)], ignore_index=True)

# (d) inconsistent text labels
dirty['diagnosis'] = dirty['diagnosis'].astype(object)
dirty.loc[dirty.sample(20, random_state=2).index, 'diagnosis'] = dirty['diagnosis'].str.upper()
dirty.loc[dirty.sample(15, random_state=3).index, 'diagnosis'] = ' ' + dirty['diagnosis'] + ' '
print("Working copy shape:", dirty.shape)

print(dirty.dtypes.value_counts())
print()
print("Duplicate rows:", dirty.duplicated().sum())
print()
print("Label values:", dirty['diagnosis'].unique())
print()
miss = dirty.isna().sum()
print("Missing values per column (only columns with missing data):")
print(miss[miss > 0])

print(dirty[['mean radius', 'mean texture', 'mean area', 'mean smoothness']].describe().round(2))

fig, ax = plt.subplots(figsize=(7, 3.6))
(miss[miss > 0] / len(dirty) * 100).sort_values().plot.barh(ax=ax, color='#c0504d')
ax.set_title('Percentage of missing values per column (before cleaning)')
ax.set_xlabel('Missing values (%)'); ax.set_ylabel('Column')
plt.tight_layout(); plt.savefig('figures/w1_missing.png', dpi=150); plt.close()

df1 = dirty.copy()
df1['diagnosis'] = df1['diagnosis'].str.strip().str.lower()
print(df1['diagnosis'].value_counts())

before = len(df1)
df1 = df1.drop_duplicates().reset_index(drop=True)
print(f"Rows before: {before}, after: {len(df1)}, removed: {before - len(df1)}")

positive_cols = ['mean radius', 'mean texture', 'mean area']
bad = (df1[positive_cols] <= 0).sum()
print("Impossible values found:"); print(bad)
df1[positive_cols] = df1[positive_cols].mask(df1[positive_cols] <= 0)
print("\nTotal missing values now:", int(df1.isna().sum().sum()))

from sklearn.impute import SimpleImputer, KNNImputer

num_cols = df1.select_dtypes('number').columns
mask = df1[num_cols].isna()
# align the ground truth with the working rows (drop duplicates in the same way)
truth = clean_truth[num_cols].reset_index(drop=True)
# the first 569 rows of df1 correspond to the original rows in the original order
truth = truth.iloc[:len(df1)].reindex(df1.index)

results = {}
imputers = {'Mean': SimpleImputer(strategy='mean'),
            'Median': SimpleImputer(strategy='median'),
            'KNN (k=5)': KNNImputer(n_neighbors=5)}
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(df1[num_cols].fillna(df1[num_cols].median()))
for name, imp in imputers.items():
    if name.startswith('KNN'):
        # KNN works on distances so I scale first, then un-scale
        Xs = scaler.transform(df1[num_cols])
        filled = pd.DataFrame(scaler.inverse_transform(imp.fit_transform(Xs)), columns=num_cols, index=df1.index)
    else:
        filled = pd.DataFrame(imp.fit_transform(df1[num_cols]), columns=num_cols, index=df1.index)
    err = (filled[mask] - truth[mask]).abs() / truth[mask].abs()
    results[name] = np.nanmean(err.values) * 100
print("Mean absolute percentage error of the imputed cells:")
for k, v in results.items(): print(f"  {k:10s}: {v:5.2f}%")

Xs = scaler.transform(df1[num_cols])
df2 = df1.copy()
df2[num_cols] = scaler.inverse_transform(KNNImputer(n_neighbors=5).fit_transform(Xs))
print("Missing values after imputation:", int(df2.isna().sum().sum()))

def iqr_outliers(s):
    q1, q3 = s.quantile([.25, .75]); i = q3 - q1
    return ((s < q1 - 1.5 * i) | (s > q3 + 1.5 * i))

out_counts = df2[num_cols].apply(iqr_outliers).sum().sort_values(ascending=False)
print("Top 8 columns by number of IQR outliers:")
print(out_counts.head(8))
print("\nTotal rows with at least one outlier:", int(df2[num_cols].apply(iqr_outliers).any(axis=1).sum()), "of", len(df2))

fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
sns.boxplot(data=df2, x='diagnosis', y='mean area', ax=axes[0], palette=['#c0504d', '#4f81bd'], hue='diagnosis', legend=False)
axes[0].set_title('mean area by diagnosis (before capping)'); axes[0].set_xlabel('Diagnosis'); axes[0].set_ylabel('Mean area')
low, high = df2[num_cols].quantile(0.01), df2[num_cols].quantile(0.99)
df3 = df2.copy(); df3[num_cols] = df2[num_cols].clip(low, high, axis=1)
sns.boxplot(data=df3, x='diagnosis', y='mean area', ax=axes[1], palette=['#c0504d', '#4f81bd'], hue='diagnosis', legend=False)
axes[1].set_title('mean area by diagnosis (after capping)'); axes[1].set_xlabel('Diagnosis'); axes[1].set_ylabel('Mean area')
plt.tight_layout(); plt.savefig('figures/w1_outliers.png', dpi=150); plt.close()
print("Max mean area before:", round(df2['mean area'].max(), 1), "| after capping:", round(df3['mean area'].max(), 1))

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = df3[num_cols]
y = (df3['diagnosis'] == 'malignant').astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

sc = StandardScaler().fit(X_train)
X_train_s = pd.DataFrame(sc.transform(X_train), columns=num_cols, index=X_train.index)
X_test_s = pd.DataFrame(sc.transform(X_test), columns=num_cols, index=X_test.index)
print("Train:", X_train_s.shape, "Test:", X_test_s.shape)
print("Malignant share - train: %.3f, test: %.3f" % (y_train.mean(), y_test.mean()))
print("Scaled train mean (first 3 cols):", X_train_s.iloc[:, :3].mean().round(3).tolist())
print("Scaled train std  (first 3 cols):", X_train_s.iloc[:, :3].std().round(3).tolist())

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score

# Crude approach: drop every row with NaN or impossible values, no scaling
crude = dirty.copy()
crude['diagnosis'] = crude['diagnosis'].str.strip().str.lower()
crude[positive_cols] = crude[positive_cols].mask(crude[positive_cols] <= 0)
crude = crude.dropna().drop_duplicates()
Xc, yc = crude[num_cols], (crude['diagnosis'] == 'malignant').astype(int)
print("Rows kept by crude row deletion:", len(crude), "of", len(dirty))

# Same clean held-out test rows for both models (taken from ground truth by index)
test_idx = X_test.index
Xt_truth = clean_truth.loc[test_idx, num_cols]
sc_truth = StandardScaler().fit(X_train)   # scaler fitted on the cleaned training data
m1 = LogisticRegression(max_iter=5000).fit(Xc[~Xc.index.isin(test_idx)], yc[~yc.index.isin(test_idx)])
m2 = LogisticRegression(max_iter=5000).fit(X_train_s, y_train)
for name, m, Xt in [('Crude (rows dropped, unscaled)', m1, Xt_truth),
                    ('Full preprocessing', m2, pd.DataFrame(sc_truth.transform(Xt_truth), columns=num_cols))]:
    pred = m.predict(Xt)
    print(f"{name:32s} accuracy={accuracy_score(y_test, pred):.3f}  malignant recall={recall_score(y_test, pred):.3f}")

final = df3.copy()
final.to_csv('breast_cancer_cleaned.csv', index=False)
print("Final shape:", final.shape)
print("Missing values:", int(final.isna().sum().sum()), "| Duplicates:", int(final.duplicated().sum()))
print(final['diagnosis'].value_counts().to_dict())
