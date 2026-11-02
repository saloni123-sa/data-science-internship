# Week2_EDA_Report - Virtual Data Science with Python Internship (Saloni Kumari)
# Run with: python Week2_EDA_Report.py   (needs numpy, pandas, matplotlib, seaborn, scikit-learn, scipy)
import os; os.makedirs('figures', exist_ok=True)

import pandas as pd, numpy as np
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.datasets import load_diabetes
sns.set_theme(style='whitegrid', context='notebook')

raw = load_diabetes(as_frame=True, scaled=False)      # scaled=False gives the original units
df = raw.frame.rename(columns={'target': 'progression'})
df['sex'] = df['sex'].map({1.0: 'Group 1', 2.0: 'Group 2'})   # the source does not say which is male/female
print(df.shape)
print(df.head())

print(df.info())
print()
print("Missing values:", int(df.isna().sum().sum()), "| Duplicates:", int(df.duplicated().sum()))

print(df.describe().T.round(2))

print(df['sex'].value_counts())
print()
print(df.groupby('sex')[['age', 'bmi', 'bp', 'progression']].mean().round(1))

fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
sns.histplot(df['progression'], kde=True, bins=25, color='#4f81bd', ax=ax[0])
ax[0].axvline(df['progression'].mean(), color='red', ls='--', label=f"Mean = {df['progression'].mean():.0f}")
ax[0].axvline(df['progression'].median(), color='green', ls=':', label=f"Median = {df['progression'].median():.0f}")
ax[0].set_title('Distribution of disease progression'); ax[0].set_xlabel('Progression score'); ax[0].set_ylabel('Number of patients'); ax[0].legend()
sns.boxplot(x=df['progression'], color='#9bbb59', ax=ax[1])
ax[1].set_title('Boxplot of disease progression'); ax[1].set_xlabel('Progression score')
plt.tight_layout(); plt.savefig('figures/w2_target.png', dpi=150); plt.close()
print("Skewness of progression: %.2f" % df['progression'].skew())

num = ['age', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
fig, axes = plt.subplots(3, 3, figsize=(10, 8))
for a, c in zip(axes.ravel(), num):
    sns.histplot(df[c], kde=True, ax=a, color='#4bacc6'); a.set_title(f'{c} (skew {df[c].skew():.2f})'); a.set_xlabel(c); a.set_ylabel('Count')
plt.suptitle('Distribution of predictor variables', y=1.01)
plt.tight_layout(); plt.savefig('figures/w2_hist.png', dpi=150, bbox_inches='tight'); plt.close()
print(df[num].skew().round(2).sort_values(ascending=False))

corr = df.drop(columns='sex').corr()
fig, ax = plt.subplots(figsize=(8, 6.4))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0, vmin=-1, vmax=1, ax=ax, annot_kws={'size': 8})
ax.set_title('Correlation matrix (Pearson)')
plt.tight_layout(); plt.savefig('figures/w2_corr.png', dpi=150); plt.close()
print("Correlation of each variable with progression:")
print(corr['progression'].drop('progression').sort_values(ascending=False).round(3))
print()
print("Strongest predictor-predictor pairs:")
pairs = corr.drop(index='progression', columns='progression').where(np.triu(np.ones((9, 9)), 1).astype(bool)).stack().sort_values(key=abs, ascending=False)
print(pairs.head(4).round(3))

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
sns.scatterplot(data=df, x='bmi', y='progression', hue='sex', alpha=.7, ax=ax[0])
sns.regplot(data=df, x='bmi', y='progression', scatter=False, color='black', ax=ax[0])
ax[0].set_title('BMI vs progression'); ax[0].set_xlabel('BMI'); ax[0].set_ylabel('Progression'); ax[0].legend(title='Sex group')
sns.scatterplot(data=df, x='s5', y='progression', hue='sex', alpha=.7, ax=ax[1])
sns.regplot(data=df, x='s5', y='progression', scatter=False, color='black', ax=ax[1])
ax[1].set_title('Log triglycerides (s5) vs progression'); ax[1].set_xlabel('s5 (log serum triglycerides)'); ax[1].set_ylabel('Progression'); ax[1].legend(title='Sex group')
plt.tight_layout(); plt.savefig('figures/w2_scatter.png', dpi=150); plt.close()

df['bmi_cat'] = pd.cut(df['bmi'], bins=[0, 18.5, 25, 30, 100], labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
df['age_band'] = pd.cut(df['age'], bins=[18, 30, 40, 50, 60, 80], labels=['19-30', '31-40', '41-50', '51-60', '61-79'])
tab = df.groupby('bmi_cat', observed=True)['progression'].agg(['count', 'mean', 'median']).round(1)
print(tab)
print()
print(df.groupby('age_band', observed=True)['progression'].agg(['count', 'mean']).round(1))
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
sns.boxplot(data=df, x='bmi_cat', y='progression', hue='bmi_cat', legend=False, palette='YlOrRd', ax=ax[0])
ax[0].set_title('Progression by BMI category'); ax[0].set_xlabel('BMI category'); ax[0].set_ylabel('Progression')
sns.barplot(data=df, x='age_band', y='progression', hue='sex', errorbar='sd', ax=ax[1])
ax[1].set_title('Mean progression by age band and sex group (bars = SD)'); ax[1].set_xlabel('Age band'); ax[1].set_ylabel('Mean progression'); ax[1].legend(title='Sex group')
plt.tight_layout(); plt.savefig('figures/w2_groups.png', dpi=150); plt.close()

cols = ['bmi', 'bp', 's5', 'progression']
g = sns.pairplot(df[cols + ['sex']], hue='sex', diag_kind='kde', plot_kws={'alpha': .5, 's': 18}, height=2.1)
g.fig.suptitle('Pair plot of the most relevant variables', y=1.02)
g.savefig('figures/w2_pair.png', dpi=130); plt.close('all')

def iqr_out(s):
    q1, q3 = s.quantile([.25, .75]); i = q3 - q1
    return ((s < q1 - 1.5 * i) | (s > q3 + 1.5 * i)).sum()
print("IQR outliers per variable:"); print(df[num + ['progression']].apply(iqr_out).sort_values(ascending=False).head(6))
