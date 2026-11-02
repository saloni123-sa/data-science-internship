# Week6_Capstone_Report - Virtual Data Science with Python Internship (Saloni Kumari)
# Run with: python Week6_Capstone_Report.py   (needs numpy, pandas, matplotlib, seaborn, scikit-learn, scipy)
import os; os.makedirs('figures', exist_ok=True)

import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from sklearn.datasets import load_diabetes
sns.set_theme(style='whitegrid')

raw = load_diabetes(as_frame=True, scaled=False)
df = raw.frame.rename(columns={'target': 'progression'})
features = [c for c in df.columns if c != 'progression']
print("Shape:", df.shape)
print(df.head(3).round(2))

print("Missing values:", int(df.isna().sum().sum()), "| Duplicate rows:", int(df.duplicated().sum()))
print("Data types:", df.dtypes.unique().tolist())
print("sex values:", sorted(df['sex'].unique()))
# plausibility checks (physical ranges)
checks = {'age 18-100': df['age'].between(18, 100).all(), 'bmi 10-60': df['bmi'].between(10, 60).all(), 'bp 40-200': df['bp'].between(40, 200).all(),
          'all s-values > 0': (df[['s1', 's2', 's3', 's4', 's5', 's6']] > 0).all().all(), 'progression > 0': (df['progression'] > 0).all()}
print(checks)
def iqr_count(s):
    q1, q3 = s.quantile([.25, .75]); i = q3 - q1
    return int(((s < q1 - 1.5 * i) | (s > q3 + 1.5 * i)).sum())
print("IQR outliers:", {c: iqr_count(df[c]) for c in features + ['progression'] if iqr_count(df[c]) > 0})

corr = df.corr()
fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0, vmin=-1, vmax=1, ax=ax[0], annot_kws={'size': 7}, cbar=False)
ax[0].set_title('Correlation matrix')
sns.histplot(df['progression'], kde=True, bins=25, ax=ax[1], color='#4f81bd'); ax[1].set_title('Distribution of progression (target)'); ax[1].set_xlabel('Progression score'); ax[1].set_ylabel('Patients')
plt.tight_layout(); plt.savefig('figures/w6_eda.png', dpi=150); plt.close()
print(corr['progression'].drop('progression').sort_values(ascending=False).round(2).to_dict())
print("Skewness of target: %.2f | mean %.1f | median %.1f" % (df['progression'].skew(), df['progression'].mean(), df['progression'].median()))
print("Highly correlated predictor pairs (|r| > 0.7):", [(a, b, round(corr.loc[a, b], 2)) for i, a in enumerate(features) for b in features[i + 1:] if abs(corr.loc[a, b]) > 0.7])

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

df['sex_f'] = (df['sex'] == 2).astype(int)           # numeric 0/1 version of the two coded groups
seg_feats = ['age', 'sex_f', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
Z = StandardScaler().fit_transform(df[seg_feats])
for k in range(2, 7):
    km = KMeans(n_clusters=k, n_init=20, random_state=42).fit(Z)
    print(f"k={k}: silhouette={silhouette_score(Z, km.labels_):.3f}  inertia={km.inertia_:.0f}")

K = 2   # the silhouette score is highest at k = 2 (see above), so I follow the stated criterion
km = KMeans(n_clusters=K, n_init=20, random_state=42).fit(Z)
df['segment'] = km.labels_
pca = PCA(2, random_state=42); P = pca.fit_transform(Z)
prof = df.groupby('segment')[['age', 'sex_f', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6', 'progression']].mean().round(2)
prof.insert(0, 'n', df['segment'].value_counts().sort_index())
print(prof.T)
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2)); pal = sns.color_palette('Set1', K)
for c in range(K):
    m = df['segment'] == c; ax[0].scatter(P[m, 0], P[m, 1], s=22, color=pal[c], alpha=.75, label=f'Segment {c} (n={m.sum()})')
ax[0].set_title('Patient segments in PCA space'); ax[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.0%})'); ax[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.0%})'); ax[0].legend(fontsize=8)
sns.boxplot(data=df, x='segment', y='progression', hue='segment', palette=pal, legend=False, ax=ax[1])
ax[1].set_title('Progression by segment (not used for clustering)'); ax[1].set_xlabel('Segment'); ax[1].set_ylabel('Progression score')
plt.tight_layout(); plt.savefig('figures/w6_seg.png', dpi=150); plt.close()
from scipy.stats import kruskal
print("Kruskal-Wallis test of progression across segments: H=%.1f, p=%.2g" % kruskal(*[g['progression'] for _, g in df.groupby('segment')]))

from sklearn.model_selection import train_test_split, KFold, cross_validate, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

Xr = df[seg_feats]; yr = df['progression']
X_tr, X_te, y_tr, y_te = train_test_split(Xr, yr, test_size=0.2, random_state=42)
cv = KFold(5, shuffle=True, random_state=42)
models = {'Baseline (mean)': DummyRegressor(),
          'Linear Regression': make_pipeline(StandardScaler(), LinearRegression()),
          'Ridge': make_pipeline(StandardScaler(), Ridge(alpha=10)),
          'Lasso': make_pipeline(StandardScaler(), Lasso(alpha=1.0)),
          'Random Forest': RandomForestRegressor(300, min_samples_leaf=5, random_state=42),
          'Gradient Boosting': GradientBoostingRegressor(n_estimators=150, learning_rate=0.05, max_depth=2, subsample=0.8, random_state=42)}
rows = []
for n, m in models.items():
    r = cross_validate(m, X_tr, y_tr, cv=cv, scoring=['neg_root_mean_squared_error', 'neg_mean_absolute_error', 'r2'])
    rows.append({'model': n, 'RMSE': -r['test_neg_root_mean_squared_error'].mean(), 'MAE': -r['test_neg_mean_absolute_error'].mean(), 'R2': r['test_r2'].mean(), 'R2_std': r['test_r2'].std()})
cvres = pd.DataFrame(rows).set_index('model'); print(cvres.round(3))

grid = GridSearchCV(make_pipeline(StandardScaler(), Ridge()), {'ridge__alpha': [0.1, 1, 3, 10, 30, 100]}, cv=cv, scoring='neg_root_mean_squared_error').fit(X_tr, y_tr)
gb_grid = GridSearchCV(GradientBoostingRegressor(random_state=42, subsample=0.8), {'n_estimators': [100, 200], 'learning_rate': [0.03, 0.05], 'max_depth': [1, 2, 3]}, cv=cv, scoring='neg_root_mean_squared_error').fit(X_tr, y_tr)
print("Ridge best alpha:", grid.best_params_, "| CV RMSE: %.2f" % -grid.best_score_)
print("GB best params:  ", gb_grid.best_params_, "| CV RMSE: %.2f" % -gb_grid.best_score_)

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
best = grid.best_estimator_; gbm = gb_grid.best_estimator_; base = DummyRegressor().fit(X_tr, y_tr)
rows = []
for n, m in [('Baseline (mean)', base), ('Ridge (tuned)', best), ('Gradient Boosting (tuned)', gbm)]:
    p_ = m.predict(X_te); rows.append({'model': n, 'RMSE': np.sqrt(mean_squared_error(y_te, p_)), 'MAE': mean_absolute_error(y_te, p_), 'R2': r2_score(y_te, p_)})
test_res = pd.DataFrame(rows).set_index('model'); print(test_res.round(3))

pred = best.predict(X_te); resid = y_te - pred
fig, ax = plt.subplots(1, 3, figsize=(13, 4))
ax[0].scatter(y_te, pred, alpha=.7, color='#4f81bd'); lim = [20, 350]; ax[0].plot(lim, lim, 'r--', label='Perfect prediction')
ax[0].set_title('Predicted vs actual (Ridge, test set)'); ax[0].set_xlabel('Actual progression'); ax[0].set_ylabel('Predicted progression'); ax[0].legend()
ax[1].scatter(pred, resid, alpha=.7, color='#9bbb59'); ax[1].axhline(0, color='r', ls='--'); ax[1].set_title('Residuals vs predicted'); ax[1].set_xlabel('Predicted progression'); ax[1].set_ylabel('Residual (actual - predicted)')
sns.histplot(resid, kde=True, ax=ax[2], color='#c0504d'); ax[2].set_title('Residual distribution'); ax[2].set_xlabel('Residual'); ax[2].set_ylabel('Patients')
plt.tight_layout(); plt.savefig('figures/w6_eval.png', dpi=150); plt.close()
print("Mean residual: %.2f | residual std: %.2f" % (resid.mean(), resid.std()))

from sklearn.inspection import permutation_importance
pi = permutation_importance(best, X_te, y_te, n_repeats=30, random_state=42, scoring='r2')
imp = pd.DataFrame({'importance': pi.importances_mean, 'std': pi.importances_std}, index=seg_feats).sort_values('importance')
fig, ax = plt.subplots(figsize=(6.5, 3.8)); ax.barh(imp.index, imp['importance'], xerr=imp['std'], color='#4bacc6')
ax.set_title('Permutation importance (drop in test R2)'); ax.set_xlabel('Decrease in R2 when the feature is shuffled'); ax.set_ylabel('Feature')
plt.tight_layout(); plt.savefig('figures/w6_imp.png', dpi=150); plt.close()
print(imp.sort_values('importance', ascending=False).round(3))

seg_dummies = pd.get_dummies(df['segment'], prefix='seg').astype(int)
Xr2 = pd.concat([Xr, seg_dummies], axis=1)
Xa, Xb, ya, yb = train_test_split(Xr2, yr, test_size=0.2, random_state=42)
m_with = make_pipeline(StandardScaler(), Ridge(alpha=grid.best_params_['ridge__alpha']))
r_with = cross_validate(m_with, Xa, ya, cv=cv, scoring=['neg_root_mean_squared_error'])
r_without = cross_validate(make_pipeline(StandardScaler(), Ridge(alpha=grid.best_params_['ridge__alpha'])), X_tr, y_tr, cv=cv, scoring=['neg_root_mean_squared_error'])
print("CV RMSE without segment: %.2f | with segment as extra feature: %.2f" % (-r_without['test_neg_root_mean_squared_error'].mean(), -r_with['test_neg_root_mean_squared_error'].mean()))
