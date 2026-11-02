# Week4_Supervised_Learning_Report - Virtual Data Science with Python Internship (Saloni Kumari)
# Run with: python Week4_Supervised_Learning_Report.py   (needs numpy, pandas, matplotlib, seaborn, scikit-learn, scipy)
import os; os.makedirs('figures', exist_ok=True)

import warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.datasets import load_breast_cancer
sns.set_theme(style='whitegrid')

data = load_breast_cancer(as_frame=True)
X = data.data.copy()
y = (1 - data.target).rename('malignant')     # in scikit-learn 0 = malignant, so I flip it: 1 = malignant (positive class)
print("Samples:", X.shape[0], "| Features:", X.shape[1], "| Missing:", int(X.isna().sum().sum()))
print("Class balance:", y.value_counts().rename({0: 'benign', 1: 'malignant'}).to_dict(), "| malignant share: %.1f%%" % (100 * y.mean()))

from sklearn.model_selection import train_test_split, StratifiedKFold
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
print("Train:", X_train.shape, "| Test:", X_test.shape)

corr = X_train.corr().abs()
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
drop_cols = [c for c in upper.columns if (upper[c] > 0.95).any()]
print(f"Dropping {len(drop_cols)} features with |r| > 0.95:", drop_cols)

def make_B(d): return d.drop(columns=drop_cols)
def make_C(d):
    d = make_B(d).copy()
    d['area_per_perimeter_sq'] = X.loc[d.index, 'mean area'] / X.loc[d.index, 'mean perimeter'] ** 2
    d['concavity_per_radius'] = X.loc[d.index, 'mean concavity'] / X.loc[d.index, 'mean radius']
    d['texture_per_smoothness'] = X.loc[d.index, 'mean texture'] / X.loc[d.index, 'mean smoothness']
    return d
sets = {'A: all 30': X_train, 'B: drop correlated': make_B(X_train), 'C: B + engineered': make_C(X_train)}
print({k: v.shape[1] for k, v in sets.items()})

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_validate

models = {
    'Logistic Regression': make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
    'SVM (RBF)': make_pipeline(StandardScaler(), SVC(probability=True, random_state=42)),
    'Random Forest': RandomForestClassifier(n_estimators=300, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42)}
scoring = ['accuracy', 'recall', 'precision', 'f1', 'roc_auc']
rows = []
for sname, Xs_ in sets.items():
    for mname, m in models.items():
        r = cross_validate(m, Xs_, y_train, cv=cv, scoring=scoring)
        rows.append({'features': sname, 'model': mname, **{k: r['test_' + k].mean() for k in scoring}, 'f1_std': r['test_f1'].std()})
cvres = pd.DataFrame(rows)
print(cvres.round(3).to_string(index=False))

best_view = cvres.pivot(index='model', columns='features', values='f1').round(3)
print("Mean cross-validated F1 by model and feature set:"); print(best_view)
fig, ax = plt.subplots(figsize=(8, 3.8))
best_view.plot.bar(ax=ax, rot=15, colormap='Set2')
ax.set_ylim(0.9, 1.0); ax.set_title('Cross-validated F1 (malignant class) by model and feature set'); ax.set_ylabel('Mean F1 (5-fold)'); ax.set_xlabel('Model'); ax.legend(title='Feature set', fontsize=8)
plt.tight_layout(); plt.savefig('figures/w4_cv.png', dpi=150); plt.close()

from sklearn.model_selection import GridSearchCV
XtrC = make_C(X_train); XteC = make_C(X_test)
XtrA = X_train; XteA = X_test
grid_lr = GridSearchCV(make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
                       {'logisticregression__C': [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]}, cv=cv, scoring='f1')
grid_svm = GridSearchCV(make_pipeline(StandardScaler(), SVC(probability=True, random_state=42)),
                        {'svc__C': [0.3, 1, 3, 10], 'svc__gamma': ['scale', 0.01, 0.03, 0.1]}, cv=cv, scoring='f1')
grid_lr.fit(XtrA, y_train); grid_svm.fit(XtrA, y_train)
print("Logistic Regression best params:", grid_lr.best_params_, "| CV F1: %.3f" % grid_lr.best_score_)
print("SVM best params:               ", grid_svm.best_params_, "| CV F1: %.3f" % grid_svm.best_score_)

from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
                             confusion_matrix, classification_report, roc_curve, precision_recall_curve)
final = grid_lr.best_estimator_
proba = final.predict_proba(XteA)[:, 1]
pred = (proba >= 0.5).astype(int)
print("Test accuracy : %.3f" % accuracy_score(y_test, pred))
print("Precision     : %.3f" % precision_score(y_test, pred))
print("Recall        : %.3f" % recall_score(y_test, pred))
print("F1            : %.3f" % f1_score(y_test, pred))
print("ROC-AUC       : %.3f" % roc_auc_score(y_test, proba))
print()
print(classification_report(y_test, pred, target_names=['benign', 'malignant']))
cm = confusion_matrix(y_test, pred)
print("Confusion matrix [rows = true benign, malignant; cols = predicted]:"); print(cm)

fig, ax = plt.subplots(1, 3, figsize=(13, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax[0], xticklabels=['benign', 'malignant'], yticklabels=['benign', 'malignant'])
ax[0].set_title('Confusion matrix (test set)'); ax[0].set_xlabel('Predicted'); ax[0].set_ylabel('True')
fpr, tpr, _ = roc_curve(y_test, proba)
ax[1].plot(fpr, tpr, color='#c0504d', label=f'AUC = {roc_auc_score(y_test, proba):.3f}'); ax[1].plot([0, 1], [0, 1], 'k--', label='Chance')
ax[1].set_title('ROC curve'); ax[1].set_xlabel('False positive rate'); ax[1].set_ylabel('True positive rate (recall)'); ax[1].legend()
pr, rc, _ = precision_recall_curve(y_test, proba)
ax[2].plot(rc, pr, color='#4f81bd'); ax[2].set_title('Precision-Recall curve'); ax[2].set_xlabel('Recall'); ax[2].set_ylabel('Precision'); ax[2].set_ylim(0.5, 1.02)
plt.tight_layout(); plt.savefig('figures/w4_eval.png', dpi=150); plt.close()

svm_best = grid_svm.best_estimator_
p_svm = svm_best.predict_proba(XteA)[:, 1]; pr_svm = (p_svm >= 0.5).astype(int)
print("For transparency, the tuned SVM on the same test set:")
print("accuracy %.3f | recall %.3f | precision %.3f | F1 %.3f | ROC-AUC %.3f" % (accuracy_score(y_test, pr_svm), recall_score(y_test, pr_svm), precision_score(y_test, pr_svm), f1_score(y_test, pr_svm), roc_auc_score(y_test, p_svm)))
print("SVM confusion matrix:", confusion_matrix(y_test, pr_svm).tolist())

rows = []
for t in [0.5, 0.4, 0.3, 0.2, 0.1]:
    p_ = (proba >= t).astype(int); tn, fp, fn, tp = confusion_matrix(y_test, p_).ravel()
    rows.append({'threshold': t, 'missed_malignant (FN)': fn, 'false_alarms (FP)': fp, 'recall': round(tp / (tp + fn), 3), 'precision': round(tp / (tp + fp), 3)})
print(pd.DataFrame(rows).to_string(index=False))

coef = pd.Series(final.named_steps['logisticregression'].coef_[0], index=XtrA.columns).sort_values()
top = pd.concat([coef.head(6), coef.tail(6)])
fig, ax = plt.subplots(figsize=(7.5, 4.6))
ax.barh(top.index, top.values, color=['#4f81bd' if v < 0 else '#c0504d' for v in top.values])
ax.set_title('Largest logistic regression coefficients (standardised features)'); ax.set_xlabel('Coefficient (positive = pushes toward malignant)'); ax.set_ylabel('Feature')
plt.tight_layout(); plt.savefig('figures/w4_coef.png', dpi=150); plt.close()
print(top.round(2))
