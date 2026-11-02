# Week3_Clustering_Report - Virtual Data Science with Python Internship (Saloni Kumari)
# Run with: python Week3_Clustering_Report.py   (needs numpy, pandas, matplotlib, seaborn, scikit-learn, scipy)
import os; os.makedirs('figures', exist_ok=True)

import pandas as pd, numpy as np
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.datasets import load_wine
sns.set_theme(style='whitegrid')

w = load_wine(as_frame=True)
X = w.data.copy()
y_true = w.target            # NOT used for clustering, only for validation at the end
print("Shape:", X.shape, "| Missing values:", int(X.isna().sum().sum()), "| Duplicates:", int(X.duplicated().sum()))
print(X.describe().T[['mean', 'std', 'min', 'max']].round(2))

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
Xs = scaler.fit_transform(X)
print("After scaling -> mean (max abs):", np.abs(Xs.mean(axis=0)).max().round(6), "| std (all):", Xs.std(axis=0).round(2)[:3], "...")

from sklearn.decomposition import PCA
pca = PCA(n_components=2, random_state=42)
Xp = pca.fit_transform(Xs)
print("Variance explained by PC1, PC2:", pca.explained_variance_ratio_.round(3), "| total:", pca.explained_variance_ratio_.sum().round(3))

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

ks = range(2, 9)
rows = []
for k in ks:
    km = KMeans(n_clusters=k, n_init=20, random_state=42).fit(Xs)
    rows.append((k, km.inertia_, silhouette_score(Xs, km.labels_), davies_bouldin_score(Xs, km.labels_)))
sel = pd.DataFrame(rows, columns=['k', 'inertia', 'silhouette', 'davies_bouldin']).set_index('k')
print(sel.round(3))

fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
ax[0].plot(sel.index, sel['inertia'], 'o-', color='#4f81bd'); ax[0].set_title('Elbow method'); ax[0].set_xlabel('Number of clusters k'); ax[0].set_ylabel('Inertia (within-cluster SS)')
ax[1].plot(sel.index, sel['silhouette'], 's-', color='#c0504d'); ax[1].set_title('Silhouette score'); ax[1].set_xlabel('Number of clusters k'); ax[1].set_ylabel('Mean silhouette')
plt.tight_layout(); plt.savefig('figures/w3_select.png', dpi=150); plt.close()

km = KMeans(n_clusters=3, n_init=20, random_state=42).fit(Xs)
labels = pd.Series(km.labels_, index=X.index, name='cluster')
print("Cluster sizes:", labels.value_counts().sort_index().to_dict())
print("Silhouette: %.3f | Davies-Bouldin: %.3f" % (silhouette_score(Xs, labels), davies_bouldin_score(Xs, labels)))

fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
pal = sns.color_palette('Set1', 3)
for c in range(3):
    m = labels == c
    ax[0].scatter(Xp[m, 0], Xp[m, 1], s=35, color=pal[c], label=f'Cluster {c} (n={m.sum()})', alpha=.8, edgecolor='white', linewidth=.4)
cent = pca.transform(km.cluster_centers_)
ax[0].scatter(cent[:, 0], cent[:, 1], marker='X', s=220, color='black', label='Centroids')
ax[0].set_title('K-Means clusters in PCA space'); ax[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.0%} variance)'); ax[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.0%} variance)'); ax[0].legend(fontsize=8)

from sklearn.metrics import silhouette_samples
sil = silhouette_samples(Xs, labels); y0 = 10
for c in range(3):
    s_c = np.sort(sil[labels == c]); ax[1].fill_betweenx(np.arange(y0, y0 + len(s_c)), 0, s_c, color=pal[c], alpha=.85); ax[1].text(-0.08, y0 + len(s_c) / 2, f'C{c}'); y0 += len(s_c) + 10
ax[1].axvline(sil.mean(), color='red', ls='--', label=f'Mean = {sil.mean():.2f}'); ax[1].set_title('Silhouette plot'); ax[1].set_xlabel('Silhouette coefficient'); ax[1].set_ylabel('Wines (sorted within cluster)'); ax[1].legend()
plt.tight_layout(); plt.savefig('figures/w3_kmeans.png', dpi=150); plt.close()

from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.metrics import adjusted_rand_score
Z = linkage(Xs, method='ward')
hier = pd.Series(fcluster(Z, 3, criterion='maxclust') - 1, index=X.index)
fig, ax = plt.subplots(figsize=(9, 3.8))
dendrogram(Z, truncate_mode='lastp', p=20, leaf_rotation=90, leaf_font_size=8, show_contracted=True, ax=ax, color_threshold=Z[-3, 2] + 1e-6)
ax.set_title('Dendrogram (Ward linkage, last 20 merges)'); ax.set_xlabel('Cluster (number of wines in brackets)'); ax.set_ylabel('Ward distance')
plt.tight_layout(); plt.savefig('figures/w3_dendro.png', dpi=150); plt.close()
print("Hierarchical cluster sizes:", hier.value_counts().sort_index().to_dict())
print("Agreement between K-Means and Ward (Adjusted Rand Index): %.3f" % adjusted_rand_score(labels, hier))

prof = X.groupby(labels).mean().T.round(2)
prof.columns = [f'Cluster {c}' for c in prof.columns]
print(prof)
z = pd.DataFrame(Xs, columns=X.columns, index=X.index).groupby(labels).mean().T
fig, ax = plt.subplots(figsize=(6.5, 5.2))
sns.heatmap(z, annot=True, fmt='.2f', cmap='RdBu_r', center=0, ax=ax, annot_kws={'size': 8}, xticklabels=[f'Cluster {c}' for c in z.columns], cbar_kws={'label': 'Std. deviations from overall mean'})
ax.set_title('Cluster profile (z-scores of cluster means)'); ax.set_ylabel('Feature')
plt.tight_layout(); plt.savefig('figures/w3_profile.png', dpi=150); plt.close()

from sklearn.metrics import adjusted_rand_score, confusion_matrix
print("ARI K-Means (scaled):    %.3f" % adjusted_rand_score(y_true, labels))
print("ARI Ward   (scaled):     %.3f" % adjusted_rand_score(y_true, hier))
km_raw = KMeans(n_clusters=3, n_init=20, random_state=42).fit(X)
print("ARI K-Means (UNscaled):  %.3f" % adjusted_rand_score(y_true, km_raw.labels_))
print()
print("Cluster (rows) vs true cultivar (columns), K-Means scaled:")
print(pd.crosstab(labels, y_true, rownames=['cluster'], colnames=['cultivar']))
