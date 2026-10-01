# Customer Segmentation with K-Means & DBSCAN

Unsupervised segmentation of retail customers from a marketing-campaign dataset.
The pipeline cleans the raw data, engineers a small set of behavioural features,
encodes and scales them, then compares two clustering algorithms (K-Means and
DBSCAN) using silhouette score, and profiles the resulting segments.

---

## Dataset

`marketing_campaign.csv` — the Kaggle *Customer Personality Analysis* dataset.

- **Separator:** tab (`sep='\t'`), not comma.
- **Rows:** 2240 raw customers.
- **Columns:** demographics (birth year, education, marital status, income,
  children), two years of spending across six product categories, purchase
  channels, and campaign-response flags.

Place the file next to the script, or update the path at the top:

```python
Data = pd.read_csv('marketing_campaign.csv', sep='\t')
```

---

## Pipeline

### 1. Cleaning
- `Income` has missing values → filled with the column mean.
- Spending columns renamed for readability:
  `MntWines → Wines`, `MntFruits → Fruits`, `MntMeatProducts → Meat`,
  `MntFishProducts → Fish`, `MntSweetProducts → Sweets`, `MntGoldProds → Gold`.

### 2. Feature engineering
| Feature | Definition |
|---|---|
| `Kids`  | `Kidhome + Teenhome` — total children in the household |
| `Age`   | `2026 - Year_Birth` |
| `Spent` | `Wines + Fruits + Meat + Fish + Sweets + Gold` — total spend |

### 3. Outlier removal
- `Age < 60` — drops the handful of records with birth years in the 1890s that
  produce impossible ages.
- `Income < 600000` — drops the single extreme income outlier.

Index is reset afterwards so the one-hot frame concatenates cleanly.

### 4. Dropped columns
IDs, campaign-response flags, and constant columns carry no segmentation signal:

```
ID, Recency, AcceptedCmp1..5, Z_CostContact, Z_Revenue,
Year_Birth, Dt_Customer, Kidhome, Teenhome
```

### 5. Encoding & scaling
- `Marital_Status` and `Education` → `OneHotEncoder(sparse_output=False)`.
- Numeric block: `Age`, `Income`, `Spent`, `Kids`.
- Everything concatenated, then `StandardScaler` — mandatory here, since both
  K-Means and DBSCAN are distance-based and `Income` would otherwise dominate.

### 6. PCA
Two separate PCA objects, for two different jobs:
- `PCA(n_components=2)` → 2-D coordinates used **only** for scatter plots.
- `PCA(n_components=0.95)` → enough components to retain 95 % of variance,
  available as a denoised feature space.

Component loadings are printed so each axis can be interpreted in terms of the
original features.

---

## Models

### K-Means
- Grid over `n_clusters` 2–10 × `n_init` 10–19, scored by silhouette.
- Elbow curve (inertia vs. *k*) saved as `Elbow.png`.
- **Final model:** `KMeans(n_clusters=9, n_init=10, random_state=42)`.
- Cluster profiles produced with `groupby('Clusters').mean()`.

### DBSCAN
- Grid over `eps ∈ [1.6 … 2.3]` × `min_samples ∈ {3, 5, 8, 10}`, recording
  cluster count, silhouette, and noise-point count for each combination.
- **Final model:** `DBSCAN(eps=1.9, min_samples=3)`.

Density-based clustering is a poor fit for high-dimensional one-hot data — the
default `eps=0.5` labels almost everything as noise, which is why the grid
starts well above it.

---

## Outputs

| File | Contents |
|---|---|
| `Age---Count.png` | Age distribution |
| `Count---Spent.png` | Total-spend distribution |
| `Marital.png` | Marital-status counts |
| `Education.png` | Education counts |
| `Corr of Features.png` | Correlation heatmap |
| `Elbow.png` | Inertia vs. number of clusters |
| `K Clusters.png` | K-Means clusters in PCA space |
| `DBscan Clusters.png` | DBSCAN clusters in PCA space |
| `Scaler.pkl` | Fitted `StandardScaler` |
| `Columns.pkl` | Feature order expected at inference |
| `K_model.pkl`, `DB_model.pkl` | Saved cluster assignments |

---

## Requirements

```
numpy
pandas
matplotlib
seaborn
scikit-learn
```

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
python customer_segmentation.py
```

---

## Reading the results

- **Silhouette score** runs from −1 to 1. On mixed one-hot + continuous data,
  values in the 0.1–0.3 range are normal; the metric is useful for *ranking*
  configurations, less so as an absolute quality judgement.
- **Cluster means** are the real deliverable. Comparing `Income`, `Spent`,
  `Age`, and `Kids` across clusters is what turns anonymous group IDs into
  usable personas — high-income low-children heavy spenders, budget families
  with several children, young low-spend customers, and so on.
- **PCA loadings** explain what PC1 and PC2 represent, so the scatter plots can
  be read rather than just admired.

---

## Known limitations & next steps

1. **PCA is computed but not used for modelling.** Both models are fitted on
   `x_scaled`; `x_pca` is never passed to a clustering call. Either fit on
   `x_pca` or drop the 95 % PCA to keep the pipeline honest.
2. **The pickles store labels, not models.** `pickle.dump(best_k_label, ...)`
   saves an array of cluster assignments, so `K_model.pkl` cannot predict on
   new customers. Dump `best_kmean` and `best_db` instead. (DBSCAN has no
   `predict`, so scoring new points needs a nearest-neighbour rule or a
   classifier trained on the labels.)
3. **DBSCAN silhouette includes noise.** Points labelled `-1` are treated as
   their own cluster, which inflates or deflates the score arbitrarily. Mask
   them out before scoring.
4. **`Age = 2026 - Year_Birth` is hard-coded.** Use
   `datetime.now().year` so the feature does not silently drift.
5. **Mean-imputing `Income`** compresses variance in a feature the clustering
   depends on heavily. Median or a KNN imputer would distort less.
6. **`Age < 60` also removes genuine customers.** The intent is to drop the
   1893/1899 birth years; a targeted filter such as `Age < 100` keeps the real
   older segment intact.
7. **`Marital_Status` has redundant levels** (`Alone`, `Absurd`, `YOLO`).
   Collapsing them into `Single` / `Partner` would reduce one-hot sparsity and
   help the distance metrics.
8. **One-hot columns dominate the distance metric.** With 4 numeric features and
   ~12 binary ones, geometry is driven by categories. Gower distance or
   K-Prototypes handles mixed types more faithfully.
