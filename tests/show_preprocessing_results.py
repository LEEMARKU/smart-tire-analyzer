"""Show preprocessing results summary."""

import pandas as pd
import os

print("=" * 72)
print("PREPROCESSING RESULTS SUMMARY")
print("=" * 72)

orig = pd.read_csv("dataset/processed/labels.csv")
cleaned = pd.read_csv("dataset/processed/labels_text_cleaned.csv")
fe = pd.read_csv("dataset/processed/features_engineered.csv")

print(f"\n--- TEXT PREPROCESSING ---")
print(f"Input:  dataset/processed/labels.csv ({len(orig)} rows)")
print(f"Output: dataset/processed/labels_text_cleaned.csv ({len(cleaned)} rows)")

for col in ["brand", "tire_model", "ocr_text", "tire_size"]:
    if col not in orig.columns or col not in cleaned.columns:
        continue
    o = orig[col].fillna("").astype(str).str.lower().str.strip()
    c = cleaned[col].fillna("").astype(str).str.lower().str.strip()
    changed = (o != c).sum()
    if changed > 0:
        changed_idx = (o != c)
        idx = changed_idx[changed_idx].index[0]
        print(f"\n  {col}: {changed} values changed")
        print(f"    Before: \"{orig[col].iloc[idx]}\"")
        print(f"    After:  \"{cleaned[col].iloc[idx]}\"")

print(f"\n--- FEATURE ENGINEERING ---")
print(f"Output: dataset/processed/features_engineered.csv")
print(f"  Shape: {fe.shape}")
cat_cols = [c for c in fe.columns if "_encoded" in c]
print(f"  Label-encoded columns: {len(cat_cols)}")
for c in cat_cols:
    base = c.replace("_encoded", "")
    n_classes = fe[c].nunique()
    print(f"    {base} -> {n_classes} classes")
print(f"  Numeric columns preserved: {len(fe.select_dtypes(include=['float','int']).columns)}")
print(f"  Tread sequence builders: {'tread_1' in fe.columns}")

print(f"\n--- CORRELATION INSIGHTS ---")
numeric = fe.select_dtypes(include=["float", "int"])
corr = numeric.corr().abs()
high_corr = (corr > 0.95) & (corr < 1.0)
high_pairs = [
    (corr.columns[i], corr.columns[j], corr.iloc[i, j])
    for i in range(len(corr.columns))
    for j in range(i + 1, len(corr.columns))
    if high_corr.iloc[i, j]
]
print(f"  Highly correlated feature pairs (>0.95): {len(high_pairs)}")
for col1, col2, val in high_pairs[:5]:
    print(f"    {col1} <-> {col2} = {val:.3f}")

print(f"\n--- OUTPUT FILES ---")
for f in ["labels_text_cleaned.csv", "features_engineered.csv"]:
    path = f"dataset/processed/{f}"
    size_kb = os.path.getsize(path) / 1024
    print(f"  {f}: {size_kb:.1f} KB")
