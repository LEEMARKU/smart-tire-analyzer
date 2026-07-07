"""
AI Feature Engineering Preprocessing.
Implements ALL missing feature engineering steps:
  1. Feature extraction
  2. Feature selection (variance threshold, mutual information)
  3. Dimensionality reduction (PCA)
  4. Correlation analysis
  5. Feature encoding (label, one-hot)
  6. Class balancing (SMOTE)
  7. Data splitting
  8. Sequence generation
"""

import numpy as np
import pandas as pd
import logging
from typing import Optional, List, Tuple, Dict, Any
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)


def select_features_variance(
    X: pd.DataFrame,
    threshold: float = 0.01,
) -> pd.DataFrame:
    """
    Remove low-variance features using variance threshold.

    Args:
        X: Feature DataFrame
        threshold: Variance threshold (features below this are removed)

    Returns:
        DataFrame with selected features
    """
    variances = X.var(numeric_only=True)
    low_var = variances[variances < threshold].index.tolist()
    selected = X.drop(columns=low_var, errors="ignore")
    if low_var:
        logger.info(f"Removed {len(low_var)} low-variance features (threshold={threshold})")
    return selected


def select_features_mutual_info(
    X: pd.DataFrame,
    y: pd.Series,
    k: int = 20,
    seed: int = 42,
) -> List[str]:
    """
    Select top-k features by mutual information with target.

    Args:
        X: Feature DataFrame (numeric only)
        y: Target series
        k: Number of top features to select
        seed: Random seed

    Returns:
        List of selected feature names
    """
    from sklearn.feature_selection import SelectKBest, mutual_info_classif

    is_classification = y.dtype == "object" or y.nunique() < 20

    numeric_X = X.select_dtypes(include=[np.number]).fillna(0)

    if is_classification:
        score_func = mutual_info_classif
    else:
        from sklearn.feature_selection import mutual_info_regression
        score_func = mutual_info_regression

    n_features = min(k, numeric_X.shape[1])
    if n_features < 1:
        return []

    selector = SelectKBest(score_func=score_func, k=n_features)
    selector.fit(numeric_X, y)

    selected = [
        numeric_X.columns[i]
        for i in selector.get_support(indices=True)
        if i < len(numeric_X.columns)
    ]

    scores = sorted(zip(selected, selector.scores_[selector.get_support()]), key=lambda x: -x[1])
    logger.info(f"Top {len(selected)} features by mutual information:")
    for name, score in scores[:5]:
        logger.info(f"  {name}: {score:.4f}")

    return selected


def apply_pca(
    X: pd.DataFrame,
    n_components: Optional[int] = None,
    variance_ratio: float = 0.95,
    scale: bool = True,
    seed: int = 42,
) -> Tuple[np.ndarray, Any, Optional[List[str]]]:
    """
    Apply PCA for dimensionality reduction.

    Args:
        X: Feature DataFrame
        n_components: Number of components (auto if None)
        variance_ratio: Target explained variance ratio (used if n_components is None)
        scale: Standardize features before PCA
        seed: Random seed

    Returns:
        (transformed_array, pca_model, component_names)
    """
    from sklearn.decomposition import PCA

    numeric_X = X.select_dtypes(include=[np.number]).fillna(0)

    if scale:
        scaler = StandardScaler()
        scaled = scaler.fit_transform(numeric_X)
    else:
        scaler = None
        scaled = numeric_X.values

    if n_components is None:
        pca = PCA(random_state=seed)
        pca.fit(scaled)
        cumsum = np.cumsum(pca.explained_variance_ratio_)
        n_components = int(np.searchsorted(cumsum, variance_ratio) + 1)
        logger.info(f"Auto-selected {n_components} PCA components ({variance_ratio*100:.0f}% variance)")

    pca = PCA(n_components=n_components, random_state=seed)
    transformed = pca.fit_transform(scaled)

    component_names = [f"PC{i+1}" for i in range(n_components)]
    logger.info(f"PCA: {numeric_X.shape[1]} -> {n_components} components")
    logger.info(f"Explained variance ratio: {pca.explained_variance_ratio_.sum():.4f}")

    return transformed, pca, component_names


def correlation_analysis(
    df: pd.DataFrame,
    target_col: Optional[str] = None,
    threshold: float = 0.95,
) -> Dict[str, Any]:
    """
    Analyze feature correlations and detect multicollinearity.

    Args:
        df: DataFrame with features
        target_col: Optional target column for target-feature correlation
        threshold: Correlation threshold for identifying highly correlated pairs

    Returns:
        dict with keys: 'correlation_matrix', 'high_corr_pairs', 'target_corr'
    """
    numeric_df = df.select_dtypes(include=[np.number]).dropna(axis=1, how="all")

    corr_matrix = numeric_df.corr()

    high_corr = []
    cols = corr_matrix.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            val = abs(corr_matrix.iloc[i, j])
            if val > threshold:
                high_corr.append({
                    "feature_1": cols[i],
                    "feature_2": cols[j],
                    "correlation": corr_matrix.iloc[i, j],
                })
                logger.info(f"High correlation: {cols[i]} <-> {cols[j]} = {corr_matrix.iloc[i, j]:.3f}")

    target_corr = None
    if target_col and target_col in numeric_df.columns:
        target_corr = corr_matrix[target_col].drop(target_col).sort_values(ascending=False)
        top_5 = target_corr.head(5)
        for feat, val in top_5.items():
            logger.info(f"Target correlation: {feat} -> {val:.3f}")

    return {
        "correlation_matrix": corr_matrix,
        "high_corr_pairs": high_corr,
        "target_corr": target_corr,
    }


def label_encode(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    """
    Label encode categorical columns.

    Args:
        df: DataFrame
        columns: Columns to encode (all object/category if None)

    Returns:
        (encoded_df, encoders_dict)
    """
    result = df.copy()
    encoders: Dict[str, LabelEncoder] = {}

    if columns is None:
        columns = result.select_dtypes(include=["object", "category"]).columns.tolist()

    for col in columns:
        if col not in result.columns:
            continue
        le = LabelEncoder()
        valid = result[col].fillna("Unknown").astype(str)
        result[col + "_encoded"] = le.fit_transform(valid)
        encoders[col] = le
        logger.info(f"Label encoded '{col}': {len(le.classes_)} classes")

    return result, encoders


def one_hot_encode(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    max_categories: int = 20,
    drop_first: bool = False,
) -> pd.DataFrame:
    """
    One-hot encode categorical columns.

    Args:
        df: DataFrame
        columns: Columns to encode (all object/category if None)
        max_categories: Max categories to encode (others dropped)
        drop_first: Drop first category to avoid multicollinearity

    Returns:
        DataFrame with one-hot encoded columns
    """
    result = df.copy()

    if columns is None:
        columns = result.select_dtypes(include=["object", "category"]).columns.tolist()

    for col in columns:
        if col not in result.columns:
            continue

        value_counts = result[col].value_counts()
        if len(value_counts) > max_categories:
            top_categories = set(value_counts.head(max_categories).index)
            result[col] = result[col].apply(
                lambda x: x if x in top_categories else "Other"
            )
            logger.info(f"Collapsed '{col}' to top {max_categories} categories + Other")

        dummies = pd.get_dummies(result[col], prefix=col, drop_first=drop_first, dummy_na=False)
        result = pd.concat([result, dummies], axis=1)
        result = result.drop(columns=[col])
        logger.info(f"One-hot encoded '{col}': {dummies.shape[1]} new columns")

    return result


def apply_smote(
    X: np.ndarray,
    y: np.ndarray,
    sampling_strategy: str = "auto",
    k_neighbors: int = 5,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply SMOTE for class balancing.

    Args:
        X: Feature array
        y: Target array (classification)
        sampling_strategy: SMOTE sampling strategy
        k_neighbors: Number of nearest neighbors
        seed: Random seed

    Returns:
        (X_resampled, y_resampled)
    """
    try:
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(
            sampling_strategy=sampling_strategy,
            k_neighbors=k_neighbors,
            random_state=seed,
        )
        X_res, y_res = smote.fit_resample(X, y)
        logger.info(f"SMOTE: {len(y)} -> {len(y_res)} samples")
        original_counts = pd.Series(y).value_counts().to_dict()
        resampled_counts = pd.Series(y_res).value_counts().to_dict()
        logger.info(f"  Original: {original_counts}")
        logger.info(f"  Resampled: {resampled_counts}")
        return X_res, y_res
    except ImportError:
        logger.warning("imbalanced-learn not installed. Install with: pip install imbalanced-learn")
        return X, y


def split_data(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    val_size: float = 0.15,
    stratify: bool = True,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """
    Split data into train/val/test sets.

    Args:
        X: Feature array
        y: Target array
        test_size: Test set proportion
        val_size: Validation set proportion (of remaining after test)
        stratify: Stratify split by target
        seed: Random seed

    Returns:
        dict with keys: 'X_train', 'y_train', 'X_val', 'y_val', 'X_test', 'y_test'
    """
    from sklearn.model_selection import train_test_split

    stratify_option = y if stratify else None

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed,
        stratify=stratify_option,
    )

    val_relative = val_size / (1.0 - test_size)
    stratify_temp = y_temp if stratify else None

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_relative, random_state=seed,
        stratify=stratify_temp,
    )

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
    }


def build_tread_sequences(
    tread_depths: np.ndarray,
    sequence_length: int = 4,
    stride: int = 1,
) -> np.ndarray:
    """
    Build sequences for time-series models from tread depth measurements.

    Args:
        tread_depths: Array of shape (n_samples, 4) with T1-T4 measurements
        sequence_length: Number of time steps per sequence
        stride: Step between sequences

    Returns:
        Array of shape (n_sequences, sequence_length, 4)
    """
    n = len(tread_depths)
    sequences = []

    for i in range(0, n - sequence_length + 1, stride):
        sequences.append(tread_depths[i:i + sequence_length])

    if not sequences:
        return np.empty((0, sequence_length, 4))

    return np.array(sequences)


def feature_engineering_pipeline(
    df: pd.DataFrame,
    target_col: Optional[str] = None,
    do_variance_selection: bool = True,
    do_mutual_info: bool = False,
    do_pca: bool = False,
    do_correlation: bool = False,
    do_label_encode: bool = True,
    do_one_hot: bool = False,
    do_smote: bool = False,
    do_split: bool = False,
    sequence_length: int = 4,
) -> Dict[str, Any]:
    """
    Complete feature engineering preprocessing pipeline.

    Args:
        df: Input DataFrame
        target_col: Target column name
        do_variance_selection: Apply variance threshold
        do_mutual_info: Apply mutual information feature selection
        do_pca: Apply PCA
        do_correlation: Run correlation analysis
        do_label_encode: Label encode categorical features
        do_one_hot: One-hot encode categorical features
        do_smote: Apply SMOTE class balancing
        do_split: Split into train/val/test
        sequence_length: Sequence length for time-series

    Returns:
        dict with all results
    """
    result: Dict[str, Any] = {}
    working = df.copy()

    if target_col and target_col in working.columns:
        y_raw = working[target_col]
        X_raw = working.drop(columns=[target_col])
    else:
        X_raw = working
        y_raw = None

    if do_correlation:
        result["correlation"] = correlation_analysis(X_raw, target_col=target_col)

    if do_variance_selection:
        X_selected = select_features_variance(X_raw)
        result["variance_selected_features"] = list(X_selected.columns)
        X_raw = X_selected

    if y_raw is not None and do_mutual_info:
        mutual_features = select_features_mutual_info(X_raw, y_raw)
        if mutual_features:
            X_raw = X_raw[mutual_features]
            result["mutual_info_features"] = mutual_features

    if do_label_encode:
        cat_cols = X_raw.select_dtypes(include=["object", "category"]).columns.tolist()
        if cat_cols:
            X_raw, encoders = label_encode(X_raw, cat_cols)
            result["label_encoders"] = encoders

    if do_one_hot:
        cat_cols = X_raw.select_dtypes(include=["object", "category"]).columns.tolist()
        if cat_cols:
            X_raw = one_hot_encode(X_raw, cat_cols)
            result["one_hot_columns"] = list(
                set(X_raw.columns) - set(df.columns)
            )

    result["features"] = X_raw
    if y_raw is not None:
        result["target"] = y_raw

    if do_pca:
        pca_result, pca_model, pc_names = apply_pca(X_raw)
        result["pca_model"] = pca_model
        result["pca_components"] = pc_names
        pca_df = pd.DataFrame(
            pca_result, columns=pc_names, index=X_raw.index
        )
        result["pca_features"] = pca_df

    numeric_X = X_raw.select_dtypes(include=[np.number]).fillna(0).values
    if result.get("pca_features") is not None:
        numeric_X = result["pca_features"].values

    if y_raw is not None and do_smote:
        y_num = (
            LabelEncoder().fit_transform(y_raw.fillna("Unknown").astype(str))
            if y_raw.dtype == "object"
            else y_raw.values
        )
        X_res, y_res = apply_smote(numeric_X, y_num)
        result["X_resampled"] = X_res
        result["y_resampled"] = y_res
        numeric_X = X_res
        if isinstance(y_raw, pd.Series):
            result["target"] = y_res

    if do_split and y_raw is not None:
        y_for_split = (
            result.get("y_resampled", y_raw)
            if isinstance(result.get("y_resampled"), np.ndarray)
            else y_raw.values if isinstance(y_raw, pd.Series) else y_raw
        )
        splits = split_data(numeric_X, y_for_split)
        result.update(splits)
        logger.info(
            f"Data split: train={len(splits['X_train'])}, "
            f"val={len(splits['X_val'])}, test={len(splits['X_test'])}"
        )

    result["sequences"] = None
    if "tread_1" in df.columns:
        tread_cols = [c for c in ["tread_1", "tread_2", "tread_3", "tread_4"] if c in df.columns]
        if len(tread_cols) == 4:
            tread_data = df[tread_cols].fillna(0).values
            sequences = build_tread_sequences(tread_data, sequence_length=sequence_length)
            result["sequences"] = sequences
            logger.info(f"Built {len(sequences)} tread sequences (len={sequence_length})")

    return result


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        "tread_1": np.random.uniform(1, 8, n),
        "tread_2": np.random.uniform(1, 8, n),
        "tread_3": np.random.uniform(1, 8, n),
        "tread_4": np.random.uniform(1, 8, n),
        "brand": np.random.choice(["Michelin", "Bridgestone", "Goodyear", "Pirelli"], n),
        "wear_pattern": np.random.choice(["even", "center", "edge", "patchy"], n),
        "temperature_c": np.random.normal(25, 5, n),
        "pressure_psi": np.random.normal(35, 3, n),
    })

    result = feature_engineering_pipeline(
        df,
        target_col="wear_pattern",
        do_label_encode=True,
        do_correlation=True,
        do_pca=False,
        do_split=True,
    )
    logger.info(f"Features shape: {result['features'].shape}")
    logger.info(f"Split keys: {[k for k in result.keys() if k.startswith('X_') or k.startswith('y_')]}")
