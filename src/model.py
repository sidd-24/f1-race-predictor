"""
F1 Race Position Prediction Model

Trains and evaluates ML models to predict race finishing positions.
Supports Random Forest and XGBoost models.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import logging
import argparse

from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from .config import (
    DATA_DIR,
    MODELS_DIR,
    RANDOM_STATE,
    TEST_SIZE,
    CV_FOLDS,
    XGBOOST_PARAMS,
    RANDOM_FOREST_PARAMS,
    DEFAULT_DATA_PREFIX
)
from .feature_engineering import get_feature_columns, get_metadata_columns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_features(prefix: str = DEFAULT_DATA_PREFIX) -> pd.DataFrame:
    """Load preprocessed features."""
    filepath = DATA_DIR / f"{prefix}_features.csv"
    if not filepath.exists():
        raise FileNotFoundError(f"Features file not found: {filepath}")

    df = pd.read_csv(filepath)
    logger.info(f"Loaded {len(df)} samples from {filepath.name}")
    return df


def validate_training_data(df: pd.DataFrame) -> None:
    """Validate training rows before fitting models."""
    required = set(get_feature_columns() + get_metadata_columns() + ['FinishingPosition'])
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Training data is missing required columns: {', '.join(missing)}")

    numeric_columns = get_feature_columns() + ['Year', 'Round', 'FinishingPosition']
    for column in numeric_columns:
        values = pd.to_numeric(df[column], errors='coerce')
        if values.isna().any():
            raise ValueError(f"Training data contains invalid values in '{column}'")

    if (df['FinishingPosition'] < 1).any() or (df['FinishingPosition'] > 30).any():
        raise ValueError("Training data contains invalid finishing positions")

    duplicate_keys = df.duplicated(subset=['Year', 'Round', 'Driver'])
    if duplicate_keys.any():
        logger.warning("Training data contains %d duplicate driver/race rows", duplicate_keys.sum())

    race_sizes = df.groupby(['Year', 'Round'])['Driver'].nunique()
    incomplete = race_sizes[race_sizes < 10]
    if not incomplete.empty:
        logger.warning(
            "Training data contains %d races with fewer than 10 drivers: %s",
            len(incomplete),
            list(incomplete.index)
        )


def prepare_xy(df: pd.DataFrame) -> tuple:
    """
    Split dataframe into features (X) and target (y), and calculate sample weights.

    Returns:
        tuple: (X, y, metadata, weights)
    """
    feature_cols = get_feature_columns()
    metadata_cols = get_metadata_columns()

    X = df[feature_cols].copy()
    y = df['FinishingPosition'].copy()
    metadata = df[metadata_cols].copy()

    # Sample Weighting for Regulation Changes
    # Give significantly more weight to the most recent season
    # we assume 'Year' is available in metadata or the original df
    # Since prepare_xy takes df, we can access it
    current_year = 2026 
    weights = np.where(df['Year'] == current_year, 10.0, 1.0)

    return X, y, metadata, weights


def train_random_forest(X_train, y_train, weights_train=None) -> RandomForestRegressor:
    """Train Random Forest model."""
    logger.info("Training Random Forest model...")

    model = RandomForestRegressor(**RANDOM_FOREST_PARAMS)
    model.fit(X_train, y_train, sample_weight=weights_train)

    return model


def train_xgboost(X_train, y_train, weights_train=None) -> XGBRegressor:
    """Train XGBoost model."""
    logger.info("Training XGBoost model...")

    model = XGBRegressor(**XGBOOST_PARAMS)
    model.fit(X_train, y_train, sample_weight=weights_train)

    return model


def evaluate_model(model, X_test, y_test, model_name: str = "Model") -> dict:
    """
    Evaluate model performance.

    Returns:
        dict: Evaluation metrics
    """
    y_pred = model.predict(X_test)

    # Round predictions to nearest integer (positions are discrete)
    y_pred_rounded = np.round(y_pred).clip(1, 20)

    mae = mean_absolute_error(y_test, y_pred_rounded)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_rounded))
    r2 = r2_score(y_test, y_pred_rounded)

    # Custom metric: % predictions within 3 positions
    within_3 = np.mean(np.abs(y_test - y_pred_rounded) <= 3) * 100

    metrics = {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'Within_3_positions': within_3
    }

    logger.info(f"\n{model_name} Performance:")
    logger.info(f"  MAE: {mae:.2f} positions")
    logger.info(f"  RMSE: {rmse:.2f}")
    logger.info(f"  R²: {r2:.3f}")
    logger.info(f"  Within 3 positions: {within_3:.1f}%")

    return metrics


def cross_validate(model, X, y, cv: int = CV_FOLDS) -> dict:
    """
    Perform cross-validation.

    Returns:
        dict: CV scores
    """
    logger.info(f"Performing {cv}-fold cross-validation...")

    # Negative MAE is used because sklearn maximizes scores
    scores = cross_val_score(model, X, y, cv=TimeSeriesSplit(n_splits=cv), scoring='neg_mean_absolute_error')

    cv_results = {
        'mean_mae': -scores.mean(),
        'std_mae': scores.std(),
        'all_scores': -scores
    }

    logger.info(f"  CV MAE: {cv_results['mean_mae']:.2f} (+/- {cv_results['std_mae']:.2f})")

    return cv_results


def get_feature_importance(model, feature_names: list) -> pd.DataFrame:
    """
    Get feature importance from trained model.

    Returns:
        DataFrame with feature importances sorted by importance
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        importances = np.zeros(len(feature_names))

    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)

    return importance_df


def save_model(model, model_name: str):
    """Save trained model to disk."""
    filepath = MODELS_DIR / f"{model_name}.joblib"
    joblib.dump(model, filepath)
    logger.info(f"Model saved to {filepath}")


def load_model(model_name: str):
    """Load trained model from disk."""
    filepath = MODELS_DIR / f"{model_name}.joblib"
    if not filepath.exists():
        raise FileNotFoundError(f"Model not found: {filepath}")

    model = joblib.load(filepath)
    logger.info(f"Model loaded from {filepath}")
    return model


def train_and_evaluate(prefix: str = DEFAULT_DATA_PREFIX):
    """
    Full training pipeline.

    Trains both Random Forest and XGBoost models,
    evaluates them, and saves the best one.
    """
    # Load data
    df = load_features(prefix)
    validate_training_data(df)
    X, y, metadata, weights = prepare_xy(df)

    # Keep the newest races as the test set to reflect real future predictions.
    ordered = df.sort_values(['Year', 'Round']).index
    test_count = max(1, int(len(ordered) * TEST_SIZE))
    train_indices = ordered[:-test_count]
    test_indices = ordered[-test_count:]
    X_train, X_test = X.loc[train_indices], X.loc[test_indices]
    y_train, y_test = y.loc[train_indices], y.loc[test_indices]
    w_train, w_test = weights[train_indices], weights[test_indices]

    logger.info(f"Training set: {len(X_train)} samples")
    logger.info(f"Test set: {len(X_test)} samples")

    # Train models with sample weights
    rf_model = train_random_forest(X_train, y_train, weights_train=w_train)
    xgb_model = train_xgboost(X_train, y_train, weights_train=w_train)

    # Evaluate models
    print("\n" + "="*50)
    print("MODEL EVALUATION")
    print("="*50)

    rf_metrics = evaluate_model(rf_model, X_test, y_test, "Random Forest")
    xgb_metrics = evaluate_model(xgb_model, X_test, y_test, "XGBoost")

    # Cross-validation
    print("\n" + "-"*50)
    print("Cross-Validation Results")
    print("-"*50)
    cv_order = df.sort_values(['Year', 'Round']).index
    X_cv = X.loc[cv_order]
    y_cv = y.loc[cv_order]
    rf_cv = cross_validate(RandomForestRegressor(**RANDOM_FOREST_PARAMS), X_cv, y_cv)
    xgb_cv = cross_validate(XGBRegressor(**XGBOOST_PARAMS), X_cv, y_cv)

    # Feature importance
    print("\n" + "-"*50)
    print("Feature Importance (XGBoost)")
    print("-"*50)
    importance_df = get_feature_importance(xgb_model, get_feature_columns())
    print(importance_df.to_string(index=False))

    # Determine best model
    best_model = xgb_model if xgb_metrics['MAE'] < rf_metrics['MAE'] else rf_model
    best_name = "xgboost" if xgb_metrics['MAE'] < rf_metrics['MAE'] else "random_forest"

    # Save models
    save_model(rf_model, "random_forest")
    save_model(xgb_model, "xgboost")
    save_model(best_model, "best_model")

    print("\n" + "="*50)
    print(f"Best model: {best_name.upper()}")
    print(f"MAE: {min(rf_metrics['MAE'], xgb_metrics['MAE']):.2f} positions")
    print("="*50)

    return {
        'random_forest': {'model': rf_model, 'metrics': rf_metrics, 'cv': rf_cv},
        'xgboost': {'model': xgb_model, 'metrics': xgb_metrics, 'cv': xgb_cv},
        'best': best_name
    }


def predict_positions(model, X: pd.DataFrame) -> np.ndarray:
    """
    Make predictions with trained model.

    Returns:
        Predicted positions (rounded and clipped to 1-20)
    """
    predictions = model.predict(X)
    # Round and clip to valid position range
    predictions = np.round(predictions).clip(1, 20).astype(int)
    return predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train F1 race prediction models")
    parser.add_argument('--prefix', type=str, default=DEFAULT_DATA_PREFIX, help='Input feature filename prefix')
    args = parser.parse_args()
    results = train_and_evaluate(args.prefix)
