#!/usr/bin/env python3
"""
Enhanced and consolidated model training module for the ML Trading Analysis System.
This module provides a unified `ModelTrainer` class that handles single model training,
ensemble methods, hyperparameter optimization, and robust time series validation.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, recall_score, f1_score,
                             mean_squared_error, mean_absolute_error, r2_score)
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, VotingClassifier,
                              RandomForestRegressor, GradientBoostingRegressor, VotingRegressor)
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
import lightgbm as lgb
import optuna
import shap
import joblib
from pathlib import Path
import warnings

from src.utils import Config
from src.features import FeatureEngineer

warnings.filterwarnings('ignore', category=UserWarning, module='lightgbm')
logger = logging.getLogger(__name__)

LightGBMModel = Union[lgb.LGBMClassifier, lgb.LGBMRegressor]


class TimeSeriesValidator:
    """Time series validation with configurable splits and test size."""

    def __init__(self, config: Config):
        self.n_splits = config.model.get('cv_folds', 5)
        self.test_size = config.model.get('cv_test_size', 0.2)
        self.logger = logging.getLogger(__name__)

    def split(self, X: pd.DataFrame, y: pd.Series = None, groups=None):
        """Generate indices for expanding window time series cross-validation."""
        self.logger.info(f"Generating {self.n_splits} splits for time series cross-validation with test size {self.test_size}.")
        total_rows = len(X)
        test_rows = int(total_rows * self.test_size)

        for i in range(self.n_splits):
            test_start = total_rows - (self.n_splits - i) * test_rows
            test_end = test_start + test_rows
            train_end = test_start

            test_start, test_end = max(0, test_start), min(total_rows, test_end)

            if test_start >= test_end or train_end <= 0:
                continue

            train_indices = np.arange(0, train_end)
            test_indices = np.arange(test_start, test_end)

            if len(train_indices) > 0 and len(test_indices) > 0:
                self.logger.info(f"Split {i+1}: Train size={len(train_indices)}, Test size={len(test_indices)}")
                yield train_indices, test_indices

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits


class ModelTrainer:
    """Unified model trainer with validation, optimization, and ensemble methods."""

    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.models: Dict[str, object] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.feature_engineers: Dict[str, FeatureEngineer] = {}
        self.results: Dict[str, dict] = {}
        self.validator = TimeSeriesValidator(config)

    def train_single_model(self, df: pd.DataFrame, label_column: str, task_type: str,
                           optimize_hyperparams: bool = False, compute_shap: bool = False) -> Dict:
        """Train a single LightGBM model with optional hyperparameter tuning and SHAP analysis."""
        self.logger.info(f"Starting single model training for '{label_column}' (Task: {task_type}).")

        feature_engineer = FeatureEngineer(self.config)
        X, y = self._prepare_features(df, feature_engineer, label_column)
        self.feature_engineers[label_column] = feature_engineer

        X_train, X_test, y_train, y_test = self._time_series_split(X, y)

        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
        self.scalers[label_column] = scaler

        model_params = self.config.model.get('lgbm_params', {})
        if optimize_hyperparams:
            self.logger.info("Optimizing hyperparameters...")
            best_params = self._optimize_hyperparameters(X_train_scaled, y_train, task_type)
            model_params.update(best_params)

        model = self._train_lgbm(X_train_scaled, y_train, X_test_scaled, y_test, task_type, model_params)
        self.models[label_column] = model

        metrics = self._evaluate_model(model, X_test_scaled, y_test, task_type)
        self.logger.info(f"Test metrics for '{label_column}': {metrics}")

        feature_importance = self._get_feature_importance(model, X.columns)

        shap_values = self._compute_shap_values(model, X_test_scaled) if compute_shap else None

        result = {
            'model': model,
            'test_metrics': metrics,
            'feature_importance': feature_importance,
            'shap_values': shap_values
        }
        self.results[label_column] = result
        return result

    def _prepare_features(self, df: pd.DataFrame, feature_engineer: FeatureEngineer, label_column: str) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and labels for training."""
        df_featured = feature_engineer.build_feature_matrix(df)
        
        exclude_cols = ['timestamp'] + [col for col in df_featured.columns if col.startswith('label_') or col.startswith('return_')]
        feature_cols = [col for col in df_featured.columns if col not in exclude_cols]
        X = df_featured[feature_cols].copy()
        y = df_featured[label_column]

        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        X.fillna(X.median(), inplace=True)
        
        common_index = X.index.intersection(y.index)
        return X.loc[common_index], y.loc[common_index]

    def _time_series_split(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split data into train and test sets based on time."""
        train_size = self.config.model.get('train_size', 0.8)
        split_index = int(len(X) * train_size)
        
        X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
        y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
        
        self.logger.info(f"Data split: Train size={len(X_train)}, Test size={len(X_test)}")
        return X_train, X_test, y_train, y_test

    def _train_lgbm(self, X_train, y_train, X_val, y_val, task_type, params):
        """Helper to train a LightGBM model."""
        if task_type == 'classification':
            model = lgb.LGBMClassifier(**params)
        else:
            model = lgb.LGBMRegressor(**params)

        model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  eval_metric='logloss' if task_type == 'classification' else 'rmse',
                  callbacks=[lgb.early_stopping(100, verbose=False)])
        return model

    def _evaluate_model(self, model, X_test, y_test, task_type):
        """Evaluate model performance."""
        preds = model.predict(X_test)
        if task_type == 'classification':
            proba_preds = model.predict_proba(X_test)[:, 1]
            return {
                'accuracy': accuracy_score(y_test, preds),
                'roc_auc': roc_auc_score(y_test, proba_preds),
                'precision': precision_score(y_test, preds, zero_division=0),
                'recall': recall_score(y_test, preds, zero_division=0),
                'f1': f1_score(y_test, preds, zero_division=0)
            }
        else:
            return {
                'r2': r2_score(y_test, preds),
                'rmse': np.sqrt(mean_squared_error(y_test, preds)),
                'mae': mean_absolute_error(y_test, preds)
            }

    def _get_feature_importance(self, model, feature_names):
        """Get feature importance from a trained model."""
        return pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False).reset_index(drop=True)

    def _compute_shap_values(self, model, X):
        """Compute SHAP values for model interpretability."""
        explainer = shap.TreeExplainer(model)
        return explainer.shap_values(X)

    def _optimize_hyperparameters(self, X, y, task_type):
        """Optimize hyperparameters using Optuna."""
        
        # Store results
        results = {
            'model': model,
            'feature_engineer': feature_engineer,
            'test_metrics': test_metrics,
            'cv_scores': cv_scores,
            'shap_results': shap_results,
            'feature_importance': model.get_feature_importance()
        }
        
        self.models[label_column] = results
        self.feature_engineers[label_column] = feature_engineer
        
        self.logger.info(f"Model training completed. Test accuracy: {test_metrics.get('accuracy', 0):.4f}")
        
        return results
    
    def hyperparameter_tuning(self, df: pd.DataFrame, label_column: str,
                            task_type: str = "classification", n_trials: int = 50) -> Dict:
        """Hyperparameter tuning with Optuna."""
        self.logger.info("Starting hyperparameter tuning")
        
        # Prepare features and labels
        feature_engineer = FeatureEngineer(self.config)
        X = feature_engineer.build_feature_matrix(df, fit_pipeline=True)
        y = df[label_column]
        
        # Align data
        common_idx = X.index.intersection(y.index)
        X = X.loc[common_idx]
        y = y.loc[common_idx]
        
        # Clean data
        X_clean = X.copy()
        X_clean = X_clean.replace([np.inf, -np.inf], np.nan)
        X_clean = X_clean.fillna(X_clean.median())
        
        # Clip extreme values
        for col in X_clean.columns:
            if X_clean[col].dtype in ['float64', 'float32']:
                Q1 = X_clean[col].quantile(0.001)
                Q3 = X_clean[col].quantile(0.999)
                X_clean[col] = X_clean[col].clip(Q1, Q3)
        
        def objective(trial):
            try:
                # LightGBM parameters
                lgb_params = {
                    'objective': 'binary',
                    'metric': 'binary_logloss',
                    'boosting_type': 'gbdt',
                    'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
                    'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
                    'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
                    'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
                    'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                    'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                    'n_estimators': 100,
                    'verbose': -1,
                    'random_state': 42
                }
                
                # Use TimeSeriesSplit for validation
                tscv = TimeSeriesSplit(n_splits=3)
                lgb_scores = cross_val_score(
                    lgb.LGBMClassifier(**lgb_params), X_clean, y, cv=tscv, scoring='roc_auc'
                )
                
                # Check for valid scores
                if np.isnan(lgb_scores).any() or np.isinf(lgb_scores).any():
                    return 0.5  # Return default score for failed trials
                
                return lgb_scores.mean()
                
            except Exception as e:
                self.logger.warning(f"Trial failed: {e}")
                return 0.5  # Return default score for failed trials
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        self.logger.info(f"Best hyperparameters found: {study.best_params}")
        self.logger.info(f"Best CV score: {study.best_value:.4f}")
        
        return {
            'best_params': study.best_params,
            'best_score': study.best_value,
            'n_trials': n_trials
        }
    
    def train_all_models(self, df: pd.DataFrame) -> Dict:
        """Train models for all label columns."""
        label_columns = [col for col in df.columns if col.startswith('label_')]
        
        results = {}
        for label_col in label_columns:
            try:
                result = self.train_single_model(df, label_col)
                results[label_col] = result
            except Exception as e:
                self.logger.error(f"Error training model for {label_col}: {e}")
        
        return results
    
    def get_best_model(self, metric: str = "accuracy") -> Tuple[str, LightGBMModel]:
        """Get the best performing model."""
        best_score = -1
        best_label = None
        best_model = None
        
        for label_col, result in self.models.items():
            score = result['test_metrics'].get(metric, 0)
            if score > best_score:
                best_score = score
                best_label = label_col
                best_model = result['model']
        
        return best_label, best_model
    
    def save_all_models(self, base_path: str) -> None:
        """Save all trained models."""
        for label_col, result in self.models.items():
            model_path = f"{base_path}/{label_col}_model.txt"
            result['model'].save_model(model_path)
            
            # Save feature engineer
            feature_engineer_path = f"{base_path}/{label_col}_feature_engineer.pkl"
            result['feature_engineer'].save_pipeline_state(feature_engineer_path)
    
    def get_summary_report(self) -> pd.DataFrame:
        """Get summary report of all models."""
        summary_data = []
        
        for label_col, result in self.models.items():
            metrics = result['test_metrics']
            summary_data.append({
                'label': label_col,
                'accuracy': metrics.get('accuracy', 0),
                'roc_auc': metrics.get('roc_auc', 0),
                'precision': metrics.get('precision', 0),
                'recall': metrics.get('recall', 0),
                'f1': metrics.get('f1', 0)
            })
        
        return pd.DataFrame(summary_data)


class RealTimePredictor:
    """Real-time predictor for a single model."""
    
    def __init__(self, config, model, feature_engineer):
        self.config = config
        self.model = model
        self.feature_engineer = feature_engineer
        self.logger = logging.getLogger(__name__)
        self.prediction_history = []
    
    def prepare_live_features(self, data):
        """Prepare features for live prediction."""
        # This should handle a single data point or a small batch
        return self.feature_engineer.build_live_feature_matrix(data)
    
    def predict(self, data: pd.DataFrame) -> Dict:
        """Make real-time prediction with confidence score."""
        try:
            # Prepare features
            X = self.prepare_live_features(data)
            
            # Make prediction
            prediction_proba = self.model.predict_proba(X)
            prediction = self.model.predict(X)
            
            # Handle single prediction case
            if len(prediction) == 1:
                pred_value = int(prediction[0])
                if len(prediction_proba.shape) > 1:
                    confidence = float(np.max(prediction_proba, axis=1)[0])
                    prob_value = float(prediction_proba[0, 1])  # Positive class probability
                else:
                    confidence = float(abs(prediction_proba[0] - 0.5) * 2)
                    prob_value = float(prediction_proba[0])
            else:
                # Handle multiple predictions - take the last one for real-time
                pred_value = int(prediction[-1])
                if len(prediction_proba.shape) > 1:
                    confidence = float(np.max(prediction_proba, axis=1)[-1])
                    prob_value = float(prediction_proba[-1, 1])
                else:
                    confidence = float(abs(prediction_proba[-1] - 0.5) * 2)
                    prob_value = float(prediction_proba[-1])
            
            return {
                'prediction': pred_value,
                'confidence': confidence,
                'probability': prob_value
            }
            
        except Exception as e:
            self.logger.error(f"Error in real-time prediction: {e}")
            return {
                'prediction': None,
                'confidence': 0.0,
                'error': str(e)
            }
    
    def get_prediction_summary(self, window: int = 100) -> Dict:
        """Get summary of recent predictions."""
        # This would need to be implemented with actual prediction history
        return {
            'total_predictions': 0,
            'accuracy': 0.0,
            'avg_confidence': 0.0
        }


class RealTimeEnsemblePredictor:
    """Real-time ensemble predictor."""
    
    def __init__(self, config, trained_models):
        self.config = config
        self.trained_models = trained_models
        self.logger = logging.getLogger(__name__)
    
    def prepare_live_features(self, data, feature_engineer):
        """Prepare features for ensemble prediction."""
        return feature_engineer.build_live_feature_matrix(data)
    
    def predict(self, data, label_name='label_class_1'):
        """Make ensemble prediction."""
        if label_name not in self.trained_models:
            self.logger.error(f"Model for {label_name} not found")
            return None
        
        model_result = self.trained_models[label_name]
        predictor = RealTimePredictor(
            self.config, 
            model_result['model'], 
            model_result['feature_engineer']
        )
        
        return predictor.predict(data)

