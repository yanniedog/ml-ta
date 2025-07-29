#!/usr/bin/env python3
"""
Enhanced model training module with ensemble methods and advanced ML techniques.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, classification_report, precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import optuna
import shap
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

from .utils import Config
from .indicators import TechnicalIndicators
from .features import FeatureEngineer, FeaturePipeline


class TimeSeriesValidator:
    """Time series validation with proper train/test separation."""
    
    def __init__(self, n_splits: int = 5, test_size: float = 0.2):
        self.n_splits = n_splits
        self.test_size = test_size
        self.logger = logging.getLogger(__name__)
    
    def split_data(self, df: pd.DataFrame) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """Split data chronologically for time series validation."""
        splits = []
        total_rows = len(df)
        test_rows = int(total_rows * self.test_size)
        
        for i in range(self.n_splits):
            # Calculate split points
            test_start = total_rows - (self.n_splits - i) * test_rows
            test_end = test_start + test_rows
            
            # Ensure valid ranges
            test_start = max(0, test_start)
            test_end = min(total_rows, test_end)
            
            if test_start >= test_end:
                continue
            
            # Split data
            train_data = df.iloc[:test_start]
            test_data = df.iloc[test_start:test_end]
            
            if len(train_data) > 0 and len(test_data) > 0:
                splits.append((train_data, test_data))
                self.logger.info(f"Split {i+1}: Train {len(train_data)}, Test {len(test_data)}")
        
        return splits


class AdvancedModelTrainer:
    """Advanced model trainer with ensemble methods and hyperparameter optimization."""
    
    def __init__(self, config):
        self.config = config
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.ensemble_model = None
        self.validator = TimeSeriesValidator(n_splits=5)
        self.feature_engineers = {}  # Store fitted feature engineers
        
    def optimize_hyperparameters(self, X, y, n_trials=50):
        """Optimize hyperparameters using Optuna with TimeSeriesSplit."""
        logger.info("Starting hyperparameter optimization...")
        
        # Clean data first
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
                # LightGBM parameters with stronger regularization to prevent overfitting
                lgb_params = {
                    'num_leaves': trial.suggest_int('num_leaves', 10, 100),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
                    'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
                    'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
                    'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
                    'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                    'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                    # Add stronger regularization
                    'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 100),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'early_stopping_rounds': 50,
                    'verbose': -1
                }
                
                # Use TimeSeriesSplit for proper validation
                tscv = TimeSeriesSplit(n_splits=3)
                scores = []
                
                for train_idx, val_idx in tscv.split(X_clean):
                    X_train_fold, X_val_fold = X_clean.iloc[train_idx], X_clean.iloc[val_idx]
                    y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
                    
                    # Train model
                    model = lgb.LGBMClassifier(**lgb_params, random_state=42)
                    model.fit(X_train_fold, y_train_fold, 
                            eval_set=[(X_val_fold, y_val_fold)],
                            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
                    
                    # Predict
                    y_pred_proba = model.predict_proba(X_val_fold)[:, 1]
                    
                    # Calculate ROC AUC (better for imbalanced data)
                    try:
                        score = roc_auc_score(y_val_fold, y_pred_proba)
                        scores.append(score)
                    except:
                        scores.append(0.5)  # Default score for edge cases
                
                # Return mean score
                return np.mean(scores)
                
            except Exception as e:
                logger.warning(f"Trial failed: {e}")
                return 0.5  # Return default score for failed trials
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        logger.info(f"Best hyperparameters found: {study.best_params}")
        logger.info(f"Best CV score: {study.best_value:.4f}")
        
        return study.best_params
    
    def train_ensemble_model(self, X, y, label_name, task_type='classification', feature_engineer=None):
        """Train an ensemble model with multiple algorithms using proper validation."""
        logger.info(f"Training ensemble model for {label_name}")
        
        # Store the feature engineer if provided
        if feature_engineer is not None:
            self.feature_engineers[label_name] = feature_engineer
        
        # Clean data - handle infinity and extreme values more aggressively
        X_clean = X.copy()
        X_clean = X_clean.replace([np.inf, -np.inf], np.nan)
        X_clean = X_clean.fillna(X_clean.median())
        
        # More aggressive clipping for extreme values
        for col in X_clean.columns:
            if X_clean[col].dtype in ['float64', 'float32']:
                Q1 = X_clean[col].quantile(0.001)  # More aggressive
                Q3 = X_clean[col].quantile(0.999)  # More aggressive
                X_clean[col] = X_clean[col].clip(Q1, Q3)
                
                # Additional check for very large values
                if X_clean[col].abs().max() > 1e6:
                    X_clean[col] = X_clean[col].clip(-1e6, 1e6)
        
        logger.info(f"Data cleaned: {X_clean.shape}")
        logger.info(f"Infinity values removed: {(X == np.inf).sum().sum()}")
        
        # Use TimeSeriesSplit for proper validation
        tscv = TimeSeriesSplit(n_splits=3)
        
        # Train individual models
        models = {}
        scores = {}
        
        # LightGBM
        logger.info("Training lgb model...")
        lgb_params = self.optimize_hyperparameters(X_clean, y, n_trials=20)
        lgb_model = lgb.LGBMClassifier(**lgb_params, random_state=42)
        
        # Cross-validate with TimeSeriesSplit
        lgb_scores = cross_val_score(lgb_model, X_clean, y, cv=tscv, scoring='roc_auc')
        lgb_model.fit(X_clean, y)
        
        models['lgb'] = lgb_model
        scores['lgb'] = {
            'roc_auc': lgb_scores.mean(),
            'roc_auc_std': lgb_scores.std(),
            'accuracy': accuracy_score(y, lgb_model.predict(X_clean)),
            'precision': precision_score(y, lgb_model.predict(X_clean)),
            'recall': recall_score(y, lgb_model.predict(X_clean)),
            'f1': f1_score(y, lgb_model.predict(X_clean))
        }
        
        # Random Forest
        logger.info("Training rf model...")
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_scores = cross_val_score(rf_model, X_clean, y, cv=tscv, scoring='roc_auc')
        rf_model.fit(X_clean, y)
        
        models['rf'] = rf_model
        scores['rf'] = {
            'roc_auc': rf_scores.mean(),
            'roc_auc_std': rf_scores.std(),
            'accuracy': accuracy_score(y, rf_model.predict(X_clean)),
            'precision': precision_score(y, rf_model.predict(X_clean)),
            'recall': recall_score(y, rf_model.predict(X_clean)),
            'f1': f1_score(y, rf_model.predict(X_clean))
        }
        
        # Gradient Boosting
        logger.info("Training gb model...")
        gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        gb_scores = cross_val_score(gb_model, X_clean, y, cv=tscv, scoring='roc_auc')
        gb_model.fit(X_clean, y)
        
        models['gb'] = gb_model
        scores['gb'] = {
            'roc_auc': gb_scores.mean(),
            'roc_auc_std': gb_scores.std(),
            'accuracy': accuracy_score(y, gb_model.predict(X_clean)),
            'precision': precision_score(y, gb_model.predict(X_clean)),
            'recall': recall_score(y, gb_model.predict(X_clean)),
            'f1': f1_score(y, gb_model.predict(X_clean))
        }
        
        # Create ensemble
        logger.info("Training ensemble model...")
        ensemble_model = VotingClassifier(
            estimators=[
                ('lgb', models['lgb']),
                ('rf', models['rf']),
                ('gb', models['gb'])
            ],
            voting='soft'
        )
        
        ensemble_scores = cross_val_score(ensemble_model, X_clean, y, cv=tscv, scoring='roc_auc')
        ensemble_model.fit(X_clean, y)
        
        models['ensemble'] = ensemble_model
        scores['ensemble'] = {
            'roc_auc': ensemble_scores.mean(),
            'roc_auc_std': ensemble_scores.std(),
            'accuracy': accuracy_score(y, ensemble_model.predict(X_clean)),
            'precision': precision_score(y, ensemble_model.predict(X_clean)),
            'recall': recall_score(y, ensemble_model.predict(X_clean)),
            'f1': f1_score(y, ensemble_model.predict(X_clean))
        }
        
        # Store results
        self.models = models
        self.ensemble_model = ensemble_model
        
        # Calculate feature importance for LightGBM
        if hasattr(lgb_model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': X_clean.columns,
                'importance': lgb_model.feature_importances_
            }).sort_values('importance', ascending=False)
            self.feature_importance = feature_importance
        
        return {
            'models': models,
            'scores': scores,
            'feature_importance': self.feature_importance
        }
    
    def cross_validate_ensemble(self, X, y, n_splits=5):
        """Cross-validate ensemble model with TimeSeriesSplit."""
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        cv_scores = {
            'accuracy': cross_val_score(self.ensemble_model, X, y, cv=tscv, scoring='accuracy'),
            'roc_auc': cross_val_score(self.ensemble_model, X, y, cv=tscv, scoring='roc_auc'),
            'precision': cross_val_score(self.ensemble_model, X, y, cv=tscv, scoring='precision'),
            'recall': cross_val_score(self.ensemble_model, X, y, cv=tscv, scoring='recall'),
            'f1': cross_val_score(self.ensemble_model, X, y, cv=tscv, scoring='f1')
        }
        
        return cv_scores
    
    def get_feature_importance(self, X, label_name):
        """Get feature importance from trained model."""
        if self.feature_importance is not None:
            return self.feature_importance
        else:
            logger.warning("No feature importance available")
            return pd.DataFrame()
    
    def predict_ensemble(self, X):
        """Make ensemble predictions."""
        if self.ensemble_model is None:
            raise ValueError("No ensemble model trained")
        
        return self.ensemble_model.predict(X)
    
    def save_models(self, filepath):
        """Save trained models."""
        model_data = {
            'models': self.models,
            'ensemble_model': self.ensemble_model,
            'feature_importance': self.feature_importance
        }
        joblib.dump(model_data, filepath)
        logger.info(f"Models saved to {filepath}")
    
    def load_models(self, filepath):
        """Load trained models."""
        model_data = joblib.load(filepath)
        self.models = model_data['models']
        self.ensemble_model = model_data['ensemble_model']
        self.feature_importance = model_data['feature_importance']
        logger.info(f"Models loaded from {filepath}")


class LightGBMModel:
    """LightGBM model with proper time series validation."""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.feature_importance = None
        self.is_trained = False
        
    def prepare_features_and_labels(self, df: pd.DataFrame, label_column: str) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and labels with proper train/test separation."""
        # Remove timestamp and labels from features
        exclude_columns = ['timestamp']
        label_columns = [col for col in df.columns if col.startswith('label_')]
        return_columns = [col for col in df.columns if col.startswith('return_')]
        exclude_columns.extend(label_columns)
        exclude_columns.extend(return_columns)
        
        feature_columns = [col for col in df.columns if col not in exclude_columns]
        X = df[feature_columns].copy()
        y = df[label_column]
        
        # Ensure all features are numeric
        for col in X.columns:
            if X[col].dtype == 'object' or X[col].dtype == 'datetime64[ns]':
                X = X.drop(columns=[col])
        
        # Align data
        common_index = X.index.intersection(y.index)
        X = X.loc[common_index]
        y = y.loc[common_index]
        
        return X, y
    
    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series, 
                   X_val: pd.DataFrame, y_val: pd.Series,
                   task_type: str = "classification") -> lgb.LGBMClassifier:
        """Train LightGBM model with proper validation."""
        self.logger.info("Training LightGBM model")
        
        # Model parameters
        params = {
            'objective': 'binary' if task_type == "classification" else 'regression',
            'metric': 'binary_logloss' if task_type == "classification" else 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 64,
            'learning_rate': 0.03,
            'n_estimators': 1200,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 1,
            'reg_lambda': 3,
            'early_stopping_rounds': 100,
            'verbose': -1,
            'random_state': 42
        }
        
        # Create dataset
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        # Train model
        self.model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            callbacks=[lgb.log_evaluation(0)]
        )
        
        self.is_trained = True
        self.logger.info("LightGBM model training completed")
        
        return self.model
    
    def evaluate_model(self, X_test: pd.DataFrame, y_test: pd.Series, 
                      task_type: str = "classification") -> Dict:
        """Evaluate model performance."""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        predictions = self.model.predict(X_test)
        
        if task_type == "classification":
            # Convert probabilities to binary predictions
            binary_predictions = (predictions > 0.5).astype(int)
            
            metrics = {
                'accuracy': accuracy_score(y_test, binary_predictions),
                'roc_auc': roc_auc_score(y_test, predictions),
                'precision': precision_score(y_test, binary_predictions),
                'recall': recall_score(y_test, binary_predictions),
                'f1': f1_score(y_test, binary_predictions)
            }
        else:
            # Regression metrics
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            metrics = {
                'rmse': np.sqrt(mean_squared_error(y_test, predictions)),
                'mae': mean_absolute_error(y_test, predictions),
                'r2': r2_score(y_test, predictions)
            }
        
        return metrics
    
    def cross_validate(self, X: pd.DataFrame, y: pd.Series, 
                      task_type: str = "classification", cv_folds: int = 3) -> Dict:
        """Cross-validate model using TimeSeriesSplit."""
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        
        cv_scores = {
            'accuracy': cross_val_score(self.model, X, y, cv=tscv, scoring='accuracy'),
            'roc_auc': cross_val_score(self.model, X, y, cv=tscv, scoring='roc_auc'),
            'precision': cross_val_score(self.model, X, y, cv=tscv, scoring='precision'),
            'recall': cross_val_score(self.model, X, y, cv=tscv, scoring='recall'),
            'f1': cross_val_score(self.model, X, y, cv=tscv, scoring='f1')
        }
        
        return cv_scores
    
    def compute_shap_values(self, X: pd.DataFrame, sample_size: Optional[int] = None) -> Dict:
        """Compute SHAP values for model interpretability."""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        # Sample data if specified
        if sample_size and len(X) > sample_size:
            X_sample = X.sample(n=sample_size, random_state=42)
        else:
            X_sample = X
        
        # Compute SHAP values
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X_sample)
        
        # Create feature importance DataFrame
        feature_importance = pd.DataFrame({
            'feature': X_sample.columns,
            'importance': np.abs(shap_values).mean(0)
        }).sort_values('importance', ascending=False)
        
        return {
            'shap_values': shap_values,
            'feature_importance': feature_importance,
            'explainer': explainer
        }
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from trained model."""
        if self.model is None:
            return pd.DataFrame()
        
        importance = self.model.feature_importance(importance_type='gain')
        feature_names = self.model.feature_name()
        
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return feature_importance
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Make probability predictions."""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        return self.model.predict(X)
    
    def save_model(self, filepath: str) -> None:
        """Save trained model."""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        self.model.save_model(filepath)
        self.logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str, task_type: str = "classification") -> None:
        """Load trained model."""
        self.model = lgb.Booster(model_file=filepath)
        self.is_trained = True
        self.logger.info(f"Model loaded from {filepath}")


class ModelTrainer:
    """Model trainer with proper time series validation."""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.models = {}
        self.feature_engineers = {}
        self.validators = {}
        
    def train_single_model(self, df: pd.DataFrame, label_column: str, 
                          task_type: str = "classification",
                          test_size: float = 0.2, random_state: int = 42,
                          perform_cv: bool = True, compute_shap: bool = True) -> Dict:
        """Train a single model with proper time series validation."""
        self.logger.info(f"Training model for {label_column}")
        
        # Create feature engineer
        feature_engineer = FeatureEngineer(self.config)
        
        # Split data chronologically
        split_idx = int(len(df) * (1 - test_size))
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]
        
        self.logger.info(f"Train set: {len(train_df)}, Test set: {len(test_df)}")
        
        # Build features for training data
        X_train = feature_engineer.build_feature_matrix(train_df, fit_pipeline=True)
        y_train = train_df[label_column]
        
        # Build features for test data (no fitting)
        X_test = feature_engineer.build_feature_matrix(test_df, fit_pipeline=False)
        y_test = test_df[label_column]
        
        # Align data
        common_train_idx = X_train.index.intersection(y_train.index)
        X_train = X_train.loc[common_train_idx]
        y_train = y_train.loc[common_train_idx]
        
        common_test_idx = X_test.index.intersection(y_test.index)
        X_test = X_test.loc[common_test_idx]
        y_test = y_test.loc[common_test_idx]
        
        # Train model
        model = LightGBMModel(self.config)
        trained_model = model.train_model(X_train, y_train, X_test, y_test, task_type)
        
        # Evaluate model
        test_metrics = model.evaluate_model(X_test, y_test, task_type)
        
        # Cross-validation
        cv_scores = None
        if perform_cv:
            cv_scores = model.cross_validate(X_train, y_train, task_type)
        
        # SHAP analysis
        shap_results = None
        if compute_shap:
            shap_results = model.compute_shap_values(X_test)
        
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
    """Real-time predictor with proper feature pipeline."""
    
    def __init__(self, config: Config, trained_model: LightGBMModel, 
                 feature_engineer: FeatureEngineer):
        self.config = config
        self.model = trained_model
        self.feature_engineer = feature_engineer
        self.logger = logging.getLogger(__name__)
        
        # Check if feature engineer is ready
        if not self.feature_engineer.is_pipeline_ready():
            self.logger.warning("Feature engineer pipeline is not ready")
    
    def prepare_live_features(self, latest_data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for live prediction."""
        try:
            # Build live feature matrix
            feature_matrix = self.feature_engineer.build_live_feature_matrix(latest_data)
            
            # Remove timestamp and label columns
            exclude_columns = ['timestamp']
            label_columns = [col for col in feature_matrix.columns if col.startswith('label_')]
            return_columns = [col for col in feature_matrix.columns if col.startswith('return_')]
            exclude_columns.extend(label_columns)
            exclude_columns.extend(return_columns)
            
            feature_columns = [col for col in feature_matrix.columns if col not in exclude_columns]
            X = feature_matrix[feature_columns]
            
            # Ensure feature consistency with the fitted pipeline
            X = self.feature_engineer.ensure_feature_consistency(X)
            
            return X
            
        except Exception as e:
            self.logger.error(f"Error preparing live features: {e}")
            return pd.DataFrame()
    
    def predict(self, latest_data: pd.DataFrame) -> Dict:
        """Make real-time prediction."""
        try:
            # Prepare features
            X = self.prepare_live_features(latest_data)
            
            if X.empty:
                return {
                    'prediction': None,
                    'confidence': 0.0,
                    'error': 'No features available'
                }
            
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


def main():
    """Main function for model training."""
    from pathlib import Path
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    
    from src.utils import load_config, setup_logging
    from src.data import DataLoader
    from src.labels import LabelConstructor
    
    # Load configuration
    config = load_config("config/settings.yaml")
    setup_logging(config)
    
    # Load data
    loader = DataLoader(config)
    df = loader.load_gold_data("SOLUSDT", "1m")
    
    if df is None or df.empty:
        print("No data available for training")
        return
    
    # Construct labels
    label_constructor = LabelConstructor(config)
    df_with_labels = label_constructor.construct_all_labels(df)
    
    # Train model
    trainer = ModelTrainer(config)
    results = trainer.train_single_model(
        df_with_labels, 
        "label_class_1", 
        "classification",
        perform_cv=True,
        compute_shap=True
    )
    
    print(f"Training completed. Test accuracy: {results['test_metrics']['accuracy']:.4f}")


if __name__ == "__main__":
    main()