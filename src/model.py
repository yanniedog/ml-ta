#!/usr/bin/env python3
"""
Enhanced model training module with ensemble methods and advanced ML techniques.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, classification_report, precision_recall_fscore_support, mean_absolute_error
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import Ridge
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
    
    def split(self, X: pd.DataFrame, y: pd.Series = None, groups=None):
        """Generate indices to split data into training and test set."""
        self.logger.info(f"Generating {self.n_splits} splits for time series cross-validation.")
        total_rows = len(X)
        test_rows = int(total_rows * self.test_size)
        
        for i in range(self.n_splits):
            test_start = total_rows - (self.n_splits - i) * test_rows
            test_end = test_start + test_rows
            
            train_end = test_start
            
            # Ensure valid ranges
            test_start = max(0, test_start)
            test_end = min(total_rows, test_end)
            
            if test_start >= test_end or train_end <= 0:
                continue

            train_indices = np.arange(0, train_end)
            test_indices = np.arange(test_start, test_end)
            
            if len(train_indices) > 0 and len(test_indices) > 0:
                self.logger.info(f"Split {i+1}: Train size={len(train_indices)}, Test size={len(test_indices)}")
                yield train_indices, test_indices

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """Returns the number of splitting iterations in the cross-validator."""
        return self.n_splits


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

    def train_ensemble_model(self, X, y, label_name, task_type='classification', feature_engineer=None):
        """Train an ensemble model with multiple algorithms using proper validation."""
        logger.info(f"Training ensemble model for {label_name} with task type {task_type}")

        if feature_engineer is not None:
            self.feature_engineers[label_name] = feature_engineer

        # 1. Data Cleaning and Scaling
        X_clean = X.copy().replace([np.inf, -np.inf], np.nan).fillna(X.median())
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X_clean), columns=X_clean.columns, index=X_clean.index)
        self.scalers[label_name] = scaler

        # 2. Model Configuration
        n_splits = self.config.model.get('cv_folds', 5)
        test_size = self.config.model.get('cv_test_size', 0.2)
        tscv = TimeSeriesValidator(n_splits=n_splits, test_size=test_size)
        
        if task_type == 'classification':
            models_config = {
                'lgb': lgb.LGBMClassifier(random_state=42),
                'rf': RandomForestClassifier(n_estimators=100, random_state=42),
                'gb': GradientBoostingClassifier(n_estimators=100, random_state=42),
                'lr': LogisticRegression(random_state=42, solver='liblinear')
            }
            scoring = 'roc_auc'
            EnsembleModel = VotingClassifier(estimators=[(name, model) for name, model in models_config.items()], voting='soft')
        else: # regression
            models_config = {
                'lgb': lgb.LGBMRegressor(random_state=42),
                'rf': RandomForestRegressor(n_estimators=100, random_state=42),
                'gb': GradientBoostingRegressor(n_estimators=100, random_state=42),
                'ridge': Ridge(random_state=42)
            }
            scoring = 'neg_mean_squared_error'
            EnsembleModel = VotingRegressor(estimators=[(name, model) for name, model in models_config.items()])

        # 3. Train and Evaluate Individual Models
        individual_scores = {}
        for name, model in models_config.items():
            logger.info(f"Evaluating model: {name}")
            cv_scores = []
            for train_index, val_index in tscv.split(X_scaled):
                X_train, X_val = X_scaled.iloc[train_index], X_scaled.iloc[val_index]
                y_train, y_val = y.iloc[train_index], y.iloc[val_index]
                model.fit(X_train, y_train)
                if task_type == 'classification':
                    preds = model.predict_proba(X_val)[:, 1]
                    score = roc_auc_score(y_val, preds)
                else:
                    preds = model.predict(X_val)
                    score = -mean_absolute_error(y_val, preds) # Optuna maximizes, so negate MAE
                cv_scores.append(score)
            individual_scores[name] = {'mean_score': np.mean(cv_scores), 'std': np.std(cv_scores)}
            logger.info(f"  {name} CV {scoring}: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")

        # 4. Train Final Ensemble Model
        logger.info("Training final ensemble model...")
        EnsembleModel.fit(X_scaled, y)
        self.models[label_name] = EnsembleModel
        self.feature_importance[label_name] = {}

        # 5. Evaluate Ensemble
        ensemble_cv_scores = []
        for train_index, val_index in tscv.split(X_scaled):
            X_train, X_val = X_scaled.iloc[train_index], X_scaled.iloc[val_index]
            y_train, y_val = y.iloc[train_index], y.iloc[val_index]
            EnsembleModel.fit(X_train, y_train)
            if task_type == 'classification':
                preds = EnsembleModel.predict_proba(X_val)[:, 1]
                score = roc_auc_score(y_val, preds)
            else:
                preds = EnsembleModel.predict(X_val)
                score = -mean_absolute_error(y_val, preds) # Optuna maximizes, so negate MAE
            ensemble_cv_scores.append(score)
        final_scores = {
            f'ensemble_{scoring}_mean': np.mean(ensemble_cv_scores),
            f'ensemble_{scoring}_std': np.std(ensemble_cv_scores),
            'individual_scores': individual_scores
        }
        logger.info(f"Ensemble training complete. CV {scoring}: {final_scores[f'ensemble_{scoring}_mean']:.4f}")

        return final_scores

    def get_feature_importance(self, label_name):
        """Get feature importance from trained model."""
        return self.feature_importance.get(label_name)

    def predict_ensemble(self, X):
        """Make ensemble predictions."""
        if self.ensemble_model:
            return self.ensemble_model.predict(X)
        return None

    def save_models(self, filepath):
        """Save trained models and scalers."""
        with open(filepath, 'wb') as f:
            joblib.dump({
                'models': self.models,
                'ensemble_model': self.ensemble_model,
                'feature_importance': self.feature_importance,
                'scalers': self.scalers,
                'feature_engineers': self.feature_engineers
            }, f)

    def load_models(self, filepath):
        """Load trained models and scalers."""
        with open(filepath, 'rb') as f:
            data = joblib.load(f)
            self.models = data.get('models', {})
            self.ensemble_model = data.get('ensemble_model')
            self.feature_importance = data.get('feature_importance', {})
            self.scalers = data.get('scalers', {})
            self.feature_engineers = data.get('feature_engineers', {})


class HyperparameterOptimizer:
    """Hyperparameter optimizer using Optuna."""
    
    def __init__(self, model_class, config, X, y, task_type='classification', n_trials=50):
        self.model_class = model_class
        self.config = config
        self.X = X
        self.y = y
        self.task_type = task_type
        self.n_trials = n_trials
        self.logger = logging.getLogger(__name__)

    def _objective(self, trial):
        # This method should be implemented by subclasses for specific models
        raise NotImplementedError

    def optimize(self):
        self.logger.info(f"Starting hyperparameter optimization for {self.model_class.__name__}...")
        study = optuna.create_study(direction='maximize')
        study.optimize(self._objective, n_trials=self.n_trials)
        
        self.logger.info(f"Best hyperparameters: {study.best_params}")
        self.logger.info(f"Best score: {study.best_value:.4f}")
        
        return study.best_params


class LightGBMHyperparameterOptimizer(HyperparameterOptimizer):
    """Hyperparameter optimizer for LightGBM."""

    def _objective(self, trial):
        params = {
            'objective': 'binary' if self.task_type == 'classification' else 'regression_l1',
            'metric': 'roc_auc' if self.task_type == 'classification' else 'mae',
            'boosting_type': 'gbdt',
            'n_estimators': trial.suggest_int('n_estimators', 200, 2000, step=100),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 100, step=5),
            'max_depth': trial.suggest_int('max_depth', 4, 12),
            'min_child_samples': trial.suggest_int('min_child_samples', 20, 100, step=5),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'random_state': self.config.app.get('seed', 42),
            'verbose': -1
        }

        tscv = TimeSeriesSplit(n_splits=self.config.model.get('cv_folds', 3))
        scores = []

        for train_index, val_index in tscv.split(self.X):
            X_train, X_val = self.X.iloc[train_index], self.X.iloc[val_index]
            y_train, y_val = self.y.iloc[train_index], self.y.iloc[val_index]

            model = lgb.LGBMClassifier(**params) if self.task_type == 'classification' else lgb.LGBMRegressor(**params)
            
            model.fit(X_train, y_train, 
                      eval_set=[(X_val, y_val)], 
                      eval_metric='roc_auc' if self.task_type == 'classification' else 'mae',
                      callbacks=[lgb.early_stopping(50, verbose=False)])

            if self.task_type == 'classification':
                preds = model.predict_proba(X_val)[:, 1]
                score = roc_auc_score(y_val, preds)
            else:
                preds = model.predict(X_val)
                score = -mean_absolute_error(y_val, preds) # Optuna maximizes, so negate MAE
            
            scores.append(score)
        
        return np.mean(scores)


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
        """Train LightGBM model with proper validation and anti-overfitting measures."""
        self.logger.info("Training LightGBM model")
        
        # CRITICAL FIX: Anti-overfitting parameters
        params = {
            'objective': 'binary' if task_type == "classification" else 'regression',
            'metric': 'binary_logloss' if task_type == "classification" else 'rmse',
            'boosting_type': 'gbdt',
            # CRITICAL: Reduce model complexity to prevent overfitting
            'num_leaves': 31,  # Reduced from 64
            'learning_rate': 0.01,  # Reduced from 0.03
            'n_estimators': 500,  # Reduced from 1200
            'subsample': 0.7,  # Reduced from 0.8
            'colsample_bytree': 0.7,  # Reduced from 0.8
            # CRITICAL: Increase regularization
            'reg_alpha': 5,  # Increased from 1
            'reg_lambda': 10,  # Increased from 3
            'min_child_samples': 20,  # Added to prevent overfitting
            'min_child_weight': 1,  # Added to prevent overfitting
            'early_stopping_rounds': 50,  # Reduced from 100
            'verbose': -1,
            'random_state': 42
        }
        
        # CRITICAL: Use scikit-learn compatible classifier
        self.model = lgb.LGBMClassifier(**params)
        
        # CRITICAL: Add validation set for early stopping
        eval_set = [(X_val, y_val)] if len(X_val) > 0 else None
        
        # Train with validation
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )
        
        # CRITICAL: Validate model performance
        train_pred = self.model.predict(X_train)
        val_pred = self.model.predict(X_val)
        
        train_acc = accuracy_score(y_train, train_pred)
        val_acc = accuracy_score(y_val, val_pred)
        
        self.logger.info(f"Train accuracy: {train_acc:.4f}")
        self.logger.info(f"Validation accuracy: {val_acc:.4f}")
        
        # CRITICAL: Check for overfitting
        if train_acc - val_acc > 0.1:  # More than 10% difference indicates overfitting
            self.logger.warning(f"POTENTIAL OVERFITTING DETECTED: Train acc {train_acc:.4f} vs Val acc {val_acc:.4f}")
        
        self.is_trained = True
        self.logger.info("LightGBM model training completed")
        
        return self.model
    
    def evaluate_model(self, X_test: pd.DataFrame, y_test: pd.Series, 
                      task_type: str = "classification") -> Dict:
        """Evaluate model performance."""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        if task_type == "classification":
            # Get probability predictions for ROC AUC
            proba_predictions = self.model.predict_proba(X_test)[:, 1]
            binary_predictions = self.model.predict(X_test)
            
            metrics = {
                'accuracy': accuracy_score(y_test, binary_predictions),
                'roc_auc': roc_auc_score(y_test, proba_predictions),
                'precision': precision_score(y_test, binary_predictions),
                'recall': recall_score(y_test, binary_predictions),
                'f1': f1_score(y_test, binary_predictions)
            }
        else:
            # Regression metrics
            predictions = self.model.predict(X_test)
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            metrics = {
                'rmse': np.sqrt(mean_squared_error(y_test, predictions)),
                'mae': mean_absolute_error(y_test, predictions),
                'r2': r2_score(y_test, predictions)
            }
        
        return metrics
    
    def cross_validate(self, X: pd.DataFrame, y: pd.Series, 
                      task_type: str = "classification", cv_folds: int = 3, 
                      use_early_stopping: bool = True) -> Dict:
        """Advanced cross-validation with proper early stopping support."""
        self.logger.info(f"Performing {cv_folds}-fold time series cross-validation...")
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        
        # Prepare model parameters from config, removing early stopping for compatibility if needed
        params = self.config.model['params'].copy()
        params.pop('early_stopping_rounds', None)
        params['verbose'] = -1

        if task_type == 'classification':
            model = lgb.LGBMClassifier(**params)
            scoring = ['accuracy', 'roc_auc', 'precision', 'recall', 'f1']
        else:
            model = lgb.LGBMRegressor(**params)
            scoring = ['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error']

        # Manual cross-validation loop to properly handle early stopping
        if use_early_stopping and 'early_stopping_rounds' in self.config.model['params']:
            self.logger.info("Using manual cross-validation with early stopping.")
            metrics = {scorer: [] for scorer in scoring}
            
            for fold, (train_index, val_index) in enumerate(tscv.split(X)):
                X_train, X_val = X.iloc[train_index], X.iloc[val_index]
                y_train, y_val = y.iloc[train_index], y.iloc[val_index]
                
                fit_params = {
                    'eval_set': [(X_val, y_val)],
                    'eval_metric': 'logloss' if task_type == 'classification' else 'rmse',
                    'callbacks': [lgb.early_stopping(self.config.model['params']['early_stopping_rounds'], verbose=False)]
                }
                
                model.fit(X_train, y_train, **fit_params)
                y_pred = model.predict(X_val)
                
                if task_type == 'classification':
                    y_proba = model.predict_proba(X_val)[:, 1]
                    metrics['accuracy'].append(accuracy_score(y_val, y_pred))
                    metrics['roc_auc'].append(roc_auc_score(y_val, y_proba))
                    metrics['precision'].append(precision_score(y_val, y_pred, zero_division=0))
                    metrics['recall'].append(recall_score(y_val, y_pred, zero_division=0))
                    metrics['f1'].append(f1_score(y_val, y_pred, zero_division=0))
                else:
                    metrics['r2'].append(r2_score(y_val, y_pred))
                    metrics['neg_mean_squared_error'].append(-mean_squared_error(y_val, y_pred))
                    metrics['neg_mean_absolute_error'].append(-mean_absolute_error(y_val, y_pred))

            # Average the metrics
            cv_results = {f"test_{key}": np.mean(value) for key, value in metrics.items()}

        else:
            self.logger.info("Using sklearn.model_selection.cross_validate.")
            cv_results = cross_validate(model, X, y, cv=tscv, scoring=scoring, n_jobs=-1)

        # Log mean and std of scores
        self.logger.info("Cross-validation results:")
        for key, value in cv_results.items():
            if key.startswith('test_'):
                self.logger.info(f"  {key}: {np.mean(value):.4f} (+/- {np.std(value):.4f})")
        
        return cv_results
    
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
        
        importance = self.model.feature_importances_
        
        # Handle different LightGBM versions
        try:
            feature_names = self.model.feature_name_in
        except AttributeError:
            try:
                feature_names = self.model.feature_name
            except AttributeError:
                # Fallback to generic feature names
                feature_names = [f'feature_{i}' for i in range(len(importance))]
        
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
        
        return self.model.predict_proba(X)
    
    def save_model(self, filepath: str) -> None:
        """Save trained model."""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        import joblib
        joblib.dump(self.model, filepath)
        self.logger.info(f"Model saved to {filepath}")
    
    def optimize_hyperparameters(self, X, y, task_type='classification', n_trials=50):
        """Optimize hyperparameters for the LightGBM model."""
        self.logger.info(f"Optimizing hyperparameters for {task_type} task...")
        
        if task_type == 'classification':
            model_class = lgb.LGBMClassifier
        else:
            model_class = lgb.LGBMRegressor

        optimizer = LightGBMHyperparameterOptimizer(model_class, self.config, X, y, task_type, n_trials)
        best_params = optimizer.optimize()
        
        # Update the model's internal parameters with the best ones found
        self.config.model['params'].update(best_params)
        self.logger.info("Model parameters updated with optimized hyperparameters.")

        return best_params
    
    def load_model(self, filepath: str, task_type: str = "classification") -> None:
        """Load trained model."""
        import joblib
        self.model = joblib.load(filepath)
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
        
        # CRITICAL FIX: Ensure training data has all required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in train_df.columns]
        if missing_columns:
            self.logger.error(f"Missing required columns in training data: {missing_columns}")
            # Try to get original data from the combined dataset
            if hasattr(self, 'original_data') and self.original_data is not None:
                train_df = self.original_data.iloc[:len(train_df)]
                self.logger.info("Using original data for training")
            else:
                raise ValueError(f"Training data missing required columns: {missing_columns}")
        
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

