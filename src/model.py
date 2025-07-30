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
from collections import deque
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, recall_score, f1_score,
                             mean_squared_error, mean_absolute_error, r2_score)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, VotingClassifier, VotingRegressor
from optuna.integration import LightGBMPruningCallback
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
    """Advanced time series validation with expanding/sliding windows, gap, and purging."""

    def __init__(self, config: Config):
        self.n_splits = config.model.get('cv_folds', 5)
        self.test_size = config.model.get('cv_test_size', 0.2)
        self.validation_strategy = config.model.get('cv_strategy', 'expanding') # 'expanding' or 'sliding'
        self.gap_size = config.model.get('cv_gap_size', 0) # Number of samples to leave between train and test
        self.logger = logging.getLogger(__name__)

    def split(self, X: pd.DataFrame, y: pd.Series = None, groups=None):
        """Generate indices for time series cross-validation."""
        self.logger.info(f"Using '{self.validation_strategy}' window CV with {self.n_splits} splits, test_size={self.test_size}, gap={self.gap_size}.")
        
        total_samples = len(X)
        indices = np.arange(total_samples)
        
        test_size_samples = int(total_samples * self.test_size)
        if self.validation_strategy == 'expanding':
            train_size_samples = (total_samples - test_size_samples * self.n_splits) // self.n_splits
        else: # sliding
            train_size_samples = (total_samples - test_size_samples * self.n_splits - self.gap_size * (self.n_splits -1)) // self.n_splits

        for i in range(self.n_splits):
            if self.validation_strategy == 'expanding':
                train_end = i * (train_size_samples + test_size_samples)
                test_start = train_end + self.gap_size
            else: # sliding
                train_start = i * (train_size_samples + test_size_samples + self.gap_size)
                train_end = train_start + train_size_samples
                test_start = train_end + self.gap_size

            test_end = test_start + test_size_samples

            if test_end > total_samples:
                continue

            if self.validation_strategy == 'expanding':
                 train_indices = indices[:train_end]
            else: # sliding
                 train_indices = indices[train_start:train_end]

            test_indices = indices[test_start:test_end]

            if len(train_indices) > 0 and len(test_indices) > 0:
                self.logger.debug(f"Split {i+1}: Train {len(train_indices)} samples, Test {len(test_indices)} samples")
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

    def train_ensemble_model(self, df: pd.DataFrame, label_column: str, task_type: str, 
                             optimize_hyperparams: bool = False, ensemble_method: str = 'voting') -> Dict:
        """Train an ensemble of models with optional hyperparameter tuning."""
        self.logger.info(f"Starting ensemble model training for '{label_column}' (Task: {task_type}).")

        feature_engineer = FeatureEngineer(self.config)
        X, y = self._prepare_features(df, feature_engineer, label_column)
        self.feature_engineers[label_column] = feature_engineer

        X_train, X_test, y_train, y_test = self._time_series_split(X, y)

        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
        self.scalers[label_column] = scaler

        lgbm_params = self.config.model.get('lgbm_params', {})
        rf_params = {'n_estimators': 100, 'random_state': self.config.app['seed']}

        if optimize_hyperparams:
            self.logger.info("Optimizing hyperparameters for ensemble...")
            best_params = self._optimize_ensemble_hyperparameters(X_train_scaled, y_train, X_test_scaled, y_test, task_type)
            lgbm_params.update(best_params['lgbm'])
            rf_params.update(best_params['rf'])

        # Define base models
        if task_type == 'classification':
            estimators = [
                ('lgbm', lgb.LGBMClassifier(**lgbm_params)),
                ('rf', RandomForestClassifier(**rf_params))
            ]
            ensemble_model = VotingClassifier(estimators, voting='soft')
        else:
            estimators = [
                ('lgbm', lgb.LGBMRegressor(**lgbm_params)),
                ('rf', RandomForestRegressor(**rf_params))
            ]
            ensemble_model = VotingRegressor(estimators)

        self.logger.info("Fitting ensemble model...")
        ensemble_model.fit(X_train_scaled, y_train)

        metrics = self._evaluate_model(ensemble_model, X_test_scaled, y_test, task_type)
        self.logger.info(f"Ensemble test metrics for '{label_column}': {metrics}")

        result = {
            'model': ensemble_model,
            'test_metrics': metrics,
        }
        self.results[f"{label_column}_ensemble"] = result
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
        def objective(trial):
            params = {
                'objective': 'binary' if task_type == 'classification' else 'regression_l1',
                'metric': 'auc' if task_type == 'classification' else 'rmse',
                'n_estimators': trial.suggest_int('n_estimators', 200, 2000, step=100),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                'max_depth': trial.suggest_int('max_depth', 5, 15),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'verbose': -1,
                'random_state': self.config.app['seed']
            }

            model = self._train_lgbm(X_train, y_train, X_val, y_val, task_type, params)
            metrics = self._evaluate_model(model, X_val, y_val, task_type)
            
            return metrics.get('roc_auc' if task_type == 'classification' else 'rmse', 0)

        study = optuna.create_study(direction='maximize' if task_type == 'classification' else 'minimize')
        study.optimize(objective, n_trials=self.config.model.get('optuna_trials', 50))

        self.logger.info(f"Best trial score: {study.best_trial.value}")
        self.logger.info(f"Best params: {study.best_params}")

        return study.best_params

    def _optimize_ensemble_hyperparameters(self, X_train, y_train, X_val, y_val, task_type):
        """Optimize hyperparameters for ensemble models using Optuna."""
        def objective(trial):
            # LightGBM parameters
            lgbm_params = {
                'objective': 'binary' if task_type == 'classification' else 'regression_l1',
                'metric': 'auc' if task_type == 'classification' else 'rmse',
                'n_estimators': trial.suggest_int('lgbm_n_estimators', 200, 1000, step=100),
                'learning_rate': trial.suggest_float('lgbm_learning_rate', 0.01, 0.1, log=True),
                'num_leaves': trial.suggest_int('lgbm_num_leaves', 20, 50),
                'verbose': -1,
                'random_state': self.config.app['seed']
            }

            # RandomForest parameters
            rf_params = {
                'n_estimators': trial.suggest_int('rf_n_estimators', 100, 500, step=50),
                'max_depth': trial.suggest_int('rf_max_depth', 5, 15),
                'min_samples_split': trial.suggest_int('rf_min_samples_split', 2, 10),
                'random_state': self.config.app['seed']
            }

            if task_type == 'classification':
                estimators = [
                    ('lgbm', lgb.LGBMClassifier(**lgbm_params)),
                    ('rf', RandomForestClassifier(**rf_params))
                ]
                ensemble = VotingClassifier(estimators, voting='soft')
            else:
                estimators = [
                    ('lgbm', lgb.LGBMRegressor(**lgbm_params)),
                    ('rf', RandomForestRegressor(**rf_params))
                ]
                ensemble = VotingRegressor(estimators)

            pruning_callback = LightGBMPruningCallback(trial, 'auc' if task_type == 'classification' else 'rmse')
            fit_params = {
                'lgbm__eval_set': [(X_val, y_val)],
                'lgbm__callbacks': [pruning_callback]
            }

            try:
                ensemble.fit(X_train, y_train, **fit_params)
            except optuna.exceptions.TrialPruned:
                raise
            except Exception as e:
                # Catch other exceptions during fitting, e.g., from RandomForest
                self.logger.warning(f"Trial failed with exception: {e}")
                # Return a poor score to let Optuna continue with other trials
                return float('inf') if task_type == 'regression' else 0.0

            metrics = self._evaluate_model(ensemble, X_val, y_val, task_type)
            return metrics.get('roc_auc' if task_type == 'classification' else 'rmse', 0)

        study = optuna.create_study(direction='maximize' if task_type == 'classification' else 'minimize')
        study.optimize(objective, n_trials=self.config.model.get('optuna_trials_ensemble', 20))

        self.logger.info(f"Best ensemble trial score: {study.best_trial.value}")
        best_params = study.best_params
        
        # Separate params for each model
        lgbm_best_params = {k.replace('lgbm_', ''): v for k, v in best_params.items() if k.startswith('lgbm_')}
        rf_best_params = {k.replace('rf_', ''): v for k, v in best_params.items() if k.startswith('rf_')}
        
        return {'lgbm': lgbm_best_params, 'rf': rf_best_params}
    
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
    """Real-time predictor for a single model with history tracking."""

    def __init__(self, config: Config, model: object, feature_engineer: FeatureEngineer, history_size: int = 1000):
        self.config = config
        self.model = model
        self.feature_engineer = feature_engineer
        self.logger = logging.getLogger(__name__)
        self.prediction_history = deque(maxlen=history_size)

    def predict(self, data: pd.DataFrame, y_true: Optional[pd.Series] = None) -> Dict:
        """Make real-time prediction, store it, and return the result."""
        try:
            X = self.feature_engineer.build_live_feature_matrix(data)
            
            # Ensure columns match training columns
            if hasattr(self.model, 'feature_name_'):
                model_cols = self.model.feature_name_()
                X = X.reindex(columns=model_cols, fill_value=0)

            is_classification = hasattr(self.model, 'predict_proba')
            
            if is_classification:
                prediction_proba = self.model.predict_proba(X)
                prediction = self.model.predict(X)
                last_pred = int(prediction[-1])
                last_proba = prediction_proba[-1]
                confidence = float(np.max(last_proba))
                prob_value = float(last_proba[1]) if len(last_proba) > 1 else float(last_proba[0])
            else: # Regression
                prediction = self.model.predict(X)
                last_pred = float(prediction[-1])
                confidence = 1.0 # Confidence is less applicable for regression
                prob_value = last_pred

            result = {
                'prediction': last_pred,
                'confidence': confidence,
                'probability': prob_value
            }

            # Store history
            history_item = result.copy()
            if y_true is not None and not y_true.empty:
                history_item['actual'] = y_true.iloc[-1]
            self.prediction_history.append(history_item)

            return result

        except Exception as e:
            self.logger.error(f"Error in real-time prediction: {e}", exc_info=True)
            return {'prediction': None, 'confidence': 0.0, 'error': str(e)}

    def get_prediction_summary(self) -> Dict:
        """Get summary of recent predictions from history."""
        if not self.prediction_history:
            return {'total_predictions': 0, 'accuracy': 0.0, 'avg_confidence': 0.0}

        history = list(self.prediction_history)
        total_predictions = len(history)
        avg_confidence = np.mean([p['confidence'] for p in history])
        
        correct_predictions = 0
        actuals_available = 0
        for p in history:
            if 'actual' in p:
                actuals_available += 1
                if p['prediction'] == p['actual']:
                    correct_predictions += 1
        
        accuracy = (correct_predictions / actuals_available) if actuals_available > 0 else 0.0

        return {
            'total_predictions': total_predictions,
            'accuracy': accuracy,
            'avg_confidence': avg_confidence,
            'predictions_with_actuals': actuals_available
        }


class RealTimeEnsemblePredictor:
    """Real-time ensemble predictor that aggregates predictions from multiple models."""

    def __init__(self, config: Config, trained_models: Dict[str, Dict]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.predictors: Dict[str, RealTimePredictor] = {}

        for label, model_data in trained_models.items():
            if 'ensemble' not in label:
                self.predictors[label] = RealTimePredictor(
                    config,
                    model_data['model'],
                    model_data['feature_engineer']
                )

    def predict(self, data: pd.DataFrame, y_true: Optional[pd.Series] = None) -> Dict:
        """Make an aggregated real-time prediction from all base models."""
        all_predictions = []
        all_probas = []

        for label, predictor in self.predictors.items():
            result = predictor.predict(data, y_true)
            if result and result['prediction'] is not None:
                all_predictions.append(result['prediction'])
                all_probas.append(result['probability'])

        if not all_predictions:
            return {'prediction': None, 'confidence': 0.0, 'error': 'No valid predictions from base models.'}

        # Simple averaging for ensemble prediction
        final_prediction = np.mean(all_predictions)
        final_confidence = np.mean([p['confidence'] for p in self.get_all_histories()])

        # For classification, you might want to vote
        if self.config.model.get('task_type', 'classification') == 'classification':
            final_prediction = int(round(final_prediction))

        return {
            'prediction': final_prediction,
            'confidence': final_confidence,
            'individual_predictions': all_predictions
        }

    def get_all_histories(self) -> List[Dict]:
        """Retrieve prediction histories from all predictors."""
        histories = []
        for predictor in self.predictors.values():
            histories.extend(predictor.prediction_history)
        return histories

