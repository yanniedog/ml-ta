"""
Model module for LightGBM implementation with enhanced evaluation and SHAP analysis.
"""
import logging
from typing import Dict, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score, roc_auc_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from .utils import Config
from .indicators import TechnicalIndicators
from .features import FeatureEngineer


class LightGBMModel:
    """LightGBM model implementation for technical analysis with enhanced evaluation."""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.feature_importance = None
        self.validation_metrics = {}
        self.shap_values = None
        self.explainer = None
        
    def prepare_features_and_labels(self, df: pd.DataFrame, label_column: str) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and labels from DataFrame."""
        # Remove timestamp and label columns from features
        exclude_columns = ['timestamp', label_column]
        feature_columns = [col for col in df.columns if col not in exclude_columns]
        
        # Also exclude return columns (they contain future information)
        feature_columns = [col for col in feature_columns if not col.startswith('return_')]
        
        X = df[feature_columns]
        y = df[label_column]
        
        self.logger.info(f"Prepared {len(feature_columns)} features for {len(df)} samples")
        return X, y
    
    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series, 
                   X_val: pd.DataFrame, y_val: pd.Series,
                   task_type: str = "classification") -> lgb.LGBMClassifier:
        """Train LightGBM model with enhanced parameters."""
        self.logger.info(f"Training {task_type} model")
        
        # Get model parameters
        params = self.config.model["params"].copy()
        
        if task_type == "classification":
            model = lgb.LGBMClassifier(**params)
        else:
            # For regression, change objective
            params['objective'] = 'regression'
            params['metric'] = 'rmse'
            model = lgb.LGBMRegressor(**params)
        
        # Train model
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=params['early_stopping_rounds'])]
        )
        
        self.model = model
        self.logger.info("Model training completed")
        return model
    
    def evaluate_model(self, X_test: pd.DataFrame, y_test: pd.Series, 
                      task_type: str = "classification") -> Dict:
        """Evaluate model performance with comprehensive metrics."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test) if task_type == "classification" else None
        
        # Calculate metrics
        metrics = {}
        
        if task_type == "classification":
            metrics['accuracy'] = accuracy_score(y_test, y_pred)
            metrics['classification_report'] = classification_report(y_test, y_pred, output_dict=True)
            
            # Additional classification metrics
            if y_pred_proba is not None:
                metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba[:, 1])
                metrics['positive_probability_mean'] = y_pred_proba[:, 1].mean()
                metrics['positive_probability_std'] = y_pred_proba[:, 1].std()
            
            # Precision, Recall, F1 for each class
            precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, average=None)
            metrics['precision_per_class'] = precision.tolist()
            metrics['recall_per_class'] = recall.tolist()
            metrics['f1_per_class'] = f1.tolist()
            metrics['support_per_class'] = support.tolist()
        
        else:  # regression
            metrics['mse'] = mean_squared_error(y_test, y_pred)
            metrics['rmse'] = np.sqrt(metrics['mse'])
            metrics['r2'] = r2_score(y_test, y_pred)
            metrics['mae'] = np.mean(np.abs(y_test - y_pred))
            metrics['mape'] = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        
        self.logger.info(f"Model evaluation completed: {metrics}")
        return metrics
    
    def cross_validate(self, X: pd.DataFrame, y: pd.Series, 
                      task_type: str = "classification", cv_folds: int = 5) -> Dict:
        """Perform cross-validation with comprehensive metrics."""
        self.logger.info(f"Performing {cv_folds}-fold cross-validation")
        
        if task_type == "classification":
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            scoring = ['accuracy', 'roc_auc', 'precision', 'recall', 'f1']
        else:
            cv = cv_folds
            scoring = ['neg_mean_squared_error', 'r2', 'neg_mean_absolute_error']
        
        cv_scores = {}
        for metric in scoring:
            try:
                # Create a copy of the model without early stopping for CV
                if hasattr(self.model, 'get_params'):
                    model_params = self.model.get_params()
                    # Remove early stopping parameters for CV
                    model_params.pop('callbacks', None)
                    model_params.pop('eval_set', None)
                    
                    if task_type == "classification":
                        cv_model = lgb.LGBMClassifier(**model_params)
                    else:
                        cv_model = lgb.LGBMRegressor(**model_params)
                    
                    scores = cross_val_score(cv_model, X, y, cv=cv, scoring=metric)
                    cv_scores[f'cv_{metric}'] = {
                        'mean': scores.mean(),
                        'std': scores.std(),
                        'scores': scores.tolist()
                    }
                else:
                    self.logger.warning(f"Could not perform CV for metric {metric}")
            except Exception as e:
                self.logger.warning(f"CV failed for metric {metric}: {e}")
        
        self.logger.info(f"Cross-validation completed: {cv_scores}")
        return cv_scores
    
    def compute_shap_values(self, X: pd.DataFrame, sample_size: Optional[int] = None) -> Dict:
        """Compute SHAP values for model interpretability."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        self.logger.info("Computing SHAP values")
        
        # Sample data if specified
        if sample_size and len(X) > sample_size:
            X_sample = X.sample(n=sample_size, random_state=42)
        else:
            X_sample = X
        
        # Create explainer
        self.explainer = shap.TreeExplainer(self.model)
        self.shap_values = self.explainer.shap_values(X_sample)
        
        # Calculate feature importance from SHAP
        if isinstance(self.shap_values, list):
            # For classification, use the positive class SHAP values
            shap_importance = np.abs(self.shap_values[1]).mean(axis=0)
        else:
            shap_importance = np.abs(self.shap_values).mean(axis=0)
        
        feature_importance_df = pd.DataFrame({
            'feature': X_sample.columns,
            'shap_importance': shap_importance
        }).sort_values('shap_importance', ascending=False)
        
        self.logger.info(f"SHAP analysis completed. Top features: {feature_importance_df.head(10)['feature'].tolist()}")
        
        return {
            'shap_values': self.shap_values,
            'explainer': self.explainer,
            'feature_importance': feature_importance_df,
            'expected_value': self.explainer.expected_value
        }
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from trained model."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        importance = self.model.feature_importances_
        feature_names = self.model.feature_name_
        
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        self.feature_importance = feature_importance_df
        return feature_importance_df
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using trained model."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Make probability predictions using trained model."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            raise ValueError("Model does not support probability predictions")
    
    def save_model(self, filepath: str) -> None:
        """Save trained model to file."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        self.model.booster_.save_model(filepath)
        self.logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str, task_type: str = "classification") -> None:
        """Load trained model from file."""
        if task_type == "classification":
            self.model = lgb.LGBMClassifier()
        else:
            self.model = lgb.LGBMRegressor()
        
        self.model.booster_ = lgb.Booster(model_file=filepath)
        self.logger.info(f"Model loaded from {filepath}")


class ModelTrainer:
    """Enhanced model trainer with hyperparameter tuning and comprehensive evaluation."""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_score = 0
    
    def train_single_model(self, df: pd.DataFrame, label_column: str, 
                          task_type: str = "classification",
                          test_size: float = 0.2, random_state: int = 42,
                          perform_cv: bool = True, compute_shap: bool = True) -> Dict:
        """Train a single model with comprehensive evaluation."""
        self.logger.info(f"Training model for {label_column} ({task_type})")
        
        # Prepare features and labels
        X, y = self.prepare_features_and_labels(df, label_column)
        
        # Auto-detect task type if not specified
        if task_type == "auto":
            if label_column.startswith('label_reg'):
                task_type = "regression"
            elif label_column.startswith('label_class'):
                task_type = "classification"
            else:
                # Check if target is binary
                unique_values = y.nunique()
                if unique_values == 2:
                    task_type = "classification"
                else:
                    task_type = "regression"
        
        # For regression tasks, ensure we have enough data
        if task_type == "regression" and len(y) < 10:
            self.logger.warning(f"Insufficient data for regression: {len(y)} samples")
            return {}
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, 
            stratify=y if task_type == "classification" and y.nunique() > 1 else None
        )
        
        # Further split training data for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=random_state,
            stratify=y_train if task_type == "classification" and y_train.nunique() > 1 else None
        )
        
        # Create and train model
        model = LightGBMModel(self.config)
        trained_model = model.train_model(X_train, y_train, X_val, y_val, task_type)
        
        # Evaluate on test set
        test_metrics = model.evaluate_model(X_test, y_test, task_type)
        
        # Perform cross-validation if requested
        cv_results = None
        if perform_cv:
            cv_results = model.cross_validate(X, y, task_type)
        
        # Compute SHAP values if requested
        shap_results = None
        if compute_shap:
            shap_results = model.compute_shap_values(X, sample_size=min(1000, len(X)))
        
        # Get feature importance
        feature_importance = model.get_feature_importance()
        
        # Store results
        results = {
            'model': model,
            'label_column': label_column,
            'task_type': task_type,
            'test_metrics': test_metrics,
            'cv_results': cv_results,
            'shap_results': shap_results,
            'feature_importance': feature_importance,
            'data_shape': X.shape,
            'train_shape': X_train.shape,
            'val_shape': X_val.shape,
            'test_shape': X_test.shape
        }
        
        self.models[label_column] = model
        self.results[label_column] = results
        
        # Update best model
        if task_type == "classification":
            score = test_metrics.get('accuracy', 0)
        else:
            score = test_metrics.get('r2', 0)
        
        if score > self.best_score:
            self.best_score = score
            self.best_model = model
        
        self.logger.info(f"Model training completed for {label_column}")
        return results
    
    def hyperparameter_tuning(self, df: pd.DataFrame, label_column: str,
                            task_type: str = "classification", n_trials: int = 50) -> Dict:
        """Perform hyperparameter tuning using Optuna."""
        try:
            import optuna
        except ImportError:
            self.logger.warning("Optuna not available. Skipping hyperparameter tuning.")
            return {}
        
        self.logger.info(f"Starting hyperparameter tuning for {label_column}")
        
        X, y = self.prepare_features_and_labels(df, label_column)
        
        def objective(trial):
            # Define hyperparameter search space
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'num_leaves': trial.suggest_int('num_leaves', 10, 100),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'random_state': 42,
                'verbose': -1
            }
            
            if task_type == "classification":
                model = lgb.LGBMClassifier(**params)
                scoring = 'accuracy'
            else:
                model = lgb.LGBMRegressor(**params)
                scoring = 'r2'
            
            # Cross-validation score
            cv_scores = cross_val_score(model, X, y, cv=5, scoring=scoring)
            return cv_scores.mean()
        
        # Create study and optimize
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        # Train model with best parameters
        best_params = study.best_params
        best_params.update({'random_state': 42, 'verbose': -1})
        
        self.logger.info(f"Best parameters: {best_params}")
        self.logger.info(f"Best CV score: {study.best_value}")
        
        return {
            'best_params': best_params,
            'best_score': study.best_value,
            'study': study
        }
    
    def prepare_features_and_labels(self, df: pd.DataFrame, label_column: str) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and labels from DataFrame."""
        # Remove timestamp and label columns from features
        exclude_columns = ['timestamp', label_column]
        feature_columns = [col for col in df.columns if col not in exclude_columns]
        
        # Also exclude return columns (they contain future information)
        feature_columns = [col for col in feature_columns if not col.startswith('return_')]
        
        X = df[feature_columns]
        y = df[label_column]
        
        self.logger.info(f"Prepared {len(feature_columns)} features for {len(df)} samples")
        return X, y
    
    def train_all_models(self, df: pd.DataFrame) -> Dict:
        """Train models for all label columns."""
        self.logger.info("Training all models")
        
        # Find all label columns
        label_columns = [col for col in df.columns if col.startswith('label_')]
        
        if not label_columns:
            self.logger.error("No label columns found in DataFrame")
            return {}
        
        all_results = {}
        
        for label_column in label_columns:
            try:
                # Determine task type
                task_type = "classification" if "class" in label_column else "regression"
                
                # Train model
                results = self.train_single_model(df, label_column, task_type)
                all_results[label_column] = results
                
            except Exception as e:
                self.logger.error(f"Error training model for {label_column}: {e}")
                continue
        
        self.logger.info(f"Completed training {len(all_results)} models")
        return all_results
    
    def get_best_model(self, metric: str = "accuracy") -> Tuple[str, LightGBMModel]:
        """Get the best model based on validation metric."""
        if not self.models:
            raise ValueError("No models trained yet")
        
        best_score = -np.inf
        best_model_key = None
        
        for label_column, results in self.results.items():
            val_metrics = results['val_metrics']
            
            if metric in val_metrics:
                score = val_metrics[metric]
                if score > best_score:
                    best_score = score
                    best_model_key = label_column
        
        if best_model_key is None:
            raise ValueError(f"Metric {metric} not found in any model results")
        
        return best_model_key, self.models[best_model_key]
    
    def save_all_models(self, base_path: str) -> None:
        """Save all trained models."""
        for label_column, model in self.models.items():
            filepath = f"{base_path}/{label_column}_model.txt"
            model.save_model(filepath)
    
    def get_summary_report(self) -> pd.DataFrame:
        """Get summary report of all model performances."""
        summary_data = []
        
        for label_column, results in self.results.items():
            val_metrics = results['val_metrics']
            test_metrics = results['test_metrics']
            
            row = {
                'label_column': label_column,
                'task_type': 'classification' if 'class' in label_column else 'regression'
            }
            
            # Add validation metrics
            for metric, value in val_metrics.items():
                if isinstance(value, (int, float)):
                    row[f'val_{metric}'] = value
            
            # Add test metrics
            for metric, value in test_metrics.items():
                if isinstance(value, (int, float)):
                    row[f'test_{metric}'] = value
            
            summary_data.append(row)
        
        return pd.DataFrame(summary_data)


class RealTimePredictor:
    """Real-time prediction engine for live trading."""
    
    def __init__(self, config: Config, trained_model: LightGBMModel, fitted_scaler=None):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.model = trained_model
        self.feature_columns = None
        self.fitted_scaler = fitted_scaler
        self.last_prediction = None
        self.prediction_history = []
        
    def prepare_live_features(self, latest_data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for real-time prediction."""
        # Calculate indicators for the latest data
        indicators = TechnicalIndicators(self.config)
        df_with_indicators = indicators.calculate_all_indicators(latest_data)
        
        # Engineer features without fitting scaler
        engineer = FeatureEngineer(self.config)
        feature_df = engineer.build_feature_matrix(df_with_indicators, fit_scaler=False)
        
        # Get the latest row for prediction
        latest_features = feature_df.iloc[-1:].copy()
        
        # Store feature columns for consistency
        if self.feature_columns is None:
            self.feature_columns = [col for col in latest_features.columns 
                                  if col not in ['timestamp'] and not col.startswith('label_')]
        
        # Ensure we have the same features as training
        latest_features = latest_features[self.feature_columns]
        
        # Apply scaling if we have a fitted scaler
        if self.fitted_scaler is not None:
            latest_features = pd.DataFrame(
                self.fitted_scaler.transform(latest_features),
                columns=latest_features.columns,
                index=latest_features.index
            )
        
        return latest_features
    
    def predict(self, latest_data: pd.DataFrame) -> Dict:
        """Make real-time prediction."""
        try:
            # Prepare features
            features = self.prepare_live_features(latest_data)
            
            # Make prediction
            prediction = self.model.predict(features)
            probability = self.model.predict_proba(features) if hasattr(self.model.model, 'predict_proba') else None
            
            # Store prediction
            result = {
                'timestamp': latest_data['timestamp'].iloc[-1],
                'prediction': prediction[0],
                'probability': probability[0] if probability is not None else None,
                'confidence': max(probability[0]) if probability is not None else None
            }
            
            self.last_prediction = result
            self.prediction_history.append(result)
            
            self.logger.info(f"Real-time prediction: {result}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in real-time prediction: {e}")
            return None
    
    def get_prediction_summary(self, window: int = 100) -> Dict:
        """Get summary of recent predictions."""
        if not self.prediction_history:
            return {}
        
        recent_predictions = self.prediction_history[-window:]
        
        summary = {
            'total_predictions': len(recent_predictions),
            'buy_signals': sum(1 for p in recent_predictions if p['prediction'] == 1),
            'sell_signals': sum(1 for p in recent_predictions if p['prediction'] == 0),
            'avg_confidence': np.mean([p['confidence'] for p in recent_predictions if p['confidence'] is not None]),
            'last_prediction': self.last_prediction
        }
        
        return summary


def main():
    """Main function for model training."""
    from .utils import load_config, setup_logging
    
    # Load configuration
    config = load_config("config/settings.yaml")
    
    # Setup logging
    logger = setup_logging(config)
    
    # Load data
    from .data import DataLoader
    loader = DataLoader(config)
    df = loader.load_gold_data("SOLUSDT", "1m")
    
    if df is None or df.empty:
        logger.error("No data available for training")
        return
    
    # Train models
    trainer = ModelTrainer(config)
    label_columns = [col for col in df.columns if col.startswith('label_')]
    
    for label_col in label_columns[:3]:  # Train first 3 models
        try:
            results = trainer.train_single_model(df, label_col, "classification")
            logger.info(f"Trained model for {label_col}")
        except Exception as e:
            logger.error(f"Error training model for {label_col}: {e}")


if __name__ == "__main__":
    main()