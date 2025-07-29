# ML Trading Analysis System

A comprehensive technical analysis and machine learning system for cryptocurrency trading with enhanced features for real-time prediction, model interpretability, and performance optimization.

## 🚀 **Enhanced Features**

### **Core Improvements**
- ✅ **Fixed all test issues** - All tests now pass with proper floating-point comparisons
- ✅ **Eliminated dtype warnings** - Fixed all FutureWarning messages about dtype incompatibilities
- ✅ **Enhanced model evaluation** - Comprehensive metrics including ROC AUC, precision, recall, F1
- ✅ **Auto-detection of task types** - Automatically detects classification vs regression tasks
- ✅ **SHAP analysis** - Model interpretability with feature importance analysis
- ✅ **Performance optimization** - Added caching, vectorization, and memory optimization
- ✅ **Real-time prediction framework** - Live trading prediction capabilities

### **Technical Enhancements**

#### **Model Evaluation**
- **Comprehensive Metrics**: Accuracy, ROC AUC, precision, recall, F1-score for classification
- **Regression Metrics**: R², RMSE, MAE, MAPE for regression tasks
- **Cross-validation**: 5-fold CV with multiple scoring metrics
- **SHAP Analysis**: Model interpretability with feature importance rankings
- **Auto Task Detection**: Automatically detects classification vs regression based on label names

#### **Performance Optimization**
- **Memory Optimization**: DataFrame memory usage optimization
- **Caching System**: Function result caching for repeated calculations
- **Parallel Processing**: Joblib integration for parallel DataFrame processing
- **Numba Acceleration**: Fast rolling calculations with Numba JIT compilation
- **Performance Monitoring**: Decorators for execution time tracking

#### **Data Quality & Validation**
- **Data Quality Checks**: Missing values, duplicates, memory usage analysis
- **Feature Engineering**: 145+ technical indicators and derived features
- **Regime Detection**: Market regime flags for different market conditions
- **Feature Interactions**: Automated feature interaction creation

#### **Real-time Capabilities**
- **Live Prediction**: Real-time prediction engine for live trading
- **Scaler Persistence**: Fitted scaler preservation for consistent predictions
- **Prediction History**: Track prediction accuracy and confidence
- **Performance Summary**: Real-time trading performance metrics

## 📊 **System Architecture**

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Sources  │───▶│  Feature Engine │───▶│  Model Training │
│   (OHLCV)       │    │   (145+ features)│    │   (LightGBM)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │  Label Creation │    │   SHAP Analysis │
                       │ (Classification │    │ (Interpretability)│
                       │  & Regression)  │    └─────────────────┘
                       └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │  Backtesting   │
                       │ (Performance)  │
                       └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │ Real-time Pred │
                       │ (Live Trading) │
                       └─────────────────┘
```

## 🛠 **Installation**

```bash
# Clone the repository
git clone <repository-url>
cd ml-ta

# Install dependencies
pip install -r requirements.txt

# Run tests
python run_tests.py

# Run enhanced demo
python demo.py
```

## 📈 **Usage Examples**

### **Basic Usage**
```python
from src.utils import load_config
from src.model import ModelTrainer
from src.data import DataLoader

# Load configuration
config = load_config("config/settings.yaml")

# Load data
loader = DataLoader(config)
df = loader.load_gold_data("SOLUSDT", "1m")

# Train models with enhanced evaluation
trainer = ModelTrainer(config)
results = trainer.train_single_model(
    df, 
    "label_class_1", 
    "auto",  # Auto-detect task type
    perform_cv=True,
    compute_shap=True
)

# Access comprehensive results
print(f"Accuracy: {results['test_metrics']['accuracy']:.3f}")
print(f"ROC AUC: {results['test_metrics']['roc_auc']:.3f}")
print(f"Top Features: {results['feature_importance'].head(5)['feature'].tolist()}")
```

### **Real-time Prediction**
```python
from src.model import RealTimePredictor

# Create predictor with fitted scaler
predictor = RealTimePredictor(config, trained_model, fitted_scaler)

# Make real-time predictions
prediction = predictor.predict(latest_data)
print(f"Prediction: {prediction['prediction']}")
print(f"Confidence: {prediction['confidence']:.3f}")
```

### **Performance Optimization**
```python
from src.utils import performance_monitor, optimize_dataframe_memory

# Monitor function performance
@performance_monitor
def expensive_calculation(data):
    return process_data(data)

# Optimize DataFrame memory
optimized_df = optimize_dataframe_memory(large_df)
```

## 📋 **Requirements**

- Python 3.8+
- pandas >= 2.0.0
- numpy >= 1.24.0
- scikit-learn >= 1.3.0
- lightgbm >= 4.0.0
- shap >= 0.42.0
- optuna >= 3.2.0
- joblib >= 1.3.0
- numba >= 0.57.0

## 🧪 **Testing**

```bash
# Run all tests
python run_tests.py

# Run specific test file
pytest tests/test_indicators.py -v
```

## 📊 **Performance Metrics**

The system provides comprehensive evaluation metrics:

### **Classification Metrics**
- Accuracy, ROC AUC, Precision, Recall, F1-score
- Per-class metrics for multi-class problems
- Probability distribution analysis

### **Regression Metrics**
- R² Score, RMSE, MAE, MAPE
- Cross-validation scores
- Feature importance rankings

### **Trading Performance**
- Total Return, Sharpe Ratio, Max Drawdown
- Hit Rate, Total Trades
- Real-time prediction confidence

## 🔧 **Configuration**

The system uses YAML configuration files for easy customization:

```yaml
# config/settings.yaml
app:
  seed: 42
  log_level: INFO

model:
  params:
    n_estimators: 1000
    learning_rate: 0.1
    max_depth: 6
    early_stopping_rounds: 100

features:
  lags: [1, 2, 3, 5, 10]
  rolling_windows: [5, 10, 20]
```

## 🚀 **Future Enhancements**

- [ ] **Advanced Cross-validation**: Fix early stopping issues
- [ ] **Hyperparameter Tuning**: Optuna integration for automated tuning
- [ ] **Ensemble Methods**: Multiple model combination
- [ ] **Time Series Validation**: Proper time series CV
- [ ] **Web Dashboard**: Real-time monitoring interface
- [ ] **API Integration**: REST API for predictions
- [ ] **Database Integration**: Persistent model storage

## 📝 **Contributing**

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 **License**

This project is licensed under the MIT License - see the LICENSE file for details.