"""
Main script for training the model.
"""
import sys
from pathlib import Path

# Add project root to the Python path
sys.path.append(str(Path(__file__).parent))

from src.utils import load_config, setup_logging
from src.data import DataLoader
from src.labels import LabelConstructor
from src.model import AdvancedModelTrainer

def main():
    """Main function for model training."""
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
    
    # Prepare data for training
    label_column = "label_class_1"
    X = df_with_labels.drop(columns=[c for c in df_with_labels.columns if 'label' in c])
    y = df_with_labels[label_column]

    # Train model
    trainer = AdvancedModelTrainer(config)
    trainer.train_ensemble_model(X, y, label_column, task_type='classification')

    print(f"Ensemble model training completed for {label_column}.")


if __name__ == "__main__":
    main()
