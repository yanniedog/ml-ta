"""
Main walkforward analysis script.
"""
import argparse
import logging
import sys
from pathlib import Path

from .data import BinanceDataFetcher
from .features import FeatureEngineer
from .labels import LabelConstructor
from .walkforward import WalkForwardAnalysis
from .report import ReportGenerator
from .utils import load_config, setup_logging, set_deterministic_seed


def main():
    """Main function for walkforward analysis."""
    parser = argparse.ArgumentParser(description="Technical Analysis Walkforward Analysis")
    parser.add_argument("--config", type=str, default="config/settings.yaml",
                       help="Path to configuration file")
    parser.add_argument("--skip-data", action="store_true",
                       help="Skip data fetching if data already exists")
    parser.add_argument("--skip-features", action="store_true",
                       help="Skip feature engineering if features already exist")
    parser.add_argument("--skip-labels", action="store_true",
                       help="Skip label construction if labels already exist")
    parser.add_argument("--skip-walkforward", action="store_true",
                       help="Skip walkforward analysis")
    parser.add_argument("--skip-reports", action="store_true",
                       help="Skip report generation")
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Setup logging
        logger = setup_logging(config)
        logger.info("Starting technical analysis walkforward pipeline")
        
        # Set deterministic seed
        set_deterministic_seed(config.app["seed"])
        
        # Step 1: Data fetching
        if not args.skip_data:
            logger.info("Step 1: Fetching data from Binance")
            fetcher = BinanceDataFetcher(config)
            fetcher.fetch_all_data()
        else:
            logger.info("Skipping data fetching")
        
        # Step 2: Feature engineering
        if not args.skip_features:
            logger.info("Step 2: Engineering features")
            engineer = FeatureEngineer(config)
            engineer.process_all_data()
        else:
            logger.info("Skipping feature engineering")
        
        # Step 3: Label construction
        if not args.skip_labels:
            logger.info("Step 3: Constructing labels")
            constructor = LabelConstructor(config)
            constructor.process_all_data()
        else:
            logger.info("Skipping label construction")
        
        # Step 4: Walkforward analysis
        if not args.skip_walkforward:
            logger.info("Step 4: Running walkforward analysis")
            walkforward = WalkForwardAnalysis(config)
            results = walkforward.run_all_analyses()
        else:
            logger.info("Skipping walkforward analysis")
            results = {}
        
        # Step 5: Report generation
        if not args.skip_reports and results:
            logger.info("Step 5: Generating reports")
            generator = ReportGenerator(config)
            report_files = generator.generate_all_reports(results)
            logger.info(f"Generated {len(report_files)} reports")
        else:
            logger.info("Skipping report generation")
        
        logger.info("Pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()