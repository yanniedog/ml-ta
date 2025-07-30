"""
Report generation module for technical analysis results.
"""
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
from jinja2 import Template

from src.utils import Config, ensure_directory, get_config_hash, format_timestamp


class ReportGenerator:
    """Generate comprehensive HTML reports with charts and analysis."""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Set style for plots
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def generate_equity_chart(self, equity_curve: List[float], title: str = "Equity Curve") -> str:
        """Generate equity curve chart."""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            y=equity_curve,
            mode='lines',
            name='Equity',
            line=dict(color='blue', width=2)
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Time",
            yaxis_title="Equity ($)",
            template="plotly_white",
            height=400
        )
        
        return fig.to_html(include_plotlyjs='cdn')
    
    def generate_drawdown_chart(self, equity_curve: List[float], title: str = "Drawdown") -> str:
        """Generate drawdown chart."""
        equity_series = pd.Series(equity_curve)
        rolling_max = equity_series.expanding().max()
        drawdown = (equity_series - rolling_max) / rolling_max * 100
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            y=drawdown,
            mode='lines',
            name='Drawdown',
            line=dict(color='red', width=2),
            fill='tonexty'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Time",
            yaxis_title="Drawdown (%)",
            template="plotly_white",
            height=400
        )
        
        return fig.to_html(include_plotlyjs='cdn')
    
    def generate_feature_importance_chart(self, feature_importance: pd.Series, 
                                       title: str = "Feature Importance", 
                                       top_n: int = 20) -> str:
        """Generate feature importance chart."""
        top_features = feature_importance.head(top_n)
        
        fig = go.Figure(data=[
            go.Bar(
                x=top_features.values,
                y=top_features.index,
                orientation='h',
                marker_color='lightblue'
            )
        ])
        
        fig.update_layout(
            title=title,
            xaxis_title="Importance",
            yaxis_title="Feature",
            template="plotly_white",
            height=max(400, len(top_features) * 20)
        )
        
        return fig.to_html(include_plotlyjs='cdn')
    
    def generate_performance_metrics_table(self, metrics: Dict) -> str:
        """Generate performance metrics table."""
        # Format metrics for display
        formatted_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, float):
                if 'return' in key.lower() or 'rate' in key.lower():
                    formatted_metrics[key] = f"{value:.2%}"
                elif 'ratio' in key.lower():
                    formatted_metrics[key] = f"{value:.3f}"
                else:
                    formatted_metrics[key] = f"{value:.2f}"
            else:
                formatted_metrics[key] = str(value)
        
        # Create HTML table
        table_html = "<table class='metrics-table'>"
        table_html += "<tr><th>Metric</th><th>Value</th></tr>"
        
        for metric, value in formatted_metrics.items():
            table_html += f"<tr><td>{metric.replace('_', ' ').title()}</td><td>{value}</td></tr>"
        
        table_html += "</table>"
        return table_html
    
    def generate_trades_table(self, trades: List[Dict], max_rows: int = 10) -> str:
        """Generate trades table."""
        if not trades:
            return "<p>No trades executed</p>"
        
        trades_df = pd.DataFrame(trades)
        
        # Format for display
        display_df = trades_df.copy()
        display_df['entry_time'] = pd.to_datetime(display_df['entry_time']).dt.strftime('%Y-%m-%d %H:%M')
        display_df['exit_time'] = pd.to_datetime(display_df['exit_time']).dt.strftime('%Y-%m-%d %H:%M')
        display_df['net_pnl'] = display_df['net_pnl'].apply(lambda x: f"${x:.2f}")
        display_df['notional'] = display_df['notional'].apply(lambda x: f"${x:.2f}")
        
        # Select columns to display
        display_columns = ['entry_time', 'exit_time', 'entry_price', 'exit_price', 'net_pnl', 'notional']
        display_df = display_df[display_columns].head(max_rows)
        
        # Create HTML table
        table_html = "<table class='trades-table'>"
        table_html += "<tr><th>Entry Time</th><th>Exit Time</th><th>Entry Price</th><th>Exit Price</th><th>P&L</th><th>Notional</th></tr>"
        
        for _, row in display_df.iterrows():
            table_html += f"<tr><td>{row['entry_time']}</td><td>{row['exit_time']}</td><td>${row['entry_price']:.2f}</td><td>${row['exit_price']:.2f}</td><td>{row['net_pnl']}</td><td>{row['notional']}</td></tr>"
        
        table_html += "</table>"
        
        if len(trades) > max_rows:
            table_html += f"<p><em>Showing first {max_rows} trades of {len(trades)} total trades</em></p>"
        
        return table_html
    
    def generate_html_report(self, results: Dict, symbol: str, interval: str, 
                           label_column: str, config: Config) -> str:
        """Generate comprehensive HTML report."""
        timestamp = format_timestamp()
        config_hash = get_config_hash(config)
        
        # Extract data from results
        backtest_results = results.get('backtest_results', {})
        performance = backtest_results.get('performance', {})
        trades = backtest_results.get('trades', [])
        equity_curve = backtest_results.get('equity_curve', [])
        feature_importance = results.get('feature_importance', pd.Series())
        
        # Generate charts
        equity_chart = self.generate_equity_chart(equity_curve, f"{symbol} {interval} Equity Curve")
        drawdown_chart = self.generate_drawdown_chart(equity_curve, f"{symbol} {interval} Drawdown")
        feature_chart = self.generate_feature_importance_chart(feature_importance, f"{symbol} {interval} Feature Importance")
        
        # Generate tables
        metrics_table = self.generate_performance_metrics_table(performance)
        trades_table = self.generate_trades_table(trades)
        
        # HTML template
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Technical Analysis Report - {{symbol}} {{interval}}</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .header { background-color: #f0f0f0; padding: 20px; border-radius: 5px; }
                .section { margin: 20px 0; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }
                .metrics-table, .trades-table { width: 100%; border-collapse: collapse; margin: 10px 0; }
                .metrics-table th, .metrics-table td, .trades-table th, .trades-table td { 
                    border: 1px solid #ddd; padding: 8px; text-align: left; 
                }
                .metrics-table th, .trades-table th { background-color: #f2f2f2; }
                .chart-container { margin: 20px 0; }
                .summary { background-color: #e8f4f8; padding: 15px; border-radius: 5px; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Technical Analysis Report</h1>
                <p><strong>Symbol:</strong> {{symbol}} | <strong>Interval:</strong> {{interval}} | <strong>Label:</strong> {{label_column}}</p>
                <p><strong>Generated:</strong> {{timestamp}} | <strong>Config Hash:</strong> {{config_hash}}</p>
            </div>
            
            <div class="section">
                <h2>Summary</h2>
                <div class="summary">
                    <p><strong>Total Return:</strong> {{total_return}}</p>
                    <p><strong>Sharpe Ratio:</strong> {{sharpe_ratio}}</p>
                    <p><strong>Max Drawdown:</strong> {{max_drawdown}}</p>
                    <p><strong>Hit Rate:</strong> {{hit_rate}}</p>
                    <p><strong>Total Trades:</strong> {{total_trades}}</p>
                </div>
            </div>
            
            <div class="section">
                <h2>Performance Metrics</h2>
                {{metrics_table}}
            </div>
            
            <div class="section">
                <h2>Equity Curve</h2>
                <div class="chart-container">
                    {{equity_chart}}
                </div>
            </div>
            
            <div class="section">
                <h2>Drawdown Analysis</h2>
                <div class="chart-container">
                    {{drawdown_chart}}
                </div>
            </div>
            
            <div class="section">
                <h2>Feature Importance</h2>
                <div class="chart-container">
                    {{feature_chart}}
                </div>
            </div>
            
            <div class="section">
                <h2>Recent Trades</h2>
                {{trades_table}}
            </div>
            
            <div class="section">
                <h2>Configuration</h2>
                <pre>{{config_yaml}}</pre>
            </div>
        </body>
        </html>
        """
        
        # Render template
        template = Template(html_template)
        html_content = template.render(
            symbol=symbol,
            interval=interval,
            label_column=label_column,
            timestamp=timestamp,
            config_hash=config_hash,
            total_return=performance.get('total_return', 0),
            sharpe_ratio=performance.get('sharpe_ratio', 0),
            max_drawdown=performance.get('max_drawdown', 0),
            hit_rate=performance.get('hit_rate', 0),
            total_trades=performance.get('total_trades', 0),
            metrics_table=metrics_table,
            equity_chart=equity_chart,
            drawdown_chart=drawdown_chart,
            feature_chart=feature_chart,
            trades_table=trades_table,
            config_yaml=str(config.dict())
        )
        
        return html_content
    
    def save_report(self, html_content: str, symbol: str, interval: str, 
                   label_column: str, output_dir: str) -> str:
        """Save HTML report to file."""
        ensure_directory(output_dir)
        
        timestamp = format_timestamp()
        filename = f"{symbol}_{interval}_{label_column}_report_{timestamp}.html"
        filepath = f"{output_dir}/{filename}"
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        self.logger.info(f"Saved report to {filepath}")
        return filepath
    
    def generate_all_reports(self, results: Dict) -> List[str]:
        """Generate reports for all results."""
        self.logger.info("Generating reports for all results")
        
        report_files = []
        
        for key, result in results.items():
            try:
                # Parse key to get symbol, interval, label_column
                parts = key.split('_')
                if len(parts) >= 3:
                    symbol = parts[0]
                    interval = parts[1]
                    label_column = '_'.join(parts[2:])
                    
                    # Generate HTML report
                    html_content = self.generate_html_report(result, symbol, interval, label_column, self.config)
                    
                    # Save report
                    timestamp = format_timestamp()
                    output_dir = f"{self.config.paths['artefacts']}/reports_{timestamp}"
                    report_file = self.save_report(html_content, symbol, interval, label_column, output_dir)
                    report_files.append(report_file)
                    
                else:
                    self.logger.warning(f"Could not parse key: {key}")
                    
            except Exception as e:
                self.logger.error(f"Error generating report for {key}: {e}")
                continue
        
        self.logger.info(f"Generated {len(report_files)} reports")
        return report_files
    
    def generate_summary_report(self, all_results: Dict) -> str:
        """Generate summary report of all results."""
        self.logger.info("Generating summary report")
        
        summary_data = []
        
        for key, result in all_results.items():
            if result and 'backtest_results' in result:
                performance = result['backtest_results'].get('performance', {})
                
                # Parse key
                parts = key.split('_')
                if len(parts) >= 3:
                    symbol = parts[0]
                    interval = parts[1]
                    label_column = '_'.join(parts[2:])
                    
                    summary_data.append({
                        'symbol': symbol,
                        'interval': interval,
                        'label_column': label_column,
                        'total_return': performance.get('total_return', 0),
                        'sharpe_ratio': performance.get('sharpe_ratio', 0),
                        'max_drawdown': performance.get('max_drawdown', 0),
                        'hit_rate': performance.get('hit_rate', 0),
                        'total_trades': performance.get('total_trades', 0)
                    })
        
        if not summary_data:
            return "<p>No results to summarize</p>"
        
        # Create summary DataFrame
        summary_df = pd.DataFrame(summary_data)
        
        # Generate summary chart
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Total Return', 'Sharpe Ratio', 'Max Drawdown', 'Hit Rate'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        # Total Return
        fig.add_trace(
            go.Bar(x=summary_df['symbol'] + ' ' + summary_df['interval'], 
                   y=summary_df['total_return'], name='Total Return'),
            row=1, col=1
        )
        
        # Sharpe Ratio
        fig.add_trace(
            go.Bar(x=summary_df['symbol'] + ' ' + summary_df['interval'], 
                   y=summary_df['sharpe_ratio'], name='Sharpe Ratio'),
            row=1, col=2
        )
        
        # Max Drawdown
        fig.add_trace(
            go.Bar(x=summary_df['symbol'] + ' ' + summary_df['interval'], 
                   y=summary_df['max_drawdown'], name='Max Drawdown'),
            row=2, col=1
        )
        
        # Hit Rate
        fig.add_trace(
            go.Bar(x=summary_df['symbol'] + ' ' + summary_df['interval'], 
                   y=summary_df['hit_rate'], name='Hit Rate'),
            row=2, col=2
        )
        
        fig.update_layout(height=800, title_text="Summary of All Results")
        summary_chart = fig.to_html(include_plotlyjs='cdn')
        
        # Create summary table
        summary_table = summary_df.to_html(index=False, float_format='%.3f')
        
        # HTML template for summary
        summary_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Summary Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .header { background-color: #f0f0f0; padding: 20px; border-radius: 5px; }
                .section { margin: 20px 0; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }
                table { width: 100%; border-collapse: collapse; margin: 10px 0; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                .chart-container { margin: 20px 0; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Summary Report</h1>
                <p><strong>Generated:</strong> {{timestamp}}</p>
                <p><strong>Total Results:</strong> {{total_results}}</p>
            </div>
            
            <div class="section">
                <h2>Summary Charts</h2>
                <div class="chart-container">
                    {{summary_chart}}
                </div>
            </div>
            
            <div class="section">
                <h2>Summary Table</h2>
                {{summary_table}}
            </div>
        </body>
        </html>
        """
        
        template = Template(summary_template)
        summary_html = template.render(
            timestamp=format_timestamp(),
            total_results=len(summary_data),
            summary_chart=summary_chart,
            summary_table=summary_table
        )
        
        return summary_html


def main():
    """Main function for report generation."""
    from src.utils import load_config, setup_logging, set_deterministic_seed
    
    # Load configuration
    config = load_config("config/settings.yaml")
    
    # Setup logging
    logger = setup_logging(config)
    
    # Set deterministic seed
    set_deterministic_seed(config.app["seed"])
    
    # Generate reports
    generator = ReportGenerator(config)
    
    # For testing, create sample results
    sample_results = {
        'SOLUSDT_1m_label_class_1': {
            'backtest_results': {
                'performance': {
                    'total_return': 0.15,
                    'sharpe_ratio': 1.2,
                    'max_drawdown': -0.05,
                    'hit_rate': 0.65,
                    'total_trades': 100
                },
                'trades': [],
                'equity_curve': [10000] * 100
            },
            'feature_importance': pd.Series({'rsi': 0.3, 'macd': 0.25, 'volume': 0.2})
        }
    }
    
    # Generate reports
    report_files = generator.generate_all_reports(sample_results)
    print(f"Generated {len(report_files)} reports")


if __name__ == "__main__":
    main()