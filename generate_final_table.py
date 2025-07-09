#!/usr/bin/env python3
"""
Final Performance Table Generator

This script generates a consolidated performance table from individual strategy metrics files.
Run this after completing all backtests to get a clean summary of all strategies.
"""

import pandas as pd
from tabulate import tabulate
import os
from pathlib import Path

# Ensure output directory exists
Path("data").mkdir(exist_ok=True)

def load_and_combine_metrics():
    """Load individual metrics files and combine into a summary table."""
    # List of metrics files and their display names
    metrics_files = [
        ('data/metrics_arima-garch_gmir.csv', 'ARIMA-GARCH GMIR'),
        ('data/metrics_arima-garch_gmv.csv', 'ARIMA-GARCH GMV'),
        ('data/metrics_xgboost_gmir.csv', 'XGBoost GMIR'),
        ('data/metrics_xgboost_gmv.csv', 'XGBoost GMV'),
    ]
    
    results = []
    
    for file_path, strategy_name in metrics_files:
        if not os.path.exists(file_path):
            print(f"Warning: File not found: {file_path}")
            continue
            
        try:
            # Read the metrics file
            metrics = pd.read_csv(file_path)
            
            # Get the last row (final metrics)
            if len(metrics) > 0:
                metrics = metrics.iloc[-1].to_dict()
                
                # Helper function to format metrics with proper handling of None/NaN
                def format_metric(value, fmt='.2f'):
                    if pd.isna(value) or value == 'nan' or value == '':
                        return 'N/A'
                    try:
                        return f"{float(value):{fmt}}"
                    except (ValueError, TypeError):
                        return str(value)
                
                # Extract and format the metrics we want
                results.append({
                    'Strategy': strategy_name,
                    'Total Return (%)': format_metric(metrics.get('Total Return')),
                    'Ann. Return (%)': format_metric(metrics.get('Annualized Return')),
                    'Ann. Volatility (%)': format_metric(metrics.get('Annualized Volatility')),
                    'Sharpe Ratio': format_metric(metrics.get('Sharpe Ratio'), '.3f'),
                    'Max Drawdown (%)': format_metric(metrics.get('Max Drawdown')),
                    'Information Ratio': format_metric(metrics.get('Information Ratio'), '.3f'),
                    'Modified IR': format_metric(metrics.get('Modified IR'), '.3f'),
                    'Mean Stocks (75% cov)': format_metric(metrics.get('Mean Stocks (75% coverage)'), '.1f'),
                    'Mean Stocks (90% cov)': format_metric(metrics.get('Mean Stocks (90% coverage)'), '.1f'),
                })
                
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
    
    if not results:
        print("No valid metrics files found.")
        return None
        
    return pd.DataFrame(results)

def save_final_table(df, output_file='data/final_performance_table.csv'):
    """Save the final table to CSV."""
    if df is not None and not df.empty:
        df.to_csv(output_file, index=False)
        print(f"\n✓ Saved final performance table to {output_file}")
    else:
        print("No data to save.")

def display_table(df):
    """Display the table in a nicely formatted way."""
    if df is not None and not df.empty:
        # Create a copy to avoid modifying the original
        display_df = df.copy()
        
        # Print table with borders
        print("\n" + "="*120)
        print("FINAL PERFORMANCE SUMMARY".center(120))
        print("="*120)
        
        # Convert numeric columns to float for proper alignment
        numeric_cols = [col for col in display_df.columns if col != 'Strategy']
        for col in numeric_cols:
            display_df[col] = display_df[col].apply(
                lambda x: float(x) if x != 'N/A' and str(x).replace('.', '').replace('-', '').isdigit() else x
            )
        
        # Convert DataFrame to list of lists for tabulate
        table_data = []
        headers = display_df.columns.tolist()
        
        for _, row in display_df.iterrows():
            table_data.append([
                row[col] if not pd.isna(row[col]) and row[col] != 'N/A' else 'N/A'
                for col in headers
            ])
        
        # Print the table with consistent formatting
        print(tabulate(
            table_data,
            headers=headers,
            tablefmt='grid',
            stralign='right',
            numalign='right',
            floatfmt=('.2f' if '%' in col or col in ['Sharpe Ratio', 'Information Ratio', 'Modified IR'] else '.1f'
                    for col in headers)
        ))
        print("="*120 + "\n")
    else:
        print("No data to display.")

def main():
    print("Generating final performance summary...")
    
    # Load and combine metrics
    final_table = load_and_combine_metrics()
    
    if final_table is not None:
        # Save to CSV
        save_final_table(final_table)
        
        # Display in console
        display_table(final_table)
    
    print("Done!")

if __name__ == "__main__":
    main()
