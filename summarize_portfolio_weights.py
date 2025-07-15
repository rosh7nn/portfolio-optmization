import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Set up directories
os.makedirs('figures', exist_ok=True)
os.makedirs('data/summaries', exist_ok=True)

# List of strategy files (corrected to data/ directory)
weight_files = {
    'ARIMA-GARCH GMIR': 'data/weights_arima_garch_gmir.csv',
    'ARIMA-GARCH GMV' : 'data/weights_arima_garch_gmv.csv',
    'XGBoost GMIR'    : 'data/weights_xgboost_gmir.csv',
    'XGBoost GMV'     : 'data/weights_xgboost_gmv.csv'
}

# Dictionary to hold top 5 average weights for each strategy
top5_summary = {}

for strategy, file in weight_files.items():
    print(f"\nProcessing {strategy} weights...")
    
    # Load weights CSV
    weights = pd.read_csv(file, index_col=0)
    
    # Compute average weight per stock
    avg_weights = weights.mean().sort_values(ascending=False)
    
    # Save average weights to file
    avg_weights.to_csv(f"data/summaries/avg_weights_{strategy.lower().replace(' ', '_')}.csv")
    
    # Get top 5 stocks
    top5 = avg_weights.head(5)
    print(top5)
    
    # Save to summary dictionary
    top5_summary[strategy] = top5
    
    # Plot heatmap of weights over time
    plt.figure(figsize=(14, 6))
    sns.heatmap(weights.T, cmap='YlGnBu', cbar_kws={'label': 'Weight'}, linewidths=0.1)
    plt.title(f'Portfolio Weights Heatmap: {strategy}')
    plt.xlabel('Date')
    plt.ylabel('Stock')
    plt.tight_layout()
    
    fig_path = f'figures/weights_heatmap_{strategy.lower().replace(" ", "_")}.png'
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"✓ Saved heatmap to {fig_path}")

# Combine top 5s into one DataFrame for table export
summary_df = pd.DataFrame()

for strategy, top5 in top5_summary.items():
    temp_df = pd.DataFrame(top5)
    temp_df.columns = [strategy]
    summary_df = pd.concat([summary_df, temp_df], axis=1)

# Save combined summary to CSV
summary_csv_path = 'data/summaries/top5_weights_summary.csv'
summary_df.to_csv(summary_csv_path)
print(f"\n✓ Saved top 5 weight summary to {summary_csv_path}")

print("\n✔️ All weight analyses complete.")
