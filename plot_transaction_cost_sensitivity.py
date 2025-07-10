import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_transaction_cost_sensitivity():
    # Set plot style consistent with paper
    sns.set_context('notebook')
    sns.set_style('whitegrid')
    plt.rcParams.update({
        'figure.figsize': (12, 7),
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'legend.fontsize': 11,
        'grid.alpha': 0.4
    })

    # List of strategies you ran
    strategies = [
        'arima_garch_gmir',
        'arima_garch_gmv',
        'xgboost_gmir',
        'xgboost_gmv'
    ]

    # Data container
    tc_data = []

    # Load transaction cost results
    for strat in strategies:
        file_path = f'data/tc_sensitivity_{strat}.csv'
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, index_col=0)
            for tc, row in df.iterrows():
                tc_data.append({
                    'Strategy': strat.upper(),
                    'Transaction Cost (%)': float(tc.rstrip('%')),
                    'Final Value': row['Final Value']
                })
        else:
            print(f"Warning: Missing {file_path}")

    if not tc_data:
        print("No transaction cost data found. Please run the backtests first.")
        return

    # Convert to DataFrame
    tc_df = pd.DataFrame(tc_data)

    # Plot
    plt.figure(figsize=(12, 7))
    ax = sns.barplot(
        x='Transaction Cost (%)',
        y='Final Value',
        hue='Strategy',
        data=tc_df,
        palette='Set2'
    )

    plt.title('Transaction Cost Sensitivity — Final Portfolio Values', fontsize=16)
    plt.ylabel('Final Portfolio Value')
    plt.xlabel('Transaction Cost (% of Trade Value)')

    # Format x-axis to show percentage with proper tick labels
    x_ticks = sorted(tc_df['Transaction Cost (%)'].unique())
    plt.xticks(range(len(x_ticks)), [f'{x:.2f}%' for x in x_ticks])
    
    plt.grid(True, axis='y', linestyle='--', alpha=0.6)
    plt.legend(title='Strategy', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    # Save to plots/
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/transaction_cost_sensitivity.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    plot_transaction_cost_sensitivity()