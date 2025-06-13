import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

# Set style for better looking plots
plt.style.use('default')
sns.set_palette("husl")

def ensure_plots_directory():
    """Create plots directory if it doesn't exist"""
    if not os.path.exists('plots'):
        os.makedirs('plots')
        print("Created 'plots' directory")

def create_visualizations(train_data, processed_data, classifier, results):
    """Create comprehensive visualizations for the NDVI classification project"""
    ensure_plots_directory()

    # Set up the plotting parameters
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 10

    # 1. Data Overview Visualization
    create_data_overview(train_data, processed_data)

    # 2. Class Distribution
    create_class_distribution(train_data)

    # 3. NDVI Time Series by Class
    create_ndvi_by_class(train_data)

    # 4. Model Performance
    create_model_performance(results)

    print("All visualizations saved to 'plots/' directory")

def create_data_overview(train_data, processed_data):
    """Create overview visualization of the dataset"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('NDVI Dataset Overview', fontsize=16, fontweight='bold')

    # Get NDVI columns
    ndvi_cols = [col for col in train_data.columns if col.endswith('_N')]
    ndvi_data = train_data[ndvi_cols]

    # 1. Missing data heatmap
    ax1 = axes[0, 0]
    missing_data = ndvi_data.isnull()
    sns.heatmap(missing_data.head(50), cbar=True, ax=ax1, cmap='viridis', 
                xticklabels=False, yticklabels=False)
    ax1.set_title('Missing Data Pattern (First 50 samples)')
    ax1.set_xlabel('NDVI Time Points')
    ax1.set_ylabel('Samples')

    # 2. NDVI value distribution
    ax2 = axes[0, 1]
    ndvi_values = ndvi_data.values.flatten()
    ndvi_values = ndvi_values[~np.isnan(ndvi_values)]
    ax2.hist(ndvi_values, bins=50, alpha=0.7, color='green', edgecolor='black')
    ax2.set_title('Distribution of All NDVI Values')
    ax2.set_xlabel('NDVI Value')
    ax2.set_ylabel('Frequency')
    ax2.axvline(ndvi_values.mean(), color='red', linestyle='--', 
                label=f'Mean: {ndvi_values.mean():.3f}')
    ax2.legend()

    # 3. Feature correlation heatmap (top features)
    ax3 = axes[1, 0]
    feature_cols = [col for col in processed_data.columns if col not in ['ID', 'class']]
    corr_matrix = processed_data[feature_cols[:10]].corr()  # Top 10 features
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax3,
                fmt='.2f', square=True)
    ax3.set_title('Feature Correlation Matrix (Top 10)')

    # 4. Time series length distribution
    ax4 = axes[1, 1]
    series_lengths = []
    for idx, row in ndvi_data.iterrows():
        non_null_count = row.count()
        series_lengths.append(non_null_count)

    ax4.hist(series_lengths, bins=20, alpha=0.7, color='blue', edgecolor='black')
    ax4.set_title('Distribution of Time Series Lengths')
    ax4.set_xlabel('Number of Non-null NDVI Values')
    ax4.set_ylabel('Frequency')
    ax4.axvline(np.mean(series_lengths), color='red', linestyle='--',
                label=f'Mean: {np.mean(series_lengths):.1f}')
    ax4.legend()

    plt.tight_layout()
    plt.savefig('plots/data_overview.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_class_distribution(train_data):
    """Create class distribution visualization"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Land Cover Class Distribution', fontsize=16, fontweight='bold')

    # 1. Bar plot
    class_counts = train_data['class'].value_counts()
    ax1 = axes[0]
    bars = ax1.bar(class_counts.index, class_counts.values, 
                   color=sns.color_palette("husl", len(class_counts)))
    ax1.set_title('Class Counts')
    ax1.set_xlabel('Land Cover Class')
    ax1.set_ylabel('Number of Samples')
    ax1.tick_params(axis='x', rotation=45)

    # Add value labels on bars
    for bar, count in zip(bars, class_counts.values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{count}', ha='center', va='bottom', fontweight='bold')

    # 2. Pie chart
    ax2 = axes[1]
    colors = sns.color_palette("husl", len(class_counts))
    wedges, texts, autotexts = ax2.pie(class_counts.values, labels=class_counts.index, 
                                       autopct='%1.1f%%', colors=colors, startangle=90)
    ax2.set_title('Class Proportions')

    # Make percentage text bold
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')

    plt.tight_layout()
    plt.savefig('plots/class_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_ndvi_by_class(train_data):
    """Create NDVI time series visualization by class"""
    ndvi_cols = [col for col in train_data.columns if col.endswith('_N')]

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('NDVI Time Series Patterns by Land Cover Class', fontsize=16, fontweight='bold')
    axes = axes.flatten()

    classes = train_data['class'].unique()
    colors = sns.color_palette("husl", len(classes))

    for i, land_class in enumerate(sorted(classes)):
        ax = axes[i]
        class_data = train_data[train_data['class'] == land_class]

        # Plot sample time series for this class
        sample_size = min(50, len(class_data))
        sample_data = class_data.sample(n=sample_size, random_state=42)

        time_points = range(len(ndvi_cols))

        for idx, row in sample_data.iterrows():
            ndvi_values = row[ndvi_cols].values
            # Handle missing values
            valid_mask = ~pd.isna(ndvi_values)
            if valid_mask.sum() > 0:
                ax.plot(np.array(time_points)[valid_mask], ndvi_values[valid_mask], 
                       alpha=0.3, color=colors[i], linewidth=0.5)

        # Plot mean time series
        mean_series = class_data[ndvi_cols].mean()
        ax.plot(time_points, mean_series, color='black', linewidth=3, 
                label=f'Mean NDVI', alpha=0.8)

        ax.set_title(f'{land_class} (n={len(class_data)})')
        ax.set_xlabel('Time Point')
        ax.set_ylabel('NDVI Value')
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Set consistent y-axis limits
        ax.set_ylim(-0.5, 1.0)

    # Hide empty subplot if odd number of classes
    if len(classes) < len(axes):
        axes[-1].set_visible(False)

    plt.tight_layout()
    plt.savefig('plots/ndvi_by_class.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_model_performance(results):
    """Create model performance visualization"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Model Performance Metrics', fontsize=16, fontweight='bold')

    # 1. Accuracy comparison
    ax1 = axes[0]
    metrics = ['Training', 'Validation', 'Cross-Val']
    values = [results['train_accuracy'], results['val_accuracy'], results['cv_mean']]
    errors = [0, 0, results['cv_std']]

    bars = ax1.bar(metrics, values, yerr=errors, capsize=5, 
                   color=['skyblue', 'lightcoral', 'lightgreen'],
                   edgecolor='black', linewidth=1)

    ax1.set_title('Model Accuracy Comparison')
    ax1.set_ylabel('Accuracy Score')
    ax1.set_ylim(0, 1)

    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

    # Add horizontal line at 1/6 (random chance for 6 classes)
    ax1.axhline(y=1/6, color='red', linestyle='--', alpha=0.7, 
                label='Random Chance (16.7%)')
    ax1.legend()

    # 2. Performance summary table
    ax2 = axes[1]
    ax2.axis('tight')
    ax2.axis('off')

    # Create table data
    table_data = [
        ['Metric', 'Value'],
        ['Training Accuracy', f'{results["train_accuracy"]:.1%}'],
        ['Validation Accuracy', f'{results["val_accuracy"]:.1%}'],
        ['CV Mean Accuracy', f'{results["cv_mean"]:.1%}'],
        ['CV Std Deviation', f'{results["cv_std"]:.1%}'],
        ['Overfitting Gap', f'{results["train_accuracy"] - results["val_accuracy"]:.1%}']
    ]

    table = ax2.table(cellText=table_data[1:], colLabels=table_data[0],
                     cellLoc='center', loc='center',
                     colWidths=[0.6, 0.4])

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)

    # Color the header
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    ax2.set_title('Detailed Performance Metrics')

    plt.tight_layout()
    plt.savefig('plots/model_performance.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    print("Visualization module loaded successfully!")
    print("Use create_visualizations() to generate all plots.")
