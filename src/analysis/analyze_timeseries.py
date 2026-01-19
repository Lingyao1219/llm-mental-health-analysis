"""
Time series analysis of LLM impact on mental health grouped by month.

Creates visualizations showing:
1. Overall average sentiment and volume over time (dual y-axes)
2. Top mental health conditions sentiment trends (2 subplots)
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib import rcParams
from datetime import datetime
import scipy.stats as stats

rcParams['figure.dpi'] = 500
rcParams['savefig.dpi'] = 500
rcParams['font.family'] = 'Arial'
rcParams['font.size'] = 12
rcParams['axes.labelsize'] = 12
rcParams['axes.titlesize'] = 14
rcParams['xtick.labelsize'] = 12
rcParams['ytick.labelsize'] = 12
rcParams['legend.fontsize'] = 11
rcParams['figure.titlesize'] = 14

# Load the cleaned dataset
df = pd.read_csv('result/final_dataset_cleaned.csv')

print(f"Loaded {len(df)} rows")

# Convert date column to datetime
df['date'] = pd.to_datetime(df['date'])

# Extract year-month for grouping
df['year_month'] = df['date'].dt.to_period('M')

# Normalize case for mental health columns
df['mapped_primary_mental'] = df['mapped_primary_mental'].str.lower()

# Function to normalize category names (remove "other(" wrapper)
def normalize_category(cat):
    """Remove other(...) wrapper to merge with base categories."""
    if pd.isna(cat):
        return cat
    cat_str = str(cat).strip()
    if cat_str.startswith('other('):
        cat_str = cat_str.replace('other(', '', 1).strip()
        if cat_str.endswith(')'):
            cat_str = cat_str[:-1].strip()
    return cat_str

# Apply normalization
df['mapped_primary_mental_normalized'] = df['mapped_primary_mental'].apply(normalize_category)

# Filter valid sentiments
df_filtered = df[df['llm_impact'].isin(['positive', 'negative', 'neutral'])].copy()

# Convert sentiment to numeric scores
# positive = 1, neutral = 0, negative = -1
sentiment_map = {'positive': 1, 'neutral': 0, 'negative': -1}
df_filtered['sentiment_score'] = df_filtered['llm_impact'].map(sentiment_map)

print(f"Filtered dataset: {len(df_filtered)} rows")
print(f"Date range: {df_filtered['date'].min()} to {df_filtered['date'].max()}")

# =============================================================================
# Group by month - Calculate average sentiment and volume
# =============================================================================
print("\n" + "="*80)
print("Calculating monthly average sentiment and volume")
print("="*80)

# Calculate average sentiment score by month with confidence intervals
monthly_stats = df_filtered.groupby('year_month')['sentiment_score'].agg(['mean', 'sem', 'count'])

# Calculate 95% confidence intervals
import scipy.stats as stats
monthly_stats['ci_lower'] = monthly_stats.apply(
    lambda row: row['mean'] - stats.t.ppf(0.975, row['count']-1) * row['sem'], axis=1
)
monthly_stats['ci_upper'] = monthly_stats.apply(
    lambda row: row['mean'] + stats.t.ppf(0.975, row['count']-1) * row['sem'], axis=1
)

# Calculate volume (count) by month
monthly_volume = df_filtered.groupby('year_month').size()

# Convert period index to timestamp for plotting
monthly_stats.index = monthly_stats.index.to_timestamp()
monthly_volume.index = monthly_volume.index.to_timestamp()

# Remove the last data point
monthly_stats = monthly_stats[:-1]
monthly_volume = monthly_volume[:-1]

# Extract mean for easier access
monthly_avg_sentiment = monthly_stats['mean']

print(f"\nMonths covered: {len(monthly_avg_sentiment)}")
print(f"Average sentiment range: [{monthly_avg_sentiment.min():.3f}, {monthly_avg_sentiment.max():.3f}]")
print(f"Volume range: [{monthly_volume.min()}, {monthly_volume.max()}]")

# =============================================================================
# Get top primary mental health conditions (>= 100 occurrences)
# =============================================================================
print("\n" + "="*80)
print("Identifying top primary mental health conditions")
print("="*80)

# Filter to include only rows with valid primary mental health categories
df_mental = df_filtered[
    (df_filtered['mapped_primary_mental_normalized'].notna()) &
    (df_filtered['mapped_primary_mental_normalized'] != 'none') &
    (df_filtered['mapped_primary_mental_normalized'] != 'other')
].copy()

primary_counts = df_mental['mapped_primary_mental_normalized'].value_counts()
top_conditions = [cat for cat in primary_counts.index if primary_counts[cat] >= 100][:5]  # Top 5

print(f"\nTop 5 conditions (>= 100 occurrences):")
for condition in top_conditions:
    print(f"  {condition}: {primary_counts[condition]}")

# Filter to only top conditions
df_top = df_mental[df_mental['mapped_primary_mental_normalized'].isin(top_conditions)].copy()

print(f"\nFiltered to top conditions: {len(df_top)} rows")

# =============================================================================
# Calculate monthly metrics for each condition
# =============================================================================
print("\n" + "="*80)
print("Calculating monthly sentiment and volume by condition")
print("="*80)

# Professional color palette (custom)
colors = ['#84ADDC', '#FFA288', '#BBC7BE', '#6AD1A3', '#FFD47D']
color_map = dict(zip(top_conditions, colors))

monthly_data = {}

for condition in top_conditions:
    # Group by both year_month AND condition
    df_cond = df_top[df_top['mapped_primary_mental_normalized'] == condition].copy()

    # Calculate monthly average sentiment grouped by month
    monthly_sentiment = df_cond.groupby('year_month')['sentiment_score'].mean()

    # Calculate monthly volume grouped by month
    monthly_volume_cond = df_cond.groupby('year_month').size()

    # Convert to timestamp
    monthly_sentiment.index = monthly_sentiment.index.to_timestamp()
    monthly_volume_cond.index = monthly_volume_cond.index.to_timestamp()

    # Remove last data point
    if len(monthly_sentiment) > 0:
        monthly_sentiment = monthly_sentiment[:-1]
    if len(monthly_volume_cond) > 0:
        monthly_volume_cond = monthly_volume_cond[:-1]

    monthly_data[condition] = {
        'sentiment': monthly_sentiment,
        'volume': monthly_volume_cond
    }

    print(f"\n{condition.title()}:")
    print(f"  Total posts: {primary_counts[condition]}")
    print(f"  Months covered: {len(monthly_sentiment)}")
    if len(monthly_sentiment) > 0:
        print(f"  Avg sentiment: {monthly_sentiment.mean():.3f}")
        print(f"  Avg monthly volume: {monthly_volume_cond.mean():.1f}")

# =============================================================================
# Key time points for LLM releases
# =============================================================================
key_events = [
    ('2024-02', 'Gemini 1.5'),
    ('2024-04', 'Llama 3'),
    ('2024-05', 'GPT-4o'),
    ('2024-06', 'Claude 3.5 Sonnet'),
    ('2024-08', 'Grok-2'),
    ('2025-01', 'DeepSeek R1'),
    ('2025-02', 'Grok-3'),
    ('2025-04', 'GPT-4.1'),
    ('2025-05', 'Claude 4'),
    ('2025-06', 'Gemini 2.5'),
    ('2025-08', 'GPT-5'),
]

# Convert to datetime and group by month to handle overlaps
from collections import defaultdict
key_events_by_month = defaultdict(list)
for date, name in key_events:
    event_date = pd.to_datetime(date + '-01')
    key_events_by_month[event_date].append(name)

# =============================================================================
# COMBINED FIGURE: All 4 subplots (2x2 layout)
# =============================================================================
print("\n" + "="*80)
print("Creating combined figure with 4 subplots")
print("="*80)

fig = plt.figure(figsize=(14, 7), facecolor='white')
gs = fig.add_gridspec(2, 2, hspace=0.15, wspace=0.20)

ax1 = fig.add_subplot(gs[0, 0])  # Top-left: Overall volume
ax2 = fig.add_subplot(gs[0, 1], sharex=ax1)  # Top-right: Overall sentiment
ax3 = fig.add_subplot(gs[1, 0], sharex=ax1)  # Bottom-left: Conditions volume (SWITCHED)
ax4 = fig.add_subplot(gs[1, 1], sharex=ax1)  # Bottom-right: Conditions sentiment (SWITCHED)

# =============================================================================
# SUBPLOT A: Overall post volume over time
# =============================================================================
color_vol = '#84ADDC'  # Soft blue
ax1.plot(monthly_volume.index, monthly_volume.values,
         marker='o', linewidth=2, color=color_vol, markersize=6,
         markerfacecolor='white', markeredgewidth=1.5, markeredgecolor=color_vol)
ax1.set_ylabel('Number of Posts', fontweight='bold')
ax1.grid(axis='both', alpha=0.5, linestyle='-', linewidth=0.5)
ax1.set_axisbelow(True)

# Add key time points as vertical lines with non-overlapping labels
for event_date, event_names in key_events_by_month.items():
    if monthly_volume.index.min() <= event_date <= monthly_volume.index.max():
        ax1.axvline(x=event_date, color='#7A8A80', linestyle='--', linewidth=0.8, alpha=0.8)
        # Add text labels with vertical offset for multiple events
        y_max = ax1.get_ylim()[1]
        for idx, event_name in enumerate(event_names):
            y_pos = y_max * (0.95 - idx * 0.18)  # Increased offset for multiple events
            ax1.text(event_date, y_pos, event_name, rotation=90,
                    verticalalignment='top', alpha=1.0, color='black')

# Add subplot label
ax1.text(-0.1, 1.10, 'a.', transform=ax1.transAxes,
         fontweight='bold', fontsize=14, va='top')

# Style
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.tick_params(labelbottom=False)  # Hide x-axis labels for top row

# =============================================================================
# SUBPLOT B: Overall average sentiment over time
# =============================================================================
color_sent = '#FFA288'  # Coral/salmon
# Calculate error bar values
yerr_lower = monthly_stats['mean'] - monthly_stats['ci_lower']
yerr_upper = monthly_stats['ci_upper'] - monthly_stats['mean']

# Plot with error bars
ax2.errorbar(monthly_avg_sentiment.index, monthly_avg_sentiment.values,
             yerr=[yerr_lower, yerr_upper],
             marker='o', linewidth=2, color=color_sent, markersize=5,
             markerfacecolor='white', markeredgewidth=1.5, markeredgecolor=color_sent,
             capsize=3, capthick=1, ecolor=color_sent, elinewidth=1)
ax2.set_ylabel('Average Sentiment Score', fontweight='bold')
ax2.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.4)
ax2.set_ylim(-0.2, 1)
ax2.grid(axis='both', alpha=0.2, linestyle='-', linewidth=0.5)
ax2.set_axisbelow(True)

# Add trend arrow and annotation from 2025-01
jan_2025 = pd.to_datetime('2025-01-01')
if jan_2025 in monthly_avg_sentiment.index:
    jan_idx = monthly_avg_sentiment.index.get_loc(jan_2025)
    if jan_idx < len(monthly_avg_sentiment) - 1:
        # Get values
        start_val = monthly_avg_sentiment.iloc[jan_idx]
        end_val = monthly_avg_sentiment.iloc[-1]

        # Add arrow
        ax2.annotate('', xy=(monthly_avg_sentiment.index[-1], end_val),
                    xytext=(jan_2025, start_val),
                    arrowprops=dict(arrowstyle='->', color='#7A8A80', lw=1.5, alpha=1.0))

        # Add annotation text
        mid_idx = (jan_idx + len(monthly_avg_sentiment) - 1) // 3
        mid_x = monthly_avg_sentiment.index[mid_idx]
        mid_y = (start_val + end_val) / 2 -0.1
        ax2.text(mid_x, mid_y, 'Declining trend',
                color='black', fontweight='bold', ha='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                         edgecolor='#7A8A80', alpha=1.0))

# Add subplot label
ax2.text(-0.1, 1.10, 'b.', transform=ax2.transAxes,
         fontweight='bold', fontsize=14, va='top')

# Style
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.tick_params(labelbottom=False)  # Hide x-axis labels for top row

# =============================================================================
# SUBPLOT C: Mental health conditions - volume trends (SWITCHED FROM D)
# =============================================================================
for condition in top_conditions:
    volume = monthly_data[condition]['volume']
    ax3.plot(volume.index, volume.values,
             marker='o', linewidth=2, label=condition.title(),
             color=color_map[condition], markersize=5,
             markerfacecolor='white', markeredgewidth=1.5, markeredgecolor=color_map[condition])

ax3.set_xlabel('Month', fontweight='bold')
ax3.set_ylabel('Number of Posts', fontweight='bold')
ax3.legend(loc='best', frameon=True, ncol=1, framealpha=0.9)
ax3.grid(axis='both', alpha=0.2, linestyle='-', linewidth=0.5)
ax3.set_axisbelow(True)

# Add subplot label
ax3.text(-0.1, 1.10, 'c.', transform=ax3.transAxes,
         fontweight='bold', fontsize=14, va='top')

# Rotate x-axis labels
ax3.tick_params(axis='x', rotation=45)
plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Style
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)

# =============================================================================
# SUBPLOT D: Mental health conditions - sentiment trends (SWITCHED FROM C)
# =============================================================================
for condition in top_conditions:
    sentiment = monthly_data[condition]['sentiment']
    ax4.plot(sentiment.index, sentiment.values,
             marker='o', linewidth=2, label=condition.title(),
             color=color_map[condition], markersize=5,
             markerfacecolor='white', markeredgewidth=1.5, markeredgecolor=color_map[condition])

ax4.set_xlabel('Month', fontweight='bold')
ax4.set_ylabel('Average Sentiment Score', fontweight='bold')
ax4.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.4)
ax4.set_ylim(-0.2, 1)
# No legend for subplot d
ax4.grid(axis='both', alpha=0.2, linestyle='-', linewidth=0.5)
ax4.set_axisbelow(True)

# Add subplot label
ax4.text(-0.1, 1.10, 'd.', transform=ax4.transAxes,
         fontweight='bold', fontsize=14, va='top')

# Rotate x-axis labels
ax4.tick_params(axis='x', rotation=45)
plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Style
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)

plt.savefig('images/timeseries.pdf', dpi=300, bbox_inches='tight', format='pdf')
plt.savefig('images/timeseries.png', dpi=300, bbox_inches='tight')
plt.close()

print("Saved: images/timeseries.pdf and .png")

# =============================================================================
# SUMMARY STATISTICS
# =============================================================================
print("\n" + "="*80)
print("Summary Statistics")
print("="*80)

print("\nOverall statistics:")
print(f"  Total posts: {len(df_filtered)}")
print(f"  Average sentiment score: {df_filtered['sentiment_score'].mean():.3f}")
print(f"  Date range: {df_filtered['date'].min()} to {df_filtered['date'].max()}")

print("\nTotal posts by sentiment:")
total_by_sentiment = df_filtered['llm_impact'].value_counts()
print(total_by_sentiment)

print("\nMonthly statistics:")
print(f"  Average monthly volume: {monthly_volume.mean():.1f}")
print(f"  Average monthly sentiment: {monthly_avg_sentiment.mean():.3f}")

print("\n" + "="*80)
print("Summary Statistics by Condition")
print("="*80)

for condition in top_conditions:
    sentiment = monthly_data[condition]['sentiment']
    volume = monthly_data[condition]['volume']

    print(f"\n{condition.title()}:")
    print(f"  Total posts: {volume.sum()}")
    print(f"  Average monthly volume: {volume.mean():.1f}")
    print(f"  Average sentiment: {sentiment.mean():.3f}")
    print(f"  Sentiment range: [{sentiment.min():.3f}, {sentiment.max():.3f}]")

print("\n" + "="*80)
print("Analysis complete!")
print("="*80)
print("\nGenerated files:")
print("  - images/timeseries.pdf and .png")
