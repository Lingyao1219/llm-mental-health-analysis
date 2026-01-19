"""
Combined analysis: Timeseries (2x2 layout) + Fingerprint (1 column).

Creates a figure with 3 columns:
- Columns 1-2: Timeseries analysis (2x2 layout from analyze_timeseries.py)
  - Top-left: Overall post volume
  - Top-right: Overall sentiment with 95% CI
  - Bottom-left: Mental health conditions volume
  - Bottom-right: Mental health conditions sentiment
- Column 3: Fingerprint analysis (stacked layout with 2 rows)
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib import rcParams
from matplotlib.dates import DateFormatter
from datetime import datetime
import scipy.stats as stats
from collections import defaultdict
import textwrap

rcParams['figure.dpi'] = 500
rcParams['savefig.dpi'] = 500
rcParams['font.family'] = 'Arial'
rcParams['font.size'] = 12
rcParams['axes.labelsize'] = 12
rcParams['axes.titlesize'] = 14
rcParams['xtick.labelsize'] = 12
rcParams['ytick.labelsize'] = 12
rcParams['legend.fontsize'] = 12
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
sentiment_map = {'positive': 1, 'neutral': 0, 'negative': -1}
df_filtered['sentiment_score'] = df_filtered['llm_impact'].map(sentiment_map)

print(f"Filtered dataset: {len(df_filtered)} rows")
print(f"Date range: {df_filtered['date'].min()} to {df_filtered['date'].max()}")

# =============================================================================
# TIMESERIES DATA PREPARATION
# =============================================================================
print("\n" + "="*80)
print("Calculating monthly average sentiment and volume")
print("="*80)

# Calculate average sentiment score by month with confidence intervals
monthly_stats = df_filtered.groupby('year_month')['sentiment_score'].agg(['mean', 'sem', 'count'])

# Calculate 95% confidence intervals
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
key_events_by_month = defaultdict(list)
for date, name in key_events:
    event_date = pd.to_datetime(date + '-01')
    key_events_by_month[event_date].append(name)

# =============================================================================
# FINGERPRINT DATA PREPARATION
# =============================================================================
print("\n" + "="*80)
print("Preparing fingerprint analysis data")
print("="*80)

# Get LLMs with >= 50 samples for fingerprint
df_fingerprint = df[
    (df['mapped_primary_mental_normalized'].notna()) &
    (df['mapped_primary_mental_normalized'] != 'none') &
    (df['mapped_primary_mental_normalized'] != 'other') &
    (df['mapped_llm_product'] != 'Others') &
    (df['llm_impact'].isin(['positive', 'negative', 'neutral']))
].copy()

# Get primary mental health categories (>= 100 occurrences)
# Keep the order from value_counts (sorted by frequency, not alphabetically)
fingerprint_primary_counts = df_fingerprint['mapped_primary_mental_normalized'].value_counts()
fingerprint_primary_categories = [cat for cat in fingerprint_primary_counts.index if fingerprint_primary_counts[cat] >= 100]

# Filter to only include selected categories
df_fingerprint = df_fingerprint[df_fingerprint['mapped_primary_mental_normalized'].isin(fingerprint_primary_categories)].copy()

# Get LLM products - use same order as original fingerprint analysis
# Original order: ['GPT', 'Claude', 'Gemini', 'Others', 'Grok', 'DeepSeek', 'Llama']
# Since we exclude 'Others', we get: ['GPT', 'Claude', 'Gemini', 'Grok', 'DeepSeek', 'Llama']
fingerprint_llm_counts = df_fingerprint['mapped_llm_product'].value_counts()
original_llm_order = ['GPT', 'Claude', 'Gemini', 'Grok', 'DeepSeek', 'Llama']
fingerprint_llms = [llm for llm in original_llm_order if llm in fingerprint_llm_counts.index and fingerprint_llm_counts[llm] >= 50]

# Filter to only include selected LLMs
df_fingerprint = df_fingerprint[df_fingerprint['mapped_llm_product'].isin(fingerprint_llms)].copy()

# Calculate percentage distribution for each LLM
fingerprint_data = []

for llm in fingerprint_llms:
    llm_df = df_fingerprint[df_fingerprint['mapped_llm_product'] == llm]
    total = len(llm_df)

    positive_pct = (llm_df['llm_impact'] == 'positive').sum() / total * 100
    neutral_pct = (llm_df['llm_impact'] == 'neutral').sum() / total * 100
    negative_pct = (llm_df['llm_impact'] == 'negative').sum() / total * 100

    fingerprint_data.append({
        'llm': llm,
        'positive': positive_pct,
        'neutral': neutral_pct,
        'negative': negative_pct,
        'n': total
    })

fingerprint_df = pd.DataFrame(fingerprint_data)

# Calculate for categories
fingerprint_cat_data = []

for cat in fingerprint_primary_categories:
    cat_df = df_fingerprint[df_fingerprint['mapped_primary_mental_normalized'] == cat]
    total = len(cat_df)

    positive_pct = (cat_df['llm_impact'] == 'positive').sum() / total * 100
    neutral_pct = (cat_df['llm_impact'] == 'neutral').sum() / total * 100
    negative_pct = (cat_df['llm_impact'] == 'negative').sum() / total * 100

    fingerprint_cat_data.append({
        'category': cat,
        'positive': positive_pct,
        'neutral': neutral_pct,
        'negative': negative_pct,
        'n': total
    })

fingerprint_cat_df = pd.DataFrame(fingerprint_cat_data)

# =============================================================================
# COMBINED VISUALIZATION: Timeseries (2x2) + Fingerprint (1 column with 2 rows)
# =============================================================================
print("\n" + "="*80)
print("Creating combined visualization")
print("="*80)

fig = plt.figure(figsize=(18, 7), facecolor='white')
# Use 5 columns: col0 (timeseries1), col1 (gap1), col2 (timeseries2), col3 (gap2-LARGE), col4 (fingerprint)
# Width ratios: [0.9 for plot, 0.15 for gap1, 0.9 for plot, 0.5 for LARGE gap2, 1.0 for fingerprint]
gs = fig.add_gridspec(2, 5, hspace=0.15, wspace=0.0,
                      width_ratios=[0.9, 0.15, 0.9, 0.5, 1.0])

# Columns: 0=timeseries1, 1=small gap, 2=timeseries2, 3=large gap, 4=fingerprint
ax1 = fig.add_subplot(gs[0, 0])  # Top-left: Overall volume
ax2 = fig.add_subplot(gs[0, 2], sharex=ax1)  # Top-right: Overall sentiment
ax3 = fig.add_subplot(gs[1, 0], sharex=ax1)  # Bottom-left: Conditions volume
ax4 = fig.add_subplot(gs[1, 2], sharex=ax1)  # Bottom-right: Conditions sentiment

# Column 4: Fingerprint subplot spanning both rows

# =============================================================================
# SUBPLOT A: Overall post volume over time
# =============================================================================
color_vol = '#84ADDC'  # Soft blue
# Create semi-transparent marker fill color
import matplotlib.colors as mcolors
marker_fill_vol = mcolors.to_rgba(color_vol, alpha=0.4)
ax1.plot(monthly_volume.index, monthly_volume.values,
         marker='o', linewidth=2, color=color_vol, markersize=6,
         markerfacecolor=marker_fill_vol, markeredgewidth=1.5, markeredgecolor=color_vol)
ax1.set_ylabel('Number of Posts', fontweight='bold')
ax1.grid(axis='both', alpha=0.5, linestyle='-', linewidth=0.5)
ax1.set_axisbelow(True)

# Add key time points as vertical lines with non-overlapping labels
for event_date, event_names in key_events_by_month.items():
    if monthly_volume.index.min() <= event_date <= monthly_volume.index.max():
        ax1.axvline(x=event_date, color='#7A8A80', linestyle='--', linewidth=0.8, alpha=0.8)
        y_max = ax1.get_ylim()[1]
        for idx, event_name in enumerate(event_names):
            y_pos = y_max * (0.95 - idx * 0.18)
            ax1.text(event_date, y_pos, event_name, rotation=90,
                    verticalalignment='top', alpha=1.0, color='black')

ax1.text(-0.1, 1.10, 'a.', transform=ax1.transAxes,
         fontweight='bold', fontsize=16, va='top')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.tick_params(labelbottom=False)

# =============================================================================
# SUBPLOT B: Overall average sentiment over time
# =============================================================================
color_sent = '#FFA288'  # Coral/salmon
marker_fill_sent = mcolors.to_rgba(color_sent, alpha=0.4)
yerr_lower = monthly_stats['mean'] - monthly_stats['ci_lower']
yerr_upper = monthly_stats['ci_upper'] - monthly_stats['mean']

ax2.errorbar(monthly_avg_sentiment.index, monthly_avg_sentiment.values,
             yerr=[yerr_lower, yerr_upper],
             marker='o', linewidth=2, color=color_sent, markersize=5,
             markerfacecolor=marker_fill_sent, markeredgewidth=1.5, markeredgecolor=color_sent,
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
        start_val = monthly_avg_sentiment.iloc[jan_idx]
        end_val = monthly_avg_sentiment.iloc[-1]

        ax2.annotate('', xy=(monthly_avg_sentiment.index[-1], end_val),
                    xytext=(jan_2025, start_val),
                    arrowprops=dict(arrowstyle='->', color='#7A8A80', lw=1.5, alpha=1.0))

        mid_idx = (jan_idx + len(monthly_avg_sentiment) - 1) // 3
        mid_x = monthly_avg_sentiment.index[mid_idx]
        mid_y = (start_val + end_val) / 2
        ax2.text(mid_x, mid_y, 'Declining trend',
                color='black', fontweight='bold', ha='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                         edgecolor='#7A8A80', alpha=1.0))

ax2.text(-0.1, 1.10, 'b.', transform=ax2.transAxes,
         fontweight='bold', fontsize=16, va='top')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.tick_params(labelbottom=False)

# =============================================================================
# SUBPLOT C: Mental health conditions - volume trends
# =============================================================================
for condition in top_conditions:
    volume = monthly_data[condition]['volume']
    # Break long labels into two lines
    label = condition.title()
    # Special handling for specific long condition names
    if 'attention-deficit/hyperactivity disorder' in condition.lower():
        label = 'Attention-Deficit/\nHyperactivity Disorder'
    elif len(label) > 20:
        # Find a good breaking point (e.g., space near middle)
        mid = len(label) // 2
        space_idx = label.find(' ', mid - 5, mid + 5)
        if space_idx != -1:
            label = label[:space_idx] + '\n' + label[space_idx + 1:]

    marker_fill = mcolors.to_rgba(color_map[condition], alpha=0.4)
    ax3.plot(volume.index, volume.values,
             marker='o', linewidth=2, label=label,
             color=color_map[condition], markersize=5,
             markerfacecolor=marker_fill, markeredgewidth=1.5, markeredgecolor=color_map[condition])

ax3.set_xlabel('Month', fontweight='bold')
ax3.set_ylabel('Number of Posts', fontweight='bold')
ax3.legend(loc='best', frameon=True, ncol=1, framealpha=0.9)
ax3.grid(axis='both', alpha=0.2, linestyle='-', linewidth=0.5)
ax3.set_axisbelow(True)

ax3.text(-0.1, 1.10, 'c.', transform=ax3.transAxes,
         fontweight='bold', fontsize=16, va='top')

# Format x-axis dates as YY-MM without rotation
date_fmt = DateFormatter('%y-%m')
ax3.xaxis.set_major_formatter(date_fmt)

ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)

# =============================================================================
# SUBPLOT D: Mental health conditions - sentiment trends
# =============================================================================
for condition in top_conditions:
    sentiment = monthly_data[condition]['sentiment']
    marker_fill = mcolors.to_rgba(color_map[condition], alpha=0.4)
    ax4.plot(sentiment.index, sentiment.values,
             marker='o', linewidth=2, label=condition.title(),
             color=color_map[condition], markersize=5,
             markerfacecolor=marker_fill, markeredgewidth=1.5, markeredgecolor=color_map[condition])

ax4.set_xlabel('Month', fontweight='bold')
ax4.set_ylabel('Average Sentiment Score', fontweight='bold')
ax4.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.4)
ax4.set_ylim(-0.2, 1)
ax4.grid(axis='both', alpha=0.2, linestyle='-', linewidth=0.5)
ax4.set_axisbelow(True)

ax4.text(-0.1, 1.10, 'd.', transform=ax4.transAxes,
         fontweight='bold', fontsize=16, va='top')

# Format x-axis dates as YY-MM without rotation
ax4.xaxis.set_major_formatter(date_fmt)

ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)

# =============================================================================
# SUBPLOT E & F: Fingerprint using bubble chart (from analyze_fingerprint.py)
# =============================================================================
# Combine both axes into one - using the full column 4 space
ax5 = fig.add_subplot(gs[:, 4])  # Span both rows in column 4

# Base colors for categories (from analyze_fingerprint.py)
base_colors = {
    'general': '#FFA288',
    'attention-deficit/hyperactivity disorder': '#7FBDDA',
    'anxiety disorders': '#6AD1A3',
    'depressive disorders': '#FFD47D',
    'autism spectrum disorders': '#FFA288',
    'schizophrenia spectrum disorders': '#C49892',
    'bipolar disorders': '#929EAB',
    'eating disorders': '#84ADDC',
    'conduct disorders': '#6AD1A3',
}

# Use the filtered LLMs and categories
llms_for_fingerprint = fingerprint_llms
categories_for_fingerprint = fingerprint_primary_categories

# Create y-positions for each category
y_positions = []
y_labels = []
y_ticks = []
spacing_between_categories = 1.5
current_y = 0

for i, primary_cat in enumerate(categories_for_fingerprint):
    y_positions.append((primary_cat, current_y))
    y_ticks.append(current_y)

    # Wrap long names
    display_name = primary_cat.title()
    wrapped = textwrap.fill(display_name, width=18)
    y_labels.append(wrapped)
    current_y += spacing_between_categories

# Create mapping from category to y-position
y_position_map = {primary_cat: pos for primary_cat, pos in y_positions}

# Group data for bubble chart
grouped_data = df_fingerprint.groupby(['mapped_llm_product', 'mapped_primary_mental_normalized']).size().reset_index(name='count')

# Plot bubbles
for _, row in grouped_data.iterrows():
    llm = row['mapped_llm_product']
    primary_cat = row['mapped_primary_mental_normalized']
    count = row['count']

    if llm not in llms_for_fingerprint or primary_cat not in categories_for_fingerprint:
        continue

    x_pos = llms_for_fingerprint.index(llm)
    y_pos = y_position_map.get(primary_cat)

    if y_pos is not None:
        # Calculate bubble size
        if count == 1:
            size = 4
        elif count < 50:
            size = 4 + (count - 1) * 2
        elif count < 120:
            size = 100 + (count - 50) * 1.5
        elif count < 600:
            size = 205 + (count - 120) * 0.8
        else:
            size = 590 + (count - 600) * 0.4

        color = base_colors.get(primary_cat, '#BBC7BE')
        ax5.scatter(x_pos, y_pos, s=size, color=color,
                   alpha=0.7, edgecolors='black', linewidth=0.5)

        # Add count labels
        label_offset_x = 0.18
        fontsize = 10 if count < 600 else 10
        ax5.text(x_pos + label_offset_x, y_pos, str(count),
                ha='left', va='center', fontsize=fontsize,
                fontweight='bold', color='black')

# Add vertical grid lines
for i in range(len(llms_for_fingerprint)):
    ax5.axvline(x=i, color='lightgray', linestyle='--', alpha=0.3, zorder=0)

# Customize the plot
ax5.set_xticks(range(len(llms_for_fingerprint)))
ax5.set_xticklabels(llms_for_fingerprint)
ax5.set_yticks(y_ticks)
ax5.set_yticklabels(y_labels)
ax5.set_xlabel('LLM Chatbot', fontweight='bold')
ax5.set_ylabel('Primary Mental Health Categories', fontweight='bold')
ax5.text(-0.15, 1.05, 'e.', transform=ax5.transAxes, fontsize=16, fontweight='bold', va='top')

ax5.spines['top'].set_visible(False)
ax5.spines['right'].set_visible(False)
ax5.spines['left'].set_linewidth(0.5)
ax5.spines['bottom'].set_linewidth(0.5)

ax5.set_xlim(-0.6, len(llms_for_fingerprint) - 0.3)
ax5.set_ylim(-0.8, current_y - spacing_between_categories + 0.5)

plt.savefig('images/timeseries_fingerprint.pdf', dpi=300, bbox_inches='tight', format='pdf')
plt.savefig('images/timeseries_fingerprint.png', dpi=300, bbox_inches='tight')
plt.close()

print("\nSaved: images/timeseries_fingerprint.pdf and .png")

# =============================================================================
# SUMMARY STATISTICS
# =============================================================================
print("\n" + "="*80)
print("Summary Statistics")
print("="*80)

print("\nTimeseries - Overall statistics:")
print(f"  Total posts: {len(df_filtered)}")
print(f"  Average sentiment score: {df_filtered['sentiment_score'].mean():.3f}")
print(f"  Date range: {df_filtered['date'].min()} to {df_filtered['date'].max()}")

print("\nTimeseries - Monthly statistics:")
print(f"  Average monthly volume: {monthly_volume.mean():.1f}")
print(f"  Average monthly sentiment: {monthly_avg_sentiment.mean():.3f}")

print("\nTimeseries - Statistics by condition:")
for condition in top_conditions:
    sentiment = monthly_data[condition]['sentiment']
    volume = monthly_data[condition]['volume']
    print(f"\n{condition.title()}:")
    print(f"  Total posts: {volume.sum()}")
    print(f"  Average monthly volume: {volume.mean():.1f}")
    print(f"  Average sentiment: {sentiment.mean():.3f}")
    print(f"  Sentiment range: [{sentiment.min():.3f}, {sentiment.max():.3f}]")

print("\nFingerprint - Distribution by LLM:")
for _, row in fingerprint_df.iterrows():
    print(f"  {row['llm']}: Pos={row['positive']:.1f}%, Neu={row['neutral']:.1f}%, Neg={row['negative']:.1f}% (n={row['n']})")

print("\nFingerprint - Distribution by category:")
for _, row in fingerprint_cat_df.iterrows():
    print(f"  {row['category']}: Pos={row['positive']:.1f}%, Neu={row['neutral']:.1f}%, Neg={row['negative']:.1f}% (n={row['n']})")

print("\n" + "="*80)
print("Analysis complete!")
print("="*80)
print("\nGenerated files:")
print("  - images/combined_timeseries_fingerprint.pdf and .png")
