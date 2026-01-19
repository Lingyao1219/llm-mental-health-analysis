"""
Analyze LLM impact with sentiment scores and 95% confidence intervals.

Creates visualizations:
1. Sentiment scores (averaged) with error bars for LLMs (ordered alphabetically)
2. Sentiment scores (averaged) with error bars for mental health categories (ordered alphabetically)
3. Statistical testing with chi-square tests

Sentiment scoring: positive=1, neutral=0, negative=-1
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib import rcParams
from scipy.stats import chi2_contingency
import scipy.stats as stats
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

# Professional color palette (custom)
colors_palette = ['#84ADDC', '#FFA288', '#BBC7BE', '#6AD1A3', '#FFD47D']

# Filter data: exclude "Others" LLM and "other" mental health category
df_filtered = df[
    (df['mapped_primary_mental_normalized'].notna()) &
    (df['mapped_primary_mental_normalized'] != 'none') &
    (df['mapped_primary_mental_normalized'] != 'other') &  # Exclude "other" category
    (df['mapped_llm_product'] != 'Others') &  # Exclude "Others" LLM
    (df['llm_impact'].isin(['positive', 'negative', 'neutral']))
].copy()

print(f"\nFiltered dataset (removed 'other' category and 'Others' LLM): {len(df_filtered)} rows")

# Get primary mental health categories (>= 100 occurrences)
primary_counts = df_filtered['mapped_primary_mental_normalized'].value_counts()
primary_categories = [cat for cat in primary_counts.index if primary_counts[cat] >= 100]

# Sort alphabetically
primary_categories = sorted(primary_categories)

print(f"\nPrimary categories (>= 100, alphabetically sorted): {len(primary_categories)}")
print(f"Categories: {primary_categories}")

# Filter to only include selected categories
df_filtered = df_filtered[df_filtered['mapped_primary_mental_normalized'].isin(primary_categories)].copy()

# Get LLM products with sample size >= 50
llm_counts = df_filtered['mapped_llm_product'].value_counts()
llms = sorted([llm for llm in llm_counts.index if llm_counts[llm] >= 50])

print(f"\nLLMs (>= 50, alphabetically sorted): {len(llms)}")
print(f"LLMs: {llms}")

# Show counts for each LLM
print("\nLLM counts:")
for llm in llms:
    print(f"  {llm}: {llm_counts[llm]}")

# Filter to only include selected LLMs
df_filtered = df_filtered[df_filtered['mapped_llm_product'].isin(llms)].copy()

print(f"\nFinal filtered dataset: {len(df_filtered)} rows")

# Convert sentiment to numeric scores
# positive = 1, neutral = 0, negative = -1
sentiment_map = {'positive': 1, 'neutral': 0, 'negative': -1}
df_filtered['sentiment_score'] = df_filtered['llm_impact'].map(sentiment_map)

# =============================================================================
# Function to calculate mean sentiment and confidence intervals
# =============================================================================
def calculate_sentiment_scores(df, group_col):
    """
    Calculate mean sentiment score and 95% CI for each group.

    Returns: DataFrame with columns [group, mean, ci_lower, ci_upper, sem, n]
    """
    results = []

    for group in sorted(df[group_col].unique()):
        group_data = df[df[group_col] == group]['sentiment_score']
        n = len(group_data)
        mean = group_data.mean()
        sem = group_data.sem()  # Standard error of the mean

        # Calculate 95% confidence interval
        ci = stats.t.interval(0.95, n-1, loc=mean, scale=sem)

        results.append({
            'group': group,
            'mean': mean,
            'ci_lower': ci[0],
            'ci_upper': ci[1],
            'sem': sem,
            'n': n
        })

    return pd.DataFrame(results)

# =============================================================================
# COMBINED VISUALIZATION: Sentiment scores by LLM and Mental Health Category
# =============================================================================
print("\n" + "="*80)
print("Creating combined visualization: Sentiment scores with 95% CI")
print("="*80)

# Calculate sentiment scores for LLMs
llm_scores = calculate_sentiment_scores(df_filtered, 'mapped_llm_product')

# Calculate sentiment scores for categories
cat_scores = calculate_sentiment_scores(df_filtered, 'mapped_primary_mental_normalized')

# Determine figure height based on maximum number of items to keep bar width consistent
max_items = max(len(llm_scores), len(cat_scores))
fig_height = max(6, max_items * 0.5)  # Scale height based on number of items

# Create figure with 1 row, 2 columns
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 4.5), facecolor='white')

# =============================================================================
# LEFT SUBPLOT: LLM Chatbots
# =============================================================================
# Calculate error bars
xerr_lower = llm_scores['mean'] - llm_scores['ci_lower']
xerr_upper = llm_scores['ci_upper'] - llm_scores['mean']

# Create horizontal bar plot with error bars
y_pos = np.arange(len(llm_scores))
ax1.barh(y_pos, llm_scores['mean'],
         height=0.6,  # Consistent bar width
         xerr=[xerr_lower, xerr_upper],
         color='#84ADDC', alpha=0.7,
         ecolor='black', capsize=5, error_kw={'linewidth': 2})

# Add value labels
for i, (mean_val, n) in enumerate(zip(llm_scores['mean'], llm_scores['n'])):
    ax1.text(mean_val + 0.03, i, f'{mean_val:.3f}\n(n={n})',
            va='center', fontweight='bold')

ax1.set_yticks(y_pos)
ax1.set_yticklabels(llm_scores['group'])
ax1.set_xlabel('Average Sentiment Score', fontweight='bold')
ax1.set_ylabel('LLM Chatbot', fontweight='bold')

# Add subplot label
ax1.text(-0.15, 1.02, 'a.', transform=ax1.transAxes, fontsize=16, fontweight='bold', va='top')

# Add vertical line at 0
ax1.axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)

# Add grid
ax1.grid(axis='x', alpha=0.3, linestyle='--')
ax1.set_axisbelow(True)

# Style
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# =============================================================================
# RIGHT SUBPLOT: Mental Health Categories
# =============================================================================
# Calculate error bars
xerr_lower = cat_scores['mean'] - cat_scores['ci_lower']
xerr_upper = cat_scores['ci_upper'] - cat_scores['mean']

# Create horizontal bar plot with error bars
y_pos = np.arange(len(cat_scores))
ax2.barh(y_pos, cat_scores['mean'],
         height=0.6,  # Consistent bar width
         xerr=[xerr_lower, xerr_upper],
         color='#FFA288', alpha=0.7,
         ecolor='black', capsize=5, error_kw={'linewidth': 2})

# Add value labels
for i, (mean_val, n) in enumerate(zip(cat_scores['mean'], cat_scores['n'])):
    ax2.text(mean_val + 0.04, i, f'{mean_val:.3f}\n(n={n})',
            va='center', fontweight='bold')

# Wrap y-axis labels
wrapped_labels = [textwrap.fill(str(label).title(), width=30) for label in cat_scores['group']]
ax2.set_yticks(y_pos)
ax2.set_yticklabels(wrapped_labels)
ax2.set_xlabel('Average Sentiment Score', fontweight='bold')
ax2.set_ylabel('Primary Mental Health Category', fontweight='bold')

# Add subplot label
ax2.text(-0.15, 1.02, 'b.', transform=ax2.transAxes, fontsize=16, fontweight='bold', va='top')

# Add vertical line at 0
ax2.axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)

# Add grid
ax2.grid(axis='x', alpha=0.3, linestyle='--')
ax2.set_axisbelow(True)

# Style
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('images/sentiment.pdf', dpi=300, bbox_inches='tight', format='pdf')
plt.savefig('images/sentiment.png', dpi=300, bbox_inches='tight')
plt.close()

print("Saved: images/sentiment.pdf and .png")

# Print statistics
print("\nSentiment scores by LLM (with 95% CI):")
for _, row in llm_scores.iterrows():
    print(f"  {row['group']}: {row['mean']:.3f} (95% CI: [{row['ci_lower']:.3f}, {row['ci_upper']:.3f}]), n={row['n']}")

print("\nSentiment scores by mental health category (with 95% CI):")
for _, row in cat_scores.iterrows():
    print(f"  {row['group']}: {row['mean']:.3f} (95% CI: [{row['ci_lower']:.3f}, {row['ci_upper']:.3f}]), n={row['n']}")

# =============================================================================
# 3. STATISTICAL TESTING: Chi-square tests
# =============================================================================
print("\n" + "="*80)
print("Statistical Testing: Chi-square tests")
print("="*80)

# Test 1: Impact distribution differs across LLMs?
print("\n1. Chi-square test: Impact distribution across LLMs")
print("-" * 60)

contingency_llm = pd.crosstab(
    df_filtered['mapped_llm_product'],
    df_filtered['llm_impact']
)

chi2_llm, p_llm, dof_llm, expected_llm = chi2_contingency(contingency_llm)

print(f"Chi-square statistic: {chi2_llm:.4f}")
print(f"p-value: {p_llm:.4e}")
print(f"Degrees of freedom: {dof_llm}")
print(f"Significance: {'YES (p < 0.05)' if p_llm < 0.05 else 'NO (p >= 0.05)'}")

# Test 2: Impact distribution differs across mental health categories?
print("\n2. Chi-square test: Impact distribution across mental health categories")
print("-" * 60)

contingency_category = pd.crosstab(
    df_filtered['mapped_primary_mental_normalized'],
    df_filtered['llm_impact']
)

chi2_cat, p_cat, dof_cat, expected_cat = chi2_contingency(contingency_category)

print(f"Chi-square statistic: {chi2_cat:.4f}")
print(f"p-value: {p_cat:.4e}")
print(f"Degrees of freedom: {dof_cat}")
print(f"Significance: {'YES (p < 0.05)' if p_cat < 0.05 else 'NO (p >= 0.05)'}")

# =============================================================================
# 4. SUMMARY STATISTICS
# =============================================================================
print("\n" + "="*80)
print("Summary Statistics")
print("="*80)

print("\nOverall impact distribution:")
overall_impact = df_filtered['llm_impact'].value_counts()
overall_impact_pct = df_filtered['llm_impact'].value_counts(normalize=True) * 100
print(pd.DataFrame({'Count': overall_impact, 'Percentage': overall_impact_pct.round(2)}))

print("\nLLMs ranked by average sentiment score:")
sentiment_by_llm = df_filtered.groupby('mapped_llm_product')['sentiment_score'].mean()
print(sentiment_by_llm.sort_values(ascending=False).round(3))

print("\nCategories ranked by average sentiment score:")
sentiment_by_category = df_filtered.groupby('mapped_primary_mental_normalized')['sentiment_score'].mean()
print(sentiment_by_category.sort_values(ascending=False).round(3))

print("\n" + "="*80)
print("Analysis complete!")
print("="*80)
print("\nGenerated files:")
print("  - images/impact_combined.pdf and .png")
