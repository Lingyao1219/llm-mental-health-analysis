"""
Create fingerprint visualization for mental health LLM analysis.
Shows LLM products (y-axis) vs Primary Mental Health Categories (x-axis).
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import textwrap

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
import matplotlib.colors as mcolors
import matplotlib.cm as cm

rcParams['figure.dpi'] = 500
rcParams['savefig.dpi'] = 500
rcParams['font.family'] = 'Arial'
rcParams['axes.labelsize'] = 14
rcParams['axes.titlesize'] = 14
rcParams['xtick.labelsize'] = 14
rcParams['ytick.labelsize'] = 14
rcParams['legend.fontsize'] = 14
rcParams['figure.titlesize'] = 16

# Load the cleaned dataset
df = pd.read_csv('result/final_dataset_cleaned.csv')

print(f"Loaded {len(df)} rows")
print(f"Columns: {df.columns.tolist()}")

# Normalize case for mental health columns
df['mapped_primary_mental'] = df['mapped_primary_mental'].str.lower()

# Function to normalize category names (remove "other(" wrapper)
def normalize_category(cat):
    """Remove other(...) wrapper to merge with base categories."""
    if pd.isna(cat):
        return cat
    cat_str = str(cat).strip()
    # Remove "other(" prefix and trailing ")"
    if cat_str.startswith('other('):
        cat_str = cat_str.replace('other(', '', 1).strip()
        if cat_str.endswith(')'):
            cat_str = cat_str[:-1].strip()
    return cat_str

# Apply normalization to merge categories
df['mapped_primary_mental_normalized'] = df['mapped_primary_mental'].apply(normalize_category)

# Get unique LLM products (top ones only)
llm_counts = df['mapped_llm_product'].value_counts()
print(f"\nLLM Product distribution:\n{llm_counts}")

# Use top 7 LLMs
llms = ['GPT', 'Claude', 'Gemini', 'Others', 'Grok', 'DeepSeek', 'Llama']

# Remove rows with 'none' or NaN in primary mental health
df_filtered = df[
    (df['mapped_primary_mental_normalized'].notna()) &
    (df['mapped_primary_mental_normalized'] != 'none')
].copy()

print(f"\nFiltered dataset (removed 'none'): {len(df_filtered)} rows")

# Get unique primary mental health categories and their counts (using normalized)
primary_counts = df_filtered['mapped_primary_mental_normalized'].value_counts()
print(f"\nPrimary Mental Health distribution (normalized):\n{primary_counts.head(15)}")
print(f"\nBefore normalization, there were {df_filtered['mapped_primary_mental'].nunique()} unique categories")
print(f"After normalization, there are {primary_counts.nunique()} unique categories")

# Define colors for primary categories using the consistent palette
# Colors from the figure: #6AD1A3, #7FBDDA, #BBC7BE, #FFD47D, #FFA288, #C49892, #929EAB, #84ADDC
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
    'idiopathic developmental intellectual disability': '#7FBDDA',
    'other': '#BBC7BE',
}

# Filter out categories with less than 100 total occurrences
primary_categories = [cat for cat in primary_counts.index if primary_counts[cat] >= 100]

print(f"\nTotal primary categories: {len(primary_categories)}")
print(f"Filtered categories (>= 100 occurrences): {len(primary_categories)}")

# Create the figure - smaller size
fig, ax = plt.subplots(figsize=(8, 8), facecolor='white')

# Create y-positions for each primary category (mental health categories now on y-axis)
y_positions = []
y_labels = []
y_ticks = []

spacing_between_categories = 1.5

current_y = 0
for i, primary_cat in enumerate(primary_categories):
    y_positions.append((primary_cat, current_y))
    y_ticks.append(current_y)

    # Wrap long names to up to three lines
    if isinstance(primary_cat, str):
        display_name = primary_cat.title()  # Capitalize first letter of each word
        # Wrap text to ~18 characters per line (allows 3 lines for long names)
        wrapped = textwrap.fill(display_name, width=18)
    else:
        wrapped = str(primary_cat)

    y_labels.append(wrapped)
    current_y += spacing_between_categories

# Create a mapping from primary category to y-position
y_position_map = {primary_cat: pos for primary_cat, pos in y_positions}

# Count occurrences for bubble sizes - group by primary category only (using normalized)
grouped_data = df_filtered.groupby(['mapped_llm_product', 'mapped_primary_mental_normalized']).size().reset_index(name='count')
grouped_data.rename(columns={'mapped_primary_mental_normalized': 'mapped_primary_mental'}, inplace=True)

print(f"\nGrouped data for visualization: {len(grouped_data)} rows")
print(f"Sample grouped data:\n{grouped_data.head(10)}")

# Plot the data (reversed axes: x = LLM, y = mental health category)
for _, row in grouped_data.iterrows():
    llm = row['mapped_llm_product']
    primary_cat = row['mapped_primary_mental']
    count = row['count']

    # Only plot if LLM is in our list and category is in filtered list
    if llm not in llms:
        continue
    if primary_cat not in primary_categories:
        continue

    # Get positions (reversed: x = LLM index, y = category position)
    x_pos = llms.index(llm)
    y_pos = y_position_map.get(primary_cat)

    if y_pos is not None:
        # Calculate bubble size - slightly smaller overall
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

        color = base_colors.get(primary_cat, 'gray')
        ax.scatter(x_pos, y_pos, s=size, color=color,
                  alpha=0.7, edgecolors='black', linewidth=0.5)

        # Add count labels
        label_offset_x = 0.18
        fontsize = 10 if count < 600 else 10
        ax.text(x_pos + label_offset_x, y_pos, str(count),
               ha='left', va='center', fontsize=fontsize,
               fontweight='bold', color='black')

# Add vertical grid lines for better readability
for i in range(len(llms)):
    ax.axvline(x=i, color='lightgray', linestyle='--', alpha=0.3, zorder=0)

# Customize the plot - consistent font sizes (12pt for both axes)
ax.set_xticks(range(len(llms)))
ax.set_xticklabels(llms)
ax.set_yticks(y_ticks)
ax.set_yticklabels(y_labels)

# Set axis labels with same font size
ax.set_xlabel('LLM Chatbot', fontweight='bold')
ax.set_ylabel('Primary Mental Health Categories', fontweight='bold')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(0.5)
ax.spines['bottom'].set_linewidth(0.5)

# Adjust margins to give more space for first row and column
ax.set_xlim(-0.6, len(llms) - 0.3)  # More space on left for GPT column
ax.set_ylim(-0.8, current_y - spacing_between_categories + 0.5)  # More space at bottom for first row

# Adjust layout with more left margin for y-axis labels
plt.tight_layout()
plt.subplots_adjust(left=0.32, bottom=0.12, right=0.96, top=0.97)

plt.savefig('images/fingerprint.pdf', dpi=300, bbox_inches='tight', format='pdf')
plt.savefig('images/fingerprint.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nVisualization saved to images/fingerprint.pdf and .png")
print(f"Total categories plotted: {len(primary_categories)}")