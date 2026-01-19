"""
Correlation Analysis: Mental Health Conditions vs. Values, Perspectives, and Sentiment

Creates two separate figures:
- Figure 1: Mental Health vs. Sentiment, Value, and Perspective (3 heatmaps in one row)
- Figure 2: Value vs. Perspective correlation (separate figure)
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from matplotlib import rcParams
from scipy.stats import chi2_contingency

# ==========================================
# SETUP
# ==========================================
rcParams['figure.dpi'] = 500
rcParams['savefig.dpi'] = 500
rcParams['font.family'] = 'Arial'
rcParams['font.size'] = 12
rcParams['axes.labelsize'] = 12
rcParams['axes.titlesize'] = 14
rcParams['xtick.labelsize'] = 14
rcParams['ytick.labelsize'] = 14
rcParams['legend.fontsize'] = 14
rcParams['figure.titlesize'] = 14

# ==========================================
# CUSTOM COLOR PALETTE
# ==========================================
C_BLUE = '#84ADDC'    # Soft blue
C_CORAL = '#FFA288'   # Coral/salmon

# Custom Diverging Colormap (Negative -> Neutral -> Positive)
# Coral -> White -> Blue (changed from Mint to Blue)
cmap_diverging = mcolors.LinearSegmentedColormap.from_list(
    "custom_diverging", [C_CORAL, '#F5F5F5', C_BLUE]
)

# ==========================================
# DATA LOADING
# ==========================================
print("Loading dataset...")
df = pd.read_csv('data/correlation_dataset.csv')

# Standardize column names
df.columns = df.columns.str.strip().str.lower()
print(f"Loaded {len(df)} rows")

# Define column names
col_sentiment = 'llm_impact'
col_mental = 'mapped_primary_mental'
col_value = 'value_combined'
col_perspective = 'perspective_combined'

# Clean data: select relevant columns and drop NA
target_cols = [c for c in [col_sentiment, col_mental, col_value, col_perspective] if c in df.columns]
df_clean = df[target_cols].dropna()

# Filter out 'none' from mental health conditions
if col_mental in df_clean.columns:
    initial_count = len(df_clean)
    df_clean = df_clean[df_clean[col_mental].str.lower() != 'none']
    final_count = len(df_clean)
    print(f"Filtered 'none' from mental health conditions. Rows reduced from {initial_count} to {final_count}.")

# ==========================================
# ACRONYM MAPPING
# ==========================================
acronym_map = {
    # Values
    'Accountability': 'V_Ac',
    'Autonomy': 'V_Au',
    'Calmness': 'V_Ca',
    'Freedom From Bias': 'V_Ffb',
    'Human Welfare': 'V_HW',
    'Identity': 'V_Id',
    'Informed Consent': 'V_IC',
    'Ownership and Property': 'V_OP',
    'Privacy': 'V_Pr',
    'Trust': 'V_Tr',
    'Environmental Sustainability': 'V_ES',
    'Universal Usability': 'V_UU',

    # Perspectives
    'Anthropomorphism': 'P_An',
    'Appraisal Support': 'P_AS',
    'Clinical Skepticism': 'P_CS',
    'Dependency': 'P_De',
    'Emotional Support': 'P_ES',
    'Ethics': 'P_Et',
    'Informational Support': 'P_InfS',
    'Instrumental Support': 'P_InsS',
    'Interaction Limitations': 'P_IntL',
    'Maladaptive Usage': 'P_MU',
    'Psychological Harm': 'P_PH',
    'Sociocultural Impact': 'P_SI'
}

print("Applying acronym mapping...")
if col_value in df_clean.columns:
    df_clean[col_value] = df_clean[col_value].astype(str).str.strip().replace(acronym_map)
if col_perspective in df_clean.columns:
    df_clean[col_perspective] = df_clean[col_perspective].astype(str).str.strip().replace(acronym_map)

# ==========================================
# RESIDUAL ANALYSIS FUNCTION
# ==========================================
def calculate_residuals(data, row, col):
    """Calculate standardized residuals from chi-square test."""
    ct = pd.crosstab(data[row], data[col])
    chi2, p, dof, ex = chi2_contingency(ct)

    # Calculate standardized residuals
    residuals = (ct - ex) / np.sqrt(ex)

    # Calculate effect size (Cramér's V)
    n = ct.sum().sum()
    v = np.sqrt((chi2/n) / min(ct.shape[0]-1, ct.shape[1]-1))

    print(f"\nChi-Square Statistics:")
    print(f"  Chi-Square = {chi2:.2f}")
    print(f"  p-value = {p:.4f}")
    print(f"  Cramér's V = {v:.3f}")

    return residuals

# ==========================================
# FIGURE 1: Mental Health vs. Sentiment, Perspective, Value
# ==========================================
print("\nCreating Figure 1: Mental Health correlation heatmaps...")

# Create figure with 3 columns, sharing one colorbar
# Adjusted width ratios based on number of columns in each heatmap
fig1 = plt.figure(figsize=(18, 5.5))
gs1 = fig1.add_gridspec(1, 4, wspace=0.15, width_ratios=[3, 12, 12, 0.5])

# ==========================================
# SUBPLOT A: Mental Health vs. Sentiment (3 categories)
# ==========================================
print("\n" + "="*80)
print("Subplot (a): Mental Health vs. Sentiment")
print("="*80)

ax1 = fig1.add_subplot(gs1[0, 0])

if col_mental in df_clean.columns and col_sentiment in df_clean.columns:
    residuals_a = calculate_residuals(df_clean, col_mental, col_sentiment)

    sns.heatmap(
        residuals_a, annot=True, fmt='.1f',
        cmap=cmap_diverging, center=0, vmin=-4, vmax=4,
        linewidths=2, linecolor='white',
        cbar=False,  # No individual colorbar
        ax=ax1
    )

    ax1.set_title('a. Sentiment', fontsize=14, fontweight='bold', pad=10)
    ax1.set_ylabel('')
    ax1.set_xlabel('')
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')

    # Keep y-axis labels and wrap long labels
    y_labels = ax1.get_yticklabels()
    wrapped_labels = []
    for label in y_labels:
        text = label.get_text()
        # Break long labels into two lines
        if len(text) > 20:
            words = text.split()
            mid = len(words) // 2
            wrapped_text = ' '.join(words[:mid]) + '\n' + ' '.join(words[mid:])
            wrapped_labels.append(wrapped_text)
        else:
            wrapped_labels.append(text)
    ax1.set_yticklabels(wrapped_labels, rotation=0)

    for spine in ax1.spines.values():
        spine.set_visible(False)
    ax1.tick_params(axis='both', which='both', length=0)

# ==========================================
# SUBPLOT B: Mental Health vs. Perspective (12 categories)
# ==========================================
print("\n" + "="*80)
print("Subplot (b): Mental Health vs. Perspective")
print("="*80)

ax2 = fig1.add_subplot(gs1[0, 1])

if col_mental in df_clean.columns and col_perspective in df_clean.columns:
    residuals_b = calculate_residuals(df_clean, col_mental, col_perspective)

    sns.heatmap(
        residuals_b, annot=True, fmt='.1f',
        cmap=cmap_diverging, center=0, vmin=-4, vmax=4,
        linewidths=2, linecolor='white',
        cbar=False,
        ax=ax2
    )

    ax2.set_title('b. User Perspective', fontsize=14, fontweight='bold', pad=10)
    ax2.set_ylabel('')
    ax2.set_xlabel('')
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
    ax2.set_yticklabels([])  # Remove y-axis labels

    for spine in ax2.spines.values():
        spine.set_visible(False)
    ax2.tick_params(axis='both', which='both', length=0)

# ==========================================
# SUBPLOT C: Mental Health vs. Value (12 categories)
# ==========================================
print("\n" + "="*80)
print("Subplot (c): Mental Health vs. Value")
print("="*80)

ax3 = fig1.add_subplot(gs1[0, 2])

if col_mental in df_clean.columns and col_value in df_clean.columns:
    residuals_c = calculate_residuals(df_clean, col_mental, col_value)

    sns.heatmap(
        residuals_c, annot=True, fmt='.1f',
        cmap=cmap_diverging, center=0, vmin=-4, vmax=4,
        linewidths=2, linecolor='white',
        cbar=False,
        ax=ax3
    )

    ax3.set_title('c. User Value', fontsize=14, fontweight='bold', pad=10)
    ax3.set_ylabel('')
    ax3.set_xlabel('')
    ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, ha='right')
    ax3.set_yticklabels([])  # Remove y-axis labels

    for spine in ax3.spines.values():
        spine.set_visible(False)
    ax3.tick_params(axis='both', which='both', length=0)

# ==========================================
# ADD SINGLE SHARED COLORBAR
# ==========================================
cax = fig1.add_subplot(gs1[0, 3])
sm = plt.cm.ScalarMappable(cmap=cmap_diverging, norm=plt.Normalize(vmin=-4, vmax=4))
sm.set_array([])
cbar = fig1.colorbar(sm, cax=cax)
cbar.set_label('Residuals', rotation=270, labelpad=20)

# ==========================================
# SAVE FIGURE 1
# ==========================================
plt.tight_layout()
output_path_pdf1 = 'images/heatmap.pdf'
output_path_png1 = 'images/heatmap.png'
fig1.savefig(output_path_pdf1, format='pdf', dpi=300, bbox_inches='tight')
fig1.savefig(output_path_png1, format='png', dpi=300, bbox_inches='tight')
print(f"\nFigure 1 saved to: {output_path_pdf1}")
print(f"Figure 1 saved to: {output_path_png1}")

plt.show()

# ==========================================
# FIGURE 2: Value vs. Perspective Correlation (using Delta PMI)
# ==========================================
print("\n" + "="*80)
print("Creating Figure 2: Value vs. Perspective correlation (Delta PMI)")
print("="*80)

# Load PMI difference data
df_pmi = pd.read_csv('data/correlation_pmi_difference.csv')
df_pmi.columns = df_pmi.columns.str.strip()

# Clean the data
df_pmi['Value'] = df_pmi['Value'].astype(str).str.strip()
df_pmi['Perspective'] = df_pmi['Perspective'].astype(str).str.strip()

# Apply acronym mapping
label_mapping = {
    # Values (V_)
    "V_Accountability": "V_Ac",
    "V_Autonomy": "V_Au",
    "V_Calmness": "V_Ca",
    "V_Freedom From Bias": "V_Ffb",
    "V_Human Welfare": "V_HW",
    "V_Identity": "V_Id",
    "V_Informed Consent": "V_IC",
    "V_Ownership and Property": "V_OP",
    "V_Privacy": "V_Pr",
    "V_Trust": "V_Tr",
    "V_Evironmental Sustainability": "V_ES",
    "V_Environmental Sustainability": "V_ES",
    "V_Universal Usability": "V_UU",

    # Perspectives (P_)
    "P_Anthropomorphism": "P_An",
    "P_Appraisal Support": "P_AS",
    "P_Clinical Skepticism": "P_CS",
    "P_Dependency": "P_De",
    "P_Emotional Support": "P_ES",
    "P_Ethics": "P_Et",
    "P_Informational Support": "P_InfS",
    "P_Instrumental Support": "P_InsS",
    "P_Interaction Limitations": "P_IntL",
    "P_Maladaptive Usage": "P_MU",
    "P_Psychological Harm": "P_PH",
    "P_Sociocultural Impact": "P_SI"
}

df_pmi['Value'] = df_pmi['Value'].replace(label_mapping)
df_pmi['Perspective'] = df_pmi['Perspective'].replace(label_mapping)

# Pivot the data
pivot_table = df_pmi.pivot_table(index='Value', columns='Perspective', values='Delta_PMI')

print(f"Delta PMI range: {pivot_table.min().min():.2f} to {pivot_table.max().max():.2f}")

fig2, ax_vp = plt.subplots(figsize=(8, 6))

sns.heatmap(
    pivot_table, annot=True, fmt='.2f',
    cmap=cmap_diverging, center=0,
    linewidths=2, linecolor='white',
    cbar_kws={'label': 'Delta PMI'},
    ax=ax_vp
)

ax_vp.set_ylabel('')
ax_vp.set_xlabel('')
ax_vp.set_xticklabels(ax_vp.get_xticklabels(), rotation=45, ha='right')
ax_vp.set_yticklabels(ax_vp.get_yticklabels(), rotation=0)

for spine in ax_vp.spines.values():
    spine.set_visible(False)
ax_vp.tick_params(axis='both', which='both', length=0)

# ==========================================
# SAVE FIGURE 2
# ==========================================
plt.tight_layout()
output_path_pdf2 = 'images/heatmap_pmi.pdf'
output_path_png2 = 'images/heatmap_pmi.png'
fig2.savefig(output_path_pdf2, format='pdf', dpi=300, bbox_inches='tight')
fig2.savefig(output_path_png2, format='png', dpi=300, bbox_inches='tight')
print(f"\nFigure 2 saved to: {output_path_pdf2}")
print(f"Figure 2 saved to: {output_path_png2}")

plt.show()
