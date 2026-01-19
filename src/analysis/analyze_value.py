import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

# --- 1. Define Columns and Data Loading ---

# List of Value (V_) and Perspective (P_) columns based on the data sample image
V_COLUMNS = [
    'V_Truth', 'V_Trust', 'V_Anthropis', 'V_Appropiate', 'V_Climate', 
    'V_Desensitiz', 'V_Emotion', 'V_Ethics'
]
P_COLUMNS = [
    'P_Anthropis', 'P_Appropiate', 'P_Climate', 'P_Desensitiz', 
    'P_Emotion', 'P_Ethics', 'P_Informati'
]
REQUIRED_COLUMNS = ['llm_impact', 'mapped_primary_mental'] + V_COLUMNS + P_COLUMNS
N = 500 # Used for simulation fallback

try:
    # --- Attempt to read the user's file into DataFrame 'df' ---
    # NOTE: This requires 'binary.csv' to be accessible in the current directory.
    df = pd.read_csv('binary.csv')
    
    # Filter for the specific columns requested
    df = df[REQUIRED_COLUMNS].copy()
    print("Successfully loaded 'binary.csv' and filtered columns.")

except FileNotFoundError:
    print("Warning: 'binary.csv' not found. Generating synthetic data instead to run analysis.")
    
    # Fallback to simulated data generation
    LLM_IMPACT_CATEGORIES = ['positive', 'negative', 'neutral']
    MENTAL_CATEGORIES = [
        'General', 'Depressive emotional', 'Anxiety disconfimational', 'Autism-specfied nexus'
    ]
    np.random.seed(42)

    data = {
        'llm_impact': np.random.choice(LLM_IMPACT_CATEGORIES, N, p=[0.6, 0.2, 0.2]),
        'mapped_primary_mental': np.random.choice(MENTAL_CATEGORIES, N, p=[0.4, 0.3, 0.2, 0.1]),
    }

    # Simulate binary (0/1) data for V_ and P_ columns
    for col in V_COLUMNS + P_COLUMNS:
        data[col] = np.random.binomial(1, 0.2, N)
        
    df = pd.DataFrame(data)


# --- 2. Data Preparation for Visualization ---

# 2a. Prepare data for Sunburst (Overall Value Distribution)
# Calculate the total count of activated V_ features per row
df['Total_V_Activated'] = df[V_COLUMNS].sum(axis=1)

# Filter out rows with zero activated V_ features for a cleaner Sunburst
df_sunburst = df[df['Total_V_Activated'] > 0].copy()

# Aggregate: Impact -> Mental State -> Count of rows
df_sunburst_agg = df_sunburst.groupby(
    ['llm_impact', 'mapped_primary_mental']
).size().reset_index(name='Count')


# 2b. Prepare data for Stacked Bar Chart (Feature Count by Mental State)
# Unpivot the V_ and P_ columns to have one row per activated feature
df_melted = df.melt(
    id_vars=['mapped_primary_mental'],
    value_vars=V_COLUMNS + P_COLUMNS,
    var_name='Feature',
    value_name='Activated'
)

# Filter only the activated features (where value is 1)
df_bar = df_melted[df_melted['Activated'] == 1].copy()

# Separate V_ and P_ for coloring/grouping
df_bar['Type'] = np.where(df_bar['Feature'].str.startswith('V_'), 'Value (V)', 'Perspective (P)')

# Aggregate the counts
df_bar_agg = df_bar.groupby(['mapped_primary_mental', 'Feature', 'Type']).size().reset_index(name='Frequency')


# --- 3. Visualization Generation ---

# Custom color scheme 
colors = px.colors.qualitative.Plotly + px.colors.qualitative.Dark24
impact_color_map = {'positive': colors[2], 'negative': colors[0], 'neutral': colors[7]}
feature_type_color_map = {'Value (V)': colors[9], 'Perspective (P)': colors[5]}

# A. Figure 1: Sunburst Chart (Impact -> Mental State -> Count)
fig_sunburst = px.sunburst(
    df_sunburst_agg,
    path=['llm_impact', 'mapped_primary_mental'],
    values='Count',
    color='llm_impact',
    color_discrete_map=impact_color_map,
    title='Distribution of Records with Activated Values (V) by LLM Impact & Mental State',
)
# Make it fancy
fig_sunburst.update_layout(
    margin=dict(t=50, l=0, r=0, b=0),
    title_font_size=18,
    title_font_color="#333",
    hoverlabel=dict(bgcolor="white", font_size=14, font_family="Arial")
)
fig_sunburst.update_traces(
    marker=dict(line=dict(color='#fff', width=1.5)), 
    hovertemplate='<b>%{label}</b><br>Count: %{value}<extra></extra>'
)


# B. Figure 2: Grouped Bar Chart (Feature Frequency by Mental State)
fig_bar = px.bar(
    df_bar_agg,
    x='Feature',
    y='Frequency',
    color='Type',
    facet_col='mapped_primary_mental',
    facet_col_wrap=2,
    color_discrete_map=feature_type_color_map,
    labels={'Feature': 'V_ / P_ Feature', 'Frequency': 'Activation Frequency'},
    title='Frequency of Value (V) and Perspective (P) Activations per Mental State'
)

# Enhance Bar Chart aesthetics
fig_bar.update_layout(
    title_font_size=18,
    title_font_color="#333",
    legend_title_text='Feature Type',
    height=600,
    hoverlabel=dict(bgcolor="white", font_size=14, font_family="Arial")
)
# Rotate x-axis labels for readability in facets
fig_bar.update_xaxes(
    tickangle=-45, 
    title_font=dict(size=12), 
    showgrid=False
)
fig_bar.update_yaxes(
    title_font=dict(size=12),
    showgrid=True,
    gridcolor='lightgray'
)
# Update facet titles for boldness
fig_bar.for_each_annotation(lambda a: a.update(text=f"<b>{a.text.split('=')[-1]}</b>"))


# --- 4. Display Figures ---

# Print the HTML code for the first figure (Sunburst)
print(fig_sunburst.to_html(full_html=False, include_plotlyjs='cdn'))
# Print a separator
print("<hr style='margin: 40px 0; border-top: 2px dashed #ccc;'>")
# Print the HTML code for the second figure (Grouped Bar Chart)
print(fig_bar.to_html(full_html=False, include_plotlyjs='cdn'))