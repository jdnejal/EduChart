import plotly.graph_objects as go

from utils.data_processing import get_color_schemes


import plotly.graph_objects as go
from utils.data_processing import get_color_schemes


def create_categorical_bar_plot(df, student_count, selected_group, color_scheme, categorize_performance, selected_students=None, current_visibility=None):
    """Create categorical bar plot with toggleable features and preserved visibility"""
    sample_df = df.sample(n=min(student_count, len(df)), random_state=42).copy()
    sample_df['Performance_Group'] = sample_df['exam_score'].apply(categorize_performance)
    
    # Filter by selected group
    if selected_group != 'All' and selected_group in ['High (≥80%)', 'Medium (50-79%)', 'Low (<50%)']:
        filtered_df = sample_df[sample_df['Performance_Group'] == selected_group]
        title_suffix = f" - {selected_group}"
    else:
        filtered_df = sample_df
        title_suffix = " - All Students"
    
    # Filter by selected students if any
    if selected_students is not None and len(selected_students) > 0:
        filtered_df = filtered_df[filtered_df['student_id'].isin(selected_students)]
        title_suffix += f" (Selected: {len(selected_students)})"
        if len(filtered_df) == 0:
            return go.Figure().add_annotation(
                text="No data for selected students",
                xref="paper", yref="paper", x=0.5, y=0.5,
                showarrow=False, font=dict(size=12, color="gray")
            )
    
    categorical_features = [
        'gender',
        'part_time_job', 
        'diet_quality',
        'parental_education_level',
        'internet_quality',
        'extracurricular_participation'
    ]
    
    # Check which columns exist in the dataset
    available_features = [col for col in categorical_features if col in filtered_df.columns]
    
    if not available_features:
        return go.Figure().add_annotation(
            text="Categorical columns not found",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=12, color="gray")
        )
    
    fig = go.Figure()

    # Define color maps
    color_schemes = get_color_schemes()
    colors = color_schemes.get(color_scheme, color_schemes['default'])
    
    # Create a visibility map from current_visibility if provided
    visibility_map = {}
    if current_visibility:
        for i, vis in enumerate(current_visibility):
            if i < len(available_features):
                visibility_map[available_features[i]] = vis
    
    for i, feature in enumerate(available_features):
        if feature in filtered_df.columns:
            counts = filtered_df[feature].value_counts()
            
            fig.add_trace(go.Bar(
                name=feature.replace('_', ' ').title(),
                x=counts.index,
                y=counts.values,
                marker_color=colors[i % len(colors)],
                opacity=0.8,
                visible=True  # Show all categories by default instead of 'legendonly'
            ))
    
    fig.update_layout(
        yaxis_title="Count",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=9),
        margin=dict(l=30, r=30, t=40, b=30),
        legend=dict(font=dict(size=8), orientation="h", y=-0.2)
    )
    
    return fig