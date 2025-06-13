import plotly.express as px
import plotly.graph_objects as go


def create_parallel_plot(df, student_count, selected_group, color_scheme, categorize_performance):
    """Create parallel coordinates plot"""
    sample_df = df.sample(n=min(student_count, len(df)), random_state=42).copy()
    sample_df['Performance_Group'] = sample_df['exam_score'].apply(categorize_performance)
    
    # Filter by selected group
    if selected_group != 'All' and selected_group in ['High (≥80%)', 'Medium (50-79%)', 'Low (<50%)']:
        filtered_df = sample_df[sample_df['Performance_Group'] == selected_group]
    else:
        filtered_df = sample_df
    
    # Updated features for parallel coordinates
    fixed_features = [
        "study_hours_per_day",
        "attendance_percentage",
        "sleep_hours",
        "mental_health_rating",
    ]
    
    if len(filtered_df) == 0:
        return go.Figure().add_annotation(
            text="No data for selected group",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=12, color="gray")
        )

    # Create label mapping
    label_map = {
        "study_hours_per_day": "Study Hours/Day",
        "attendance_percentage": "Attendance (%)",
        "sleep_hours": "Sleep Hours",
        "mental_health_rating": "Mental Health",
    }
    
    fig = px.parallel_coordinates(
        filtered_df,
        dimensions=fixed_features,
        color='exam_score',
        color_continuous_scale=color_scheme,
        labels=label_map
    )

    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=9),
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig