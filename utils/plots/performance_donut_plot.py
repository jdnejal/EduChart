import plotly.graph_objects as go

from utils.data_processing import get_color_schemes


def create_performance_donut_plot(df, student_count, color_scheme, categorize_performance):
    """Create donut chart showing performance distribution"""
    sample_df = df.sample(n=min(student_count, len(df)), random_state=42)
    sample_df['Performance_Group'] = sample_df['exam_score'].apply(categorize_performance)
    
    counts = sample_df['Performance_Group'].value_counts()
    labels = counts.index.tolist()

    # Define color maps for each color scheme (3 colors: [Low, Medium, High])
    color_schemes = get_color_schemes()

    # Default color fallback
    selected_colors = color_schemes.get(color_scheme, ['#5961a4', '#2ecc71', '#e1ff00'])

    # Map fixed labels to fixed order
    label_order = ['Low (<50%)', 'Medium (50-79%)', 'High (≥80%)']
    color_map = dict(zip(label_order, selected_colors))

    # Use colors in the order the counts are returned
    mapped_colors = [color_map.get(label, '#cccccc') for label in labels]

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=counts.values,
        hole=0.4,
        marker=dict(
            colors=mapped_colors,
            line=dict(color='white', width=2)
        ),
        textinfo='label+percent+value',
        textposition='auto',
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
    )])
    
    fig.update_layout(
        showlegend=False,
        margin=dict(t=10, b=10, l=10, r=10),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=10)
    )

    return fig