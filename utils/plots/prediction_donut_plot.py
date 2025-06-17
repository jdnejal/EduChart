import plotly.graph_objects as go
import pandas as pd
from utils.data_processing import get_color_schemes

def create_prediction_donut_plot(scaler, prediction_model, study_hours_per_day, mental_health_rating,
                                social_media_hours, sleep_hours, netflix_hours, exercise_frequency,
                                attendance_percentage, color_scheme):
    """Create donut chart showing predicted exam score"""
    # Create input data with all 7 features in the correct order
    input_data = pd.DataFrame([{
        "study_hours_per_day": study_hours_per_day,
        "mental_health_rating": mental_health_rating,
        "social_media_hours": social_media_hours,
        "sleep_hours": sleep_hours,
        "netflix_hours": netflix_hours,
        "exercise_frequency": exercise_frequency,
        "attendance_percentage": attendance_percentage
    }])
    
    input_scaled = scaler.transform(input_data)
    prediction = min(prediction_model.predict(input_scaled)[0], 100)
    
    # Use dynamic color based on prediction bucket
    color_schemes = get_color_schemes()
    scheme = color_schemes.get(color_scheme, color_schemes['default'])
    colors = [scheme[2] if prediction >= 80 else scheme[1] if prediction >= 50 else scheme[0], '#ecf0f1']
    
    fig = go.Figure(data=[go.Pie(
        values=[prediction, 100 - prediction],
        labels=['Predicted Score', 'Remaining'],
        hole=0.7,
        marker=dict(colors=colors, line=dict(color='white', width=2)),
        textinfo='none',
        hovertemplate='<b>%{label}</b><br>%{value:.1f}%<extra></extra>',
        direction='clockwise',  # Force clockwise direction
        sort=False,            # Don't sort values, keep order as specified
        rotation=0           # Start from top (12 o'clock position)
    )])
    
    fig.add_annotation(
        text=f"<b>{prediction:.1f}%</b>",
        x=0.5, y=0.5,
        font_size=20,
        font_color=colors[0],
        showarrow=False
    )
    
    fig.update_layout(
        showlegend=False,
        margin=dict(t=10, b=10, l=10, r=10),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig