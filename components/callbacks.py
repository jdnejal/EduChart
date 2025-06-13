from dash import Input, Output, State
from utils.data_processing import categorize_performance


import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
from sklearn.manifold import TSNE

from utils.plots.categorical_bar_plot import create_categorical_bar_plot
from utils.plots.parallel_plot import create_parallel_plot
from utils.plots.performance_donut_plot import create_performance_donut_plot
from utils.plots.prediction_donut_plot import create_prediction_donut_plot
from utils.plots.scatter_tsne_plot import create_scatter_tsne_plot


def register_callbacks(app):
    """Register all callbacks for the dashboard"""
    
    # Access the globally stored data
    df = app.df
    prediction_model = app.prediction_model

    # Callbacks
    @app.callback(
        Output('popup-container', 'style'),
        Input('options-button', 'n_clicks'),
        State('popup-container', 'style'),
        prevent_initial_call=True
    )
    def toggle_popup(n_clicks, style):
        # If style is None (first time), initialize it
        if style is None:
            style = {'display': 'none'}

        # Toggle display based on current state
        is_visible = style.get('display', 'none') == 'block'
        style['display'] = 'none' if is_visible else 'block'
        return style

    @app.callback(
        Output('selected-performance-group-store', 'data'),
        Input('performance-donut-plot', 'clickData')
    )
    def update_selected_group(clickData):
        if clickData is None:
            return 'All'
        return clickData['points'][0]['label']

    @app.callback(
        Output('x-axis-dropdown', 'style'),
        Output('y-axis-dropdown', 'style'),
        Input('plot-type-dropdown', 'value')
    )
    def toggle_axis_dropdowns(plot_type):
        common_style = {'width': '85%', 'fontSize': '11px', 'display': 'inline-block'}
        
        if plot_type == 'scatter':
            return (
                {**common_style, 'visibility': 'visible'},
                {**common_style, 'visibility': 'visible'}
            )
        else:
            return (
                {**common_style, 'visibility': 'hidden'},
                {**common_style, 'visibility': 'hidden'}
            )

    @app.callback(
        Output('scatter-tsne-plot', 'figure'),
        Input('plot-type-dropdown', 'value'),
        Input('x-axis-dropdown', 'value'),
        Input('y-axis-dropdown', 'value'),
        Input('search-input', 'value'),
        Input('student-count-slider', 'value'),
        Input('selected-performance-group-store', 'data'),
        Input('color-scheme-dropdown', 'value')
    )
    def update_scatter_tsne_plot(plot_type, x_axis, y_axis, search_value, student_count, selected_group, color_scheme):
        return create_scatter_tsne_plot(df, plot_type, x_axis, y_axis, search_value, student_count, selected_group, color_scheme, categorize_performance)

    @app.callback(
        Output('performance-donut-plot', 'figure'),
        Input('student-count-slider', 'value'),
        Input('color-scheme-dropdown', 'value')
    )
    def update_performance_donut_plot(student_count, color_scheme):
        return create_performance_donut_plot(df, student_count, color_scheme, categorize_performance)

    @app.callback(
        Output('parallel-plot', 'figure'),
        Input('student-count-slider', 'value'),
        Input('selected-performance-group-store', 'data'),
        Input('color-scheme-dropdown', 'value')
    )
    def update_parallel_plot(student_count, selected_group, color_scheme):
        return create_parallel_plot(df, student_count, selected_group, color_scheme, categorize_performance)

    @app.callback(
        Output('categorical-bar-plot', 'figure'),
        Input('student-count-slider', 'value'),
        Input('selected-performance-group-store', 'data'),
        Input('color-scheme-dropdown', 'value')
    )
    def update_categorical_bar_plot(student_count, selected_group, color_scheme):
       return create_categorical_bar_plot(df, student_count, selected_group, color_scheme, categorize_performance)

    # Updated callback to include all 7 features
    @app.callback(
        Output('donut-chart', 'figure'),
        Input('study_hours_per_day', 'value'),
        Input('mental_health_rating', 'value'),
        Input('social_media_hours', 'value'),
        Input('sleep_hours', 'value'),
        Input('netflix_hours', 'value'),
        Input('exercise_frequency', 'value'),
        Input('attendance_percentage', 'value'),
        Input('color-scheme-dropdown', 'value')
    )
    def update_donut_chart(study_hours_per_day, mental_health_rating, social_media_hours,
                           sleep_hours, netflix_hours, exercise_frequency, attendance_percentage, color_scheme):

        return create_prediction_donut_plot(prediction_model, study_hours_per_day, mental_health_rating, 
                                social_media_hours, sleep_hours, netflix_hours, exercise_frequency, 
                                attendance_percentage, color_scheme)