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

    # NEW: Callback to capture selected data from scatter/t-SNE plot
    @app.callback(
        Output('selected-students-store', 'data'),
        Input('scatter-tsne-plot', 'selectedData'),
        Input('scatter-tsne-plot', 'figure'),  # Reset selection when plot changes
        Input('ai-student-selection-store', 'data'),  # NEW: Listen to AI selections
        State('student-count-slider', 'value'),
        State('selected-performance-group-store', 'data')
    )
    def update_selected_students(selectedData, figure, ai_selection, student_count, selected_group):
        """Store selected student IDs from scatter/t-SNE plot or AI chat"""
        from dash import callback_context
        
        ctx = callback_context
        if not ctx.triggered:
            return None
        
        # Check which input triggered the callback
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        # If triggered by AI, return the AI-selected students
        if trigger_id == 'ai-student-selection-store':
            if ai_selection and len(ai_selection) > 0:
                print(f"AI selected students: {ai_selection}")
                return ai_selection
            return None
        
        # Original plot selection logic
        print("=== DEBUG: Plot selection callback triggered ===")
        print(f"selectedData: {selectedData}")
        
        if selectedData is None:
            print("No selectedData")
            return None
            
        if selectedData.get('points') is None:
            print("No points in selectedData")
            return None
        
        print(f"Number of selected points: {len(selectedData['points'])}")
        
        # Extract student IDs from selected points
        selected_student_ids = []
        for i, point in enumerate(selectedData['points']):
            print(f"Point {i}: {point}")
            
            # Check different possible locations for student_id
            if 'customdata' in point and point['customdata']:
                print(f"  customdata: {point['customdata']}")
                student_id = point['customdata'][0]  # student_id is first in customdata
                selected_student_ids.append(student_id)
            elif 'hovertext' in point:
                print(f"  hovertext: {point['hovertext']}")
            elif 'text' in point:
                print(f"  text: {point['text']}")
            else:
                print(f"  No student_id found in point")
        
        print(f"Selected student IDs: {selected_student_ids}")
        return selected_student_ids if selected_student_ids else None

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

    # UPDATED: Performance donut plot with selection filtering
    @app.callback(
        Output('performance-donut-plot', 'figure'),
        Input('student-count-slider', 'value'),
        Input('color-scheme-dropdown', 'value'),
        Input('selected-students-store', 'data')  # NEW: Add selection input
    )
    def update_performance_donut_plot(student_count, color_scheme, selected_students):
        # Add this debugging line to each callback
        if selected_students:
            print(f"DEBUG: Filtering {len(df)} students to {selected_students}")
            # Check if filtering works
            if 'student_id' in df.columns:
                filtered = df[df['student_id'].astype(str).isin([str(s) for s in selected_students])]
                print(f"DEBUG: Found {len(filtered)} matching students")
            else:
                print("DEBUG: No student_id column found!")

        return create_performance_donut_plot(df, student_count, color_scheme, categorize_performance, selected_students)

    # UPDATED: Parallel plot with selection filtering
    @app.callback(
        Output('parallel-plot', 'figure'),
        Input('student-count-slider', 'value'),
        Input('selected-performance-group-store', 'data'),
        Input('color-scheme-dropdown', 'value'),
        Input('selected-students-store', 'data')  # NEW: Add selection input
    )
    def update_parallel_plot(student_count, selected_group, color_scheme, selected_students):
        # Add this debugging line to each callback
        if selected_students:
            print(f"DEBUG: Filtering {len(df)} students to {selected_students}")
            # Check if filtering works
            if 'student_id' in df.columns:
                filtered = df[df['student_id'].astype(str).isin([str(s) for s in selected_students])]
                print(f"DEBUG: Found {len(filtered)} matching students")
            else:
                print("DEBUG: No student_id column found!")

        return create_parallel_plot(df, student_count, selected_group, color_scheme, categorize_performance, selected_students)

    # UPDATED: Categorical bar plot with selection filtering
    @app.callback(
        Output('categorical-bar-plot', 'figure'),
        Input('student-count-slider', 'value'),
        Input('selected-performance-group-store', 'data'),
        Input('color-scheme-dropdown', 'value'),
        Input('selected-students-store', 'data')  # NEW: Add selection input
    )
    def update_categorical_bar_plot(student_count, selected_group, color_scheme, selected_students):
       # Add this debugging line to each callback
       if selected_students:
            print(f"DEBUG: Filtering {len(df)} students to {selected_students}")
            # Check if filtering works
            if 'student_id' in df.columns:
                filtered = df[df['student_id'].astype(str).isin([str(s) for s in selected_students])]
                print(f"DEBUG: Found {len(filtered)} matching students")
            else:
                print("DEBUG: No student_id column found!")
                
       return create_categorical_bar_plot(df, student_count, selected_group, color_scheme, categorize_performance, selected_students)

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
    

