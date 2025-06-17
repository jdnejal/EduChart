from dash import Input, Output, State, callback_context, no_update
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
    scaler = app.scaler

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
        if (search_value): search_value = search_value.upper()

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

    @app.callback(
        Output('selected-categories-store', 'data'),
        Input('categorical-bar-plot', 'restyleData'),
        State('selected-categories-store', 'data'),
        State('categorical-bar-plot', 'figure'),
        prevent_initial_call=True
    )
    def update_selected_categories(restyle_data, current_selected_categories, current_figure):
        """Update selected categories when legend items are clicked"""
        
        # Debug prints - remove these after testing
        print("=== DEBUG INFO ===")
        print("restyle_data:", restyle_data)
        print("current_selected_categories:", current_selected_categories)
        
        if restyle_data is None:
            print("restyle_data is None, returning current")
            return current_selected_categories
        
        # Get all available feature names (must match the ones in create_categorical_bar_plot)
        categorical_features = [
            'Gender',
            'Part Time Job', 
            'Diet Quality',
            'Parental Education Level',
            'Internet Quality',
            'Extracurricular Participation'
        ]
        
        # Initialize selected_categories if None
        if current_selected_categories is None:
            current_selected_categories = categorical_features.copy()
            print("Initialized categories:", current_selected_categories)
        
        # Process restyle data
        print("Processing restyle_data...")
        
        if isinstance(restyle_data, list) and len(restyle_data) >= 2:
            restyle_dict = restyle_data[0]
            trace_indices = restyle_data[1]
            
            print("restyle_dict:", restyle_dict)
            print("trace_indices:", trace_indices)
            
            # Handle visibility changes
            if 'visible' in restyle_dict:
                visibility_changes = restyle_dict['visible']
                print("visibility_changes:", visibility_changes)
                
                # Ensure trace_indices is a list
                if not isinstance(trace_indices, list):
                    trace_indices = [trace_indices]
                
                # Get current figure trace names to map correctly
                if current_figure and 'data' in current_figure:
                    trace_names = [trace.get('name', '') for trace in current_figure['data']]
                    print("Current trace names:", trace_names)
                    
                    updated_categories = current_selected_categories.copy()
                    
                    for i, trace_idx in enumerate(trace_indices):
                        if trace_idx < len(trace_names):
                            feature_name = trace_names[trace_idx]
                            
                            # Get the visibility value for this trace
                            if isinstance(visibility_changes, list):
                                visibility_value = visibility_changes[i] if i < len(visibility_changes) else visibility_changes[0]
                            else:
                                visibility_value = visibility_changes
                            
                            print(f"Feature: {feature_name}, visibility_value: {visibility_value}")
                            
                            # In Plotly: True = visible, False = hidden, 'legendonly' = hidden but in legend
                            is_visible = visibility_value is True
                            
                            print(f"Feature: {feature_name}, is_visible: {is_visible}")
                            
                            # Update the selected categories list
                            if is_visible and feature_name not in updated_categories:
                                updated_categories.append(feature_name)
                                print(f"Added {feature_name}")
                            elif not is_visible and feature_name in updated_categories:
                                updated_categories.remove(feature_name)
                                print(f"Removed {feature_name}")
                    
                    print("Updated categories:", updated_categories)
                    return updated_categories
        
        # Alternative approach: extract visibility directly from current figure
        if current_figure and 'data' in current_figure:
            visible_categories = []
            for trace in current_figure['data']:
                if trace.get('visible', True) is True:  # True or not specified means visible
                    visible_categories.append(trace.get('name', ''))
            
            print("Visible categories from figure:", visible_categories)
            if visible_categories:
                return visible_categories
        
        print("Returning current_selected_categories")
        return current_selected_categories

    # UPDATED: Categorical bar plot with selection filtering
    @app.callback(
        Output('categorical-bar-plot', 'figure'),
        Input('student-count-slider', 'value'),
        Input('selected-performance-group-store', 'data'),
        Input('color-scheme-dropdown', 'value'),
        Input('selected-students-store', 'data'),
        Input('selected-categories-store', 'data')
    )
    def update_categorical_bar_plot(student_count, selected_group, color_scheme, selected_students, selected_categories):
        return create_categorical_bar_plot(df, student_count, selected_group, color_scheme, categorize_performance, selected_students, selected_categories)

    # Updated callback to include all 7 features
    @app.callback(
        Output('donut-chart', 'figure'),
        Output('study_hours_per_day', 'value'),
        Output('mental_health_rating', 'value'),
        Output('social_media_hours', 'value'),
        Output('sleep_hours', 'value'),
        Output('netflix_hours', 'value'),
        Output('exercise_frequency', 'value'),
        Output('attendance_percentage', 'value'),
        Input('study_hours_per_day', 'value'),
        Input('mental_health_rating', 'value'),
        Input('social_media_hours', 'value'),
        Input('sleep_hours', 'value'),
        Input('netflix_hours', 'value'),
        Input('exercise_frequency', 'value'),
        Input('attendance_percentage', 'value'),
        Input('search-input', 'value'),
        Input('color-scheme-dropdown', 'value'),

    )
    def update_donut_chart(study_hours_per_day, mental_health_rating, social_media_hours,
                        sleep_hours, netflix_hours, exercise_frequency, attendance_percentage, 
                        search_input, color_scheme):
        
        from dash import callback_context, no_update
        
        # Check what triggered the callback
        ctx = callback_context
        triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None
        
        # If search input changed and has a value, update sliders to student's values
        if triggered_id == 'search-input' and search_input:
            search_input = search_input.upper()
            selected_row = df[df['student_id'] == search_input]
            
            if not selected_row.empty:
                # Get student's values
                student_study_hours = float(selected_row['study_hours_per_day'].iloc[0])
                student_mental_health = int(selected_row['mental_health_rating'].iloc[0])
                student_social_media = float(selected_row['social_media_hours'].iloc[0])
                student_sleep_hours = float(selected_row['sleep_hours'].iloc[0])
                student_netflix_hours = float(selected_row['netflix_hours'].iloc[0])
                student_exercise_freq = int(selected_row['exercise_frequency'].iloc[0])
                student_attendance = int(selected_row['attendance_percentage'].iloc[0])
                
                # Create donut chart with student's values
                donut_figure = create_prediction_donut_plot(
                    scaler, prediction_model, student_study_hours, student_mental_health, 
                    student_social_media, student_sleep_hours, student_netflix_hours, 
                    student_exercise_freq, student_attendance, color_scheme
                )
                
                # Return updated sliders and chart
                return (donut_figure, student_study_hours, student_mental_health, 
                        student_social_media, student_sleep_hours, student_netflix_hours, 
                        student_exercise_freq, student_attendance)
        
        # If search input was cleared, don't update sliders but update chart
        elif triggered_id == 'search-input' and not search_input:
            # Create donut chart with current slider values
            donut_figure = create_prediction_donut_plot(
                scaler, prediction_model, study_hours_per_day, mental_health_rating, 
                social_media_hours, sleep_hours, netflix_hours, exercise_frequency, 
                attendance_percentage, color_scheme
            )
            
            # Don't update sliders, just the chart
            return (donut_figure, no_update, no_update, no_update, no_update, 
                    no_update, no_update, no_update)
        
        # For all other cases (slider changes, color scheme changes), update only the chart
        else:
            # Create donut chart with current slider values
            donut_figure = create_prediction_donut_plot(
                scaler, prediction_model, study_hours_per_day, mental_health_rating, 
                social_media_hours, sleep_hours, netflix_hours, exercise_frequency, 
                attendance_percentage, color_scheme
            )
            
            # Don't update sliders, just the chart
            return (donut_figure, no_update, no_update, no_update, no_update, 
                    no_update, no_update, no_update)