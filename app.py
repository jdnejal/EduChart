from dash import Dash, html, dcc, callback, Output, Input, State, ALL, MATCH
import plotly.express as px
import pandas as pd
import joblib
import numpy as np
import plotly.graph_objects as go
import uuid

df = pd.read_csv("./data/student_performance_large_dataset.csv")

# Load the pre-trained prediction model
prediction_model = joblib.load('models/prediction_model.pkl')

app = Dash()

selectable_features = [
    "Assignment_Completion_Rate (%)", 
    "Attendance_Rate (%)",   
    "Time_Spent_on_Social_Media (hours/week)", 
    "Study_Hours_per_Week", 
    "Online_Courses_Completed",
    "Exam_Score (%)"
]

# Custom CSS styles
tile_style = {
    'backgroundColor': 'white',
    'borderRadius': '15px',
    'boxShadow': '0 4px 15px rgba(0, 0, 0, 0.1)',
    'padding': '20px',
    'margin': '15px',
    'border': '1px solid #e0e0e0'
}

top_bar_style = {
    'display': 'flex',
    'justifyContent': 'space-between',
    'alignItems': 'center',
    'backgroundColor': '#2c3e50',
    'padding': '15px 30px',
    'boxShadow': '0 2px 10px rgba(0, 0, 0, 0.15)',
    'position': 'sticky',
    'top': '0',
    'zIndex': '1000'
}

main_container_style = {
    'backgroundColor': '#f8f9fa',
    'minHeight': '100vh',
    'fontFamily': 'Segoe UI, Tahoma, Geneva, Verdana, sans-serif'
}

# Top bar component
def create_top_bar():
    return html.Div([
        html.Div([
            html.H1("📊 EduChart", style={
                'fontSize': '28px',
                'fontWeight': '600',
                'color': 'white',
                'margin': '0',
                'letterSpacing': '1px'
            }),
        ]),
        
        html.Div([
            html.Div([
                html.Label("Sample Size:", style={
                    'color': 'white', 
                    'marginRight': '10px',
                    'fontSize': '14px',
                    'fontWeight': '500'
                }),
                dcc.Slider(
                    id='student-count-slider',
                    min=100,
                    max=min(5000, len(df)),
                    step=100,
                    value=1000,
                    marks={i: {'label': f'{i}', 'style': {'color': 'white', 'fontSize': '12px'}} 
                           for i in range(100, min(5001, len(df)+1), 1000)},
                    tooltip={"placement": "bottom", "always_visible": True}
                )
            ], style={'width': '200px', 'marginRight': '20px'}),
            
            dcc.Input(
                id='search-input',
                placeholder='🔍 Search by student ID...',
                type='text',
                debounce=True,
                style={
                    'padding': '8px 15px',
                    'fontSize': '14px',
                    'borderRadius': '25px',
                    'border': 'none',
                    'outline': 'none',
                    'width': '250px',
                    'backgroundColor': 'white'
                }
            )
        ], style={'display': 'flex', 'alignItems': 'center'})
    ], style=top_bar_style)

# Scatter plot tile
def create_scatter_tile():
    return html.Div([
        html.Div([
            html.H3("📈 Student Performance Scatter Plot", style={
                'margin': '0 0 20px 0',
                'color': '#2c3e50',
                'fontSize': '20px',
                'fontWeight': '600'
            }),
            
            html.Div([
                html.Span(id='visible-count', style={
                    'fontWeight': '600', 
                    'color': '#34495e',
                    'backgroundColor': '#ecf0f1',
                    'padding': '5px 12px',
                    'borderRadius': '15px',
                    'fontSize': '12px',
                    'marginRight': '20px'
                }),
                
                html.Div([
                    html.Label("X-axis:", style={'marginRight': '8px', 'fontWeight': '500'}),
                    dcc.Dropdown(
                        id='x-axis-dropdown',
                        options=[{'label': f.replace('_', ' ').replace(' (%)', ''), 'value': f} for f in selectable_features],
                        value='Study_Hours_per_Week',
                        clearable=False,
                        style={'width': '180px', 'fontSize': '13px'}
                    )
                ], style={'display': 'flex', 'alignItems': 'center', 'marginRight': '20px'}),
                
                html.Div([
                    html.Label("Y-axis:", style={'marginRight': '8px', 'fontWeight': '500'}),
                    dcc.Dropdown(
                        id='y-axis-dropdown',
                        options=[{'label': f.replace('_', ' ').replace(' (%)', ''), 'value': f} for f in selectable_features],
                        value='Exam_Score (%)',
                        clearable=False,
                        style={'width': '180px', 'fontSize': '13px'}
                    )
                ], style={'display': 'flex', 'alignItems': 'center'})
            ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '15px', 'flexWrap': 'wrap', 'gap': '10px'}),
            
            dcc.Graph(id='scatter-plot', style={'height': '400px'})
        ])
    ], style=tile_style)

# Prediction tile
def create_prediction_tile():
    return html.Div([
        html.H3("🎯 Exam Score Prediction", style={
            'textAlign': 'center',
            'margin': '0 0 20px 0',
            'color': '#2c3e50',
            'fontSize': '20px',
            'fontWeight': '600'
        }),
        
        dcc.Graph(id='donut-chart', style={'height': '200px'}),
        
        html.Div([
            html.Div([
                html.Label('Assignment Completion (%)', style={'fontWeight': '500', 'marginBottom': '5px'}),
                dcc.Slider(
                    id='assignment_completion', 
                    min=0, max=100, step=1, value=85,
                    marks={i: str(i) for i in range(0, 101, 25)},
                    tooltip={"placement": "bottom", "always_visible": True}
                )
            ], style={'marginBottom': '20px'}),
            
            html.Div([
                html.Label('Attendance Rate (%)', style={'fontWeight': '500', 'marginBottom': '5px'}),
                dcc.Slider(
                    id='attendance_rate', 
                    min=0, max=100, step=1, value=90,
                    marks={i: str(i) for i in range(0, 101, 25)},
                    tooltip={"placement": "bottom", "always_visible": True}
                )
            ], style={'marginBottom': '20px'}),
            
            html.Div([
                html.Label('Social Media (hrs/week)', style={'fontWeight': '500', 'marginBottom': '5px'}),
                dcc.Slider(
                    id='social_media_hours', 
                    min=0, max=40, step=1, value=10,
                    marks={i: str(i) for i in range(0, 41, 10)},
                    tooltip={"placement": "bottom", "always_visible": True}
                )
            ], style={'marginBottom': '20px'}),
            
            html.Div([
                html.Label('Study Hours/Week', style={'fontWeight': '500', 'marginBottom': '5px'}),
                dcc.Slider(
                    id='study_hours_per_week', 
                    min=0, max=40, step=1, value=15,
                    marks={i: str(i) for i in range(0, 41, 10)},
                    tooltip={"placement": "bottom", "always_visible": True}
                )
            ], style={'marginBottom': '20px'}),
            
            html.Div([
                html.Label('Online Courses Completed', style={'fontWeight': '500', 'marginBottom': '5px'}),
                dcc.Slider(
                    id='online_courses', 
                    min=0, max=20, step=1, value=2,
                    marks={i: str(i) for i in range(0, 21, 5)},
                    tooltip={"placement": "bottom", "always_visible": True}
                )
            ], style={'marginBottom': '10px'})
        ])
    ], style={**tile_style, 'height': 'fit-content'})

# Add tile button
def create_add_tile_button():
    return html.Div([
        html.Button([
            html.Span("➕", style={'marginRight': '8px', 'fontSize': '16px'}),
            "Add New Visualization"
        ], 
        id='add-tile-btn',
        style={
            'width': '100%',
            'padding': '15px',
            'backgroundColor': '#3498db',
            'color': 'white',
            'border': 'none',
            'borderRadius': '15px',
            'fontSize': '16px',
            'fontWeight': '600',
            'cursor': 'pointer',
            'transition': 'all 0.3s ease',
            'boxShadow': '0 4px 15px rgba(52, 152, 219, 0.3)'
        })
    ], style={**tile_style, 'textAlign': 'center'})

# Dynamic tile creation
def create_visualization_tile(tile_id):
    return html.Div([
        html.Div([
            html.H3("📊 Custom Visualization", style={
                'margin': '0 0 15px 0',
                'color': '#2c3e50',
                'fontSize': '18px',
                'fontWeight': '600',
                'display': 'inline-block'
            }),
            html.Button("✕", 
                id={'type': 'close-tile', 'index': tile_id},
                style={
                    'float': 'right',
                    'backgroundColor': '#e74c3c',
                    'color': 'white',
                    'border': 'none',
                    'borderRadius': '50%',
                    'width': '25px',
                    'height': '25px',
                    'cursor': 'pointer',
                    'fontSize': '12px'
                }
            )
        ]),
        
        html.Div([
            html.Label("Chart Type:", style={'fontWeight': '500', 'marginRight': '10px'}),
            dcc.Dropdown(
                id={'type': 'chart-selector', 'index': tile_id},
                options=[
                    {'label': '📈 Parallel Coordinates', 'value': 'parallel-coords'},
                    {'label': '🎯 Radar Chart', 'value': 'radar'},
                    {'label': '📦 Box Plot', 'value': 'box-plot'}
                ],
                value='parallel-coords',
                clearable=False,
                style={'width': '200px', 'fontSize': '13px'}
            )
        ], style={'marginBottom': '15px'}),
        
        html.Div([
            html.Label("Features:", style={'fontWeight': '500', 'marginBottom': '5px'}),
            dcc.Dropdown(
                id={'type': 'feature-selector', 'index': tile_id},
                options=[{'label': col.replace('_', ' '), 'value': col} 
                        for col in df.columns if col not in ['student_id', 'Student_ID']],
                value=[
                    "Assignment_Completion_Rate (%)",
                    "Attendance_Rate (%)",
                    "Time_Spent_on_Social_Media (hours/week)",
                    "Study_Hours_per_Week"
                ],
                multi=True,
                placeholder="Select features",
                style={'fontSize': '13px'}
            )
        ], style={'marginBottom': '15px'}),
        
        dcc.Graph(id={'type': 'dynamic-chart', 'index': tile_id}, style={'height': '400px'})
    ], style=tile_style, id={'type': 'tile-container', 'index': tile_id})

# Main layout
app.layout = html.Div([
    create_top_bar(),
    
    html.Div([
        # Left column - main visualizations
        html.Div([
            create_scatter_tile(),
            create_add_tile_button(),
            html.Div(id='dynamic-tiles-container', children=[])
        ], style={'width': '70%', 'display': 'inline-block', 'verticalAlign': 'top'}),
        
        # Right column - prediction
        html.Div([
            create_prediction_tile()
        ], style={'width': '28%', 'display': 'inline-block', 'verticalAlign': 'top'})
    ], style={'padding': '0 20px'})
], style=main_container_style)

# Store for managing tiles
app.layout.children.append(dcc.Store(id='tiles-store', data=[]))

# Callbacks
@app.callback(
    Output('scatter-plot', 'figure'),
    Output('visible-count', 'children'),
    Input('search-input', 'value'),
    Input('x-axis-dropdown', 'value'),
    Input('y-axis-dropdown', 'value'),
    Input('student-count-slider', 'value')
)
def update_scatter_plot(search_value, x_axis, y_axis, student_count):
    filtered_df = df.sample(n=min(student_count, len(df)), random_state=42).copy()

    if search_value and search_value.strip():
        matched = filtered_df[filtered_df['Student_ID'].astype(str).str.contains(search_value.strip(), case=False)]
        if not matched.empty:
            student = matched.iloc[0]
            zoom_x = [max(0, student[x_axis] - 2), student[x_axis] + 2]
            zoom_y = [max(0, student[y_axis] - 10), student[y_axis] + 10]
        else:
            zoom_x = zoom_y = None
    else:
        zoom_x = zoom_y = None

    fig = px.scatter(
        filtered_df,
        x=x_axis,
        y=y_axis,
        size="Attendance_Rate (%)",
        color="Exam_Score (%)",
        size_max=12,
        hover_data=["Student_ID"],
        color_continuous_scale="viridis"
    )
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=12),
        margin=dict(l=40, r=40, t=40, b=40)
    )

    if zoom_x and zoom_y:
        fig.update_layout(xaxis_range=zoom_x, yaxis_range=zoom_y)
        fig.add_trace(go.Scatter(
            x=[student[x_axis]],
            y=[student[y_axis]],
            mode='markers+text',
            marker=dict(color='red', size=15, symbol='star'),
            text=[f"ID: {student['Student_ID']}"],
            textposition='top center',
            showlegend=False
        ))

    return fig, f"📊 {len(filtered_df):,} students"

@app.callback(
    Output('donut-chart', 'figure'),
    Input('assignment_completion', 'value'),
    Input('attendance_rate', 'value'),
    Input('social_media_hours', 'value'),
    Input('study_hours_per_week', 'value'),
    Input('online_courses', 'value')
)
def update_donut_chart(assignment_completion, attendance_rate, social_media_hours,
                       study_hours_per_week, online_courses):

    input_data = pd.DataFrame([{
        "Assignment_Completion_Rate (%)": assignment_completion,
        "Attendance_Rate (%)": attendance_rate,
        "Time_Spent_on_Social_Media (hours/week)": social_media_hours,
        "Study_Hours_per_Week": study_hours_per_week,
        "Online_Courses_Completed": online_courses
    }])

    prediction = min(prediction_model.predict(input_data)[0], 100)
    
    colors = ['#2ecc71' if prediction >= 70 else '#f39c12' if prediction >= 50 else '#e74c3c', '#ecf0f1']

    fig = go.Figure(data=[go.Pie(
        values=[prediction, 100 - prediction],
        labels=['Predicted Score', 'Remaining'],
        hole=0.7,
        marker=dict(colors=colors, line=dict(color='white', width=2)),
        textinfo='none',
        hovertemplate='<b>%{label}</b><br>%{value:.1f}%<extra></extra>'
    )])
    
    fig.add_annotation(
        text=f"<b>{prediction:.1f}%</b>",
        x=0.5, y=0.5,
        font_size=24,
        font_color=colors[0],
        showarrow=False
    )
    
    fig.update_layout(
        showlegend=False,
        margin=dict(t=20, b=20, l=20, r=20),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )

    return fig

@app.callback(
    Output('dynamic-tiles-container', 'children'),
    Output('tiles-store', 'data'),
    Input('add-tile-btn', 'n_clicks'),
    Input({'type': 'close-tile', 'index': ALL}, 'n_clicks'),
    State('tiles-store', 'data'),
    prevent_initial_call=True
)
def manage_tiles(add_clicks, close_clicks, current_tiles):
    from dash import ctx
    
    if not ctx.triggered:
        return [], current_tiles
    
    trigger_id = ctx.triggered[0]['prop_id']
    
    if 'add-tile-btn' in trigger_id and add_clicks:
        new_tile_id = str(uuid.uuid4())
        current_tiles.append(new_tile_id)
    elif 'close-tile' in trigger_id:
        # Extract tile ID from the trigger
        import json
        trigger_data = json.loads(trigger_id.split('.')[0])
        tile_to_remove = trigger_data['index']
        if tile_to_remove in current_tiles:
            current_tiles.remove(tile_to_remove)
    
    # Generate tile components
    tiles = [create_visualization_tile(tile_id) for tile_id in current_tiles]
    
    return tiles, current_tiles

@app.callback(
    Output({'type': 'dynamic-chart', 'index': MATCH}, 'figure'),
    Input({'type': 'chart-selector', 'index': MATCH}, 'value'),
    Input({'type': 'feature-selector', 'index': MATCH}, 'value'),
    Input('search-input', 'value'),
    Input('student-count-slider', 'value'),
    prevent_initial_call=True
)
def update_dynamic_chart(chart_type, selected_features, search_student, student_count):
    if not selected_features or len(selected_features) < 2:
        return go.Figure().add_annotation(
            text="Please select at least 2 features",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="gray")
        )

    sample_df = df.sample(n=min(student_count, len(df)), random_state=42)
    features = selected_features[:5]  # Limit to 5 for performance

    if chart_type == 'parallel-coords':
        fig = px.parallel_coordinates(
            sample_df,
            dimensions=features,
            color='Exam_Score (%)',
            color_continuous_scale='viridis'
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=40, r=40, t=40, b=40)
        )
        return fig

    elif chart_type == 'radar':
        if search_student and search_student.strip():
            matched = sample_df[sample_df['Student_ID'].astype(str).str.contains(search_student.strip(), case=False)]
            student_row = matched.iloc[0] if not matched.empty else sample_df.iloc[0]
        else:
            student_row = sample_df.iloc[0]

        top_performers = sample_df[sample_df['Exam_Score (%)'] >= sample_df['Exam_Score (%)'].quantile(0.8)]
        top_avg = top_performers[features].mean()

        # Normalize data
        feature_min = [sample_df[feat].min() for feat in features]
        feature_max = [sample_df[feat].max() for feat in features]
        
        def normalize(values, min_vals, max_vals):
            return [(v - min_v) / (max_v - min_v) if max_v > min_v else 0
                    for v, min_v, max_v in zip(values, min_vals, max_vals)]

        student_data = [student_row[feat] for feat in features]
        student_norm = normalize(student_data, feature_min, feature_max)
        top_avg_norm = normalize(top_avg.values, feature_min, feature_max)

        theta_labels = [label.replace('_', ' ').replace(' (%)', '') for label in features]
        
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=student_norm + [student_norm[0]],
            theta=theta_labels + [theta_labels[0]],
            fill='toself',
            name='Selected Student',
            line_color='#3498db'
        ))
        fig.add_trace(go.Scatterpolar(
            r=top_avg_norm + [top_avg_norm[0]],
            theta=theta_labels + [theta_labels[0]],
            fill='toself',
            name='Top Performers Avg',
            line_color='#e74c3c',
            opacity=0.7
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=40, r=40, t=40, b=40)
        )
        return fig

    elif chart_type == 'box-plot':
        df_long = sample_df.melt(
            id_vars=["Student_ID"], 
            value_vars=features, 
            var_name="Feature", 
            value_name="Value"
        )
        fig = px.box(
            df_long, 
            x="Feature", 
            y="Value", 
            color="Feature",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig.update_layout(
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=40, r=40, t=40, b=40)
        )
        fig.update_xaxes(tickangle=45)
        return fig

    return go.Figure()

if __name__ == '__main__':
    app.run(debug=True)