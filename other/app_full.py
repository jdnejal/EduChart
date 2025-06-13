from dash import Dash, html, dcc, callback, Output, Input, State
import plotly.express as px
import pandas as pd
import joblib
import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, LabelEncoder

df = pd.read_csv("./data/student_habits_performance.csv")

# Load the pre-trained prediction model
prediction_model = joblib.load('models/prediction_model_small_data.pkl')

app = Dash()

# Updated selectable features for new dataset
selectable_features = [
    "study_hours_per_day", 
    "attendance_percentage",   
    "social_media_hours", 
    "netflix_hours", 
    "sleep_hours",
    "exercise_frequency",
    "mental_health_rating",
    "exam_score"
]

# Custom CSS styles
small_tile_style = {
    'backgroundColor': 'white',
    'borderRadius': '12px',
    'boxShadow': '0 3px 10px rgba(0, 0, 0, 0.1)',
    'padding': '15px',
    'margin': '8px',
    'border': '1px solid #e0e0e0',
    'height': '335px'
}

prediction_tile_style = {
    'backgroundColor': 'white',
    'borderRadius': '12px',
    'boxShadow': '0 3px 10px rgba(0, 0, 0, 0.1)',
    'padding': '20px',
    'margin': '8px',
    'border': '1px solid #e0e0e0',
    'height': '700px'  # Increased height to accommodate all 7 sliders
}

chatbot_tile_style = {
    'backgroundColor': 'white',
    'borderRadius': '12px',
    'boxShadow': '0 3px 10px rgba(0, 0, 0, 0.1)',
    'padding': '15px',
    'margin': '8px',
    'border': '1px solid #e0e0e0',
    'height': '150px'
}

top_bar_style = {
    'display': 'flex',
    'justifyContent': 'space-between',
    'alignItems': 'center',
    'backgroundColor': '#2c3e50',
    'padding': '12px 30px',
    'boxShadow': '0 2px 8px rgba(0, 0, 0, 0.15)',
    'position': 'sticky',
    'top': '0',
    'zIndex': '1000'
}

main_container_style = {
    'backgroundColor': '#f8f9fa',
    'height': '100vh',
    'fontFamily': 'Segoe UI, Tahoma, Geneva, Verdana, sans-serif',
    'overflow': 'hidden'
}

# Top bar component
def create_top_bar():
    return html.Div([
        html.Div([
            html.H1("📊 EduChart", style={
                'fontSize': '24px',
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
                    'marginRight': '8px',
                    'fontSize': '13px',
                    'fontWeight': '500'
                }),
                dcc.Slider(
                    id='student-count-slider',
                    min=100,
                    max=len(df),
                    step=50,
                    value=800,
                    marks={i: {'label': f'{i}', 'style': {'color': 'white', 'fontSize': '10px'}} 
                           for i in range(100, len(df)+1, 200)},
                    tooltip={"placement": "bottom", "always_visible": True}
                )
            ], style={'width': '400px', 'marginRight': '15px'}),

            html.Div([
                dcc.Input(
                    id='search-input',
                    placeholder='🔍 Search student ID...',
                    type='text',
                    debounce=True,
                    style={
                        'padding': '6px 12px',
                        'fontSize': '13px',
                        'borderRadius': '20px',
                        'border': 'none',
                        'outline': 'none',
                        'width': '200px',
                        'backgroundColor': 'white',
                        'marginRight': '10px'
                    }
                ),
                html.Button("⋮", id='options-button', n_clicks=0, title="Options", style={
                    'background': 'transparent',
                    'color': 'white',
                    'border': 'none',
                    'fontSize': '30px',
                    'cursor': 'pointer',
                    'padding': '0 8px'
                }),
                
                html.Div(id='popup-container', children=[
                    html.Div([
                        html.Button("Reset", id='reset-button', style={
                            'marginBottom': '8px',
                            'width': '100%',
                            'padding': '5px',
                            'fontSize': '12px'
                        }),
                        dcc.Dropdown(
                            id='color-scheme-dropdown',
                            options=[
                                {'label': 'Viridis', 'value': 'viridis'},
                                {'label': 'Plasma', 'value': 'plasma'},
                                {'label': 'Cividis', 'value': 'cividis'},
                                {'label': 'Inferno', 'value': 'inferno'}
                            ],
                            value='viridis',
                            clearable=False,
                            style={'fontSize': '12px'}
                        )
                    ], style={
                        'background': '#2c3e50',
                        'border': '1px solid #ccc',
                        'borderRadius': '6px',
                        'padding': '10px',
                        'boxShadow': '0 2px 6px rgba(0,0,0,0.2)',
                        'position': 'absolute',
                        'top': '45px',
                        'right': '0',
                        'zIndex': '1000',
                        'minWidth': '140px'
                    })
                ], style={'display': 'none', 'position': 'relative'})
            ], style={'position': 'relative'})
        ], style={'display': 'flex', 'alignItems': 'center'})
    ], style=top_bar_style)


# Updated tile components
def create_scatter_tsne_tile():
    return html.Div([
        html.H4("📈 Performance Analysis", style={
            'margin': '0 0 10px 0',
            'color': '#2c3e50',
            'fontSize': '16px',
            'fontWeight': '600'
        }),
        
        html.Div([
            dcc.Dropdown(
                id='plot-type-dropdown',
                options=[
                    {'label': 'Scatter Plot', 'value': 'scatter'},
                    {'label': 't-SNE Plot', 'value': 'tsne'}
                ],
                value='scatter',
                clearable=False,
                style={'width': '80%', 'fontSize': '11px', 'display': 'inline-block', 'marginRight': '10%'}
            ),
            dcc.Dropdown(
                id='x-axis-dropdown',
                options=[{'label': f.replace('_', ' ').title(), 'value': f} for f in selectable_features],
                value='study_hours_per_day',
                clearable=False,
                style={'width': '85%', 'fontSize': '11px', 'display': 'inline-block', 'visibility':'visible'}
            ),
            dcc.Dropdown(
                id='y-axis-dropdown',
                options=[{'label': f.replace('_', ' ').title(), 'value': f} for f in selectable_features],
                value='attendance_percentage',
                clearable=False,
                style={'width': '85%', 'fontSize': '11px', 'display': 'inline-block', 'visibility':'visible'}
            )
        ], style={'marginBottom': '8px','display': 'flex'}),
        
        dcc.Graph(id='scatter-tsne-plot', style={'height': '240px'})
    ], style=small_tile_style)

def create_performance_donut_tile():
    return html.Div([
        html.H4("🎯 Performance Distribution", style={
            'margin': '0 0 10px 0',
            'color': '#2c3e50',
            'fontSize': '16px',
            'fontWeight': '600'
        }),
        
        dcc.Graph(id='performance-donut-plot', style={'height': '275px'})
    ], style=small_tile_style)

def create_parallel_tile():
    return html.Div([
        html.H4("📊 Student Profiles", style={
            'margin': '0 0 10px 0',
            'color': '#2c3e50',
            'fontSize': '16px',
            'fontWeight': '600'
        }),
        
        dcc.Graph(id='parallel-plot', style={'height': '275px'})
    ], style=small_tile_style)

def create_categorical_bar_tile():
    return html.Div([
        html.H4("📋 Categorical Analysis", style={
            'margin': '0 0 10px 0',
            'color': '#2c3e50',
            'fontSize': '16px',
            'fontWeight': '600'
        }),
        
        dcc.Graph(id='categorical-bar-plot', style={'height': '275px'})
    ], style=small_tile_style)

# Updated prediction tile with all 7 features
def create_prediction_tile():
    return html.Div([
        html.H3("🎯 Score Prediction", style={
            'textAlign': 'center',
            'margin': '0 0 15px 0',
            'color': '#2c3e50',
            'fontSize': '18px',
            'fontWeight': '600'
        }),
        
        dcc.Graph(id='donut-chart', style={'height': '160px'}),
        
        html.Div([
            html.Div([
                html.Label('Study Hours/Day', style={'fontWeight': '500', 'fontSize': '12px'}),
                dcc.Slider(id='study_hours_per_day', min=0, max=8, step=0.5, value=4,
                          marks={i: str(i) for i in range(0, 13, 3)}, tooltip={"always_visible": False})
            ], style={'marginBottom': '10px'}),
            
            html.Div([
                html.Label('Mental Health Rating (1-10)', style={'fontWeight': '500', 'fontSize': '12px'}),
                dcc.Slider(id='mental_health_rating', min=1, max=10, step=1, value=7,
                          marks={i: str(i) for i in range(1, 11, 2)}, tooltip={"always_visible": False})
            ], style={'marginBottom': '10px'}),
            
            html.Div([
                html.Label('Social Media Hours', style={'fontWeight': '500', 'fontSize': '12px'}),
                dcc.Slider(id='social_media_hours', min=0, max=10, step=0.5, value=3,
                          marks={i: str(i) for i in range(0, 11, 2)}, tooltip={"always_visible": False})
            ], style={'marginBottom': '10px'}),
            
            html.Div([
                html.Label('Sleep Hours', style={'fontWeight': '500', 'fontSize': '12px'}),
                dcc.Slider(id='sleep_hours', min=4, max=12, step=0.5, value=7,
                          marks={i: str(i) for i in range(4, 13, 2)}, tooltip={"always_visible": False})
            ], style={'marginBottom': '10px'}),
            
            html.Div([
                html.Label('Netflix Hours', style={'fontWeight': '500', 'fontSize': '12px'}),
                dcc.Slider(id='netflix_hours', min=0, max=8, step=0.5, value=2,
                          marks={i: str(i) for i in range(0, 9, 2)}, tooltip={"always_visible": False})
            ], style={'marginBottom': '10px'}),
            
            html.Div([
                html.Label('Exercise Frequency (times/week)', style={'fontWeight': '500', 'fontSize': '12px'}),
                dcc.Slider(id='exercise_frequency', min=0, max=7, step=1, value=3,
                          marks={i: str(i) for i in range(0, 8, 1)}, tooltip={"always_visible": False})
            ], style={'marginBottom': '10px'}),
            
            html.Div([
                html.Label('Attendance Percentage', style={'fontWeight': '500', 'fontSize': '12px'}),
                dcc.Slider(id='attendance_percentage', min=0, max=100, step=5, value=85,
                          marks={i: str(i) for i in range(0, 101, 25)}, tooltip={"always_visible": False})
            ])
        ])
    ], style=prediction_tile_style)

# Chatbot tile (unchanged)
def create_chatbot_tile():
    return html.Div([
        html.H4("💬 AI Assistant", style={
            'margin': '0 0 10px 0',
            'color': '#2c3e50',
            'fontSize': '16px',
            'fontWeight': '600'
        }),
        
        html.Div([
            dcc.Textarea(
                id='chatbot-input',
                placeholder='Ask me about the student data...',
                style={
                    'width': '93%',
                    'height': '50px',
                    'padding': '10px',
                    'border': '1px solid #ddd',
                    'borderRadius': '8px',
                    'fontSize': '13px',
                    'resize': 'none',
                    'fontFamily': 'inherit'
                }
            ),
            html.Button('Send', 
                id='chatbot-send',
                style={
                    'marginTop': '8px',
                    'padding': '6px 15px',
                    'backgroundColor': '#3498db',
                    'color': 'white',
                    'border': 'none',
                    'borderRadius': '5px',
                    'cursor': 'pointer',
                    'fontSize': '12px',
                    'fontWeight': '500'
                }
            )
        ])
    ], style=chatbot_tile_style)

# Hidden div to store selected performance group
hidden_div = html.Div(id='selected-performance-group', style={'display': 'none'})

# Main layout
app.layout = html.Div([
    create_top_bar(),
    hidden_div,
    
    html.Div([
        # Left section - 4 visualization tiles in 2x2 grid
        html.Div([
            html.Div([
                html.Div([create_parallel_tile()], style={'width': '70%', 'display': 'inline-block'}),
                html.Div([create_performance_donut_tile()], style={'width': '30%', 'display': 'inline-block', 'float': 'right'})
            ]),
            html.Div([
                html.Div([create_scatter_tsne_tile()], style={'width': '50%', 'display': 'inline-block'}),
                html.Div([create_categorical_bar_tile()], style={'width': '50%', 'display': 'inline-block', 'float': 'right'})
            ])
        ], style={'width': '75%', 'display': 'inline-block', 'verticalAlign': 'top'}),
        
        # Right section - prediction model and chatbot
        html.Div([
            create_prediction_tile(),
            # create_chatbot_tile()
        ], style={'width': '25%', 'display': 'inline-block', 'verticalAlign': 'top'})
    ], style={'padding': '5px 5px', 'height': 'calc(100vh - 60px)', 'overflow': 'hidden'})
], style=main_container_style)

# Helper function to categorize performance
def categorize_performance(score):
    if score >= 80:
        return 'High (≥80%)'
    elif score >= 50:
        return 'Medium (50-79%)'
    else:
        return 'Low (<50%)'

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
    Output('selected-performance-group', 'children'),
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
    Input('selected-performance-group', 'children'),
    Input('color-scheme-dropdown', 'value')
)
def update_scatter_tsne_plot(plot_type, x_axis, y_axis, search_value, student_count, selected_group, color_scheme):
    filtered_df = df.sample(n=min(student_count, len(df)), random_state=42).copy()
    filtered_df['Performance_Group'] = filtered_df['exam_score'].apply(categorize_performance)
    
    if plot_type == 'scatter':
        fig = px.scatter(
            filtered_df,
            x=x_axis,
            y=y_axis,
            color='exam_score',
            size="attendance_percentage",
            size_max=8,
            hover_data=["student_id"],
            color_continuous_scale=color_scheme,
        )
        
        # Highlight selected group
        if selected_group != 'All' and selected_group in ['High (≥80%)', 'Medium (50-79%)', 'Low (<50%)']:
            # Fade non-selected groups
            for trace in fig.data:
                if trace.name != selected_group:
                    trace.marker.opacity = 0.3
                else:
                    trace.marker.opacity = 1.0
                    trace.marker.line = dict(width=2, color='black')
        
    else:  # t-SNE
        # Hide feature selection drop downs
        

        def encodedData(df):
            # Encode categorical features
            categorical_cols = ['gender', 'part_time_job', 'diet_quality',
                                'parental_education_level', 'internet_quality',
                                'extracurricular_participation', 'Performance_Group']

            df_encoded = df.copy()
            for col in categorical_cols:
                df_encoded[col] = LabelEncoder().fit_transform(df_encoded[col])

            # Drop student_id (not useful for prediction)
            df_encoded = df_encoded.drop(columns=['student_id'])

            # Separate features and target
            X = df_encoded.drop(columns=['exam_score'])

            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            return X_scaled
        
        X = encodedData(filtered_df)
        reducer = TSNE(n_components=2, perplexity=30, random_state=42)
        X_reduced = reducer.fit_transform(X)
        
        fig = px.scatter(
            x=X_reduced[:, 0],
            y=X_reduced[:, 1],
            color=filtered_df['exam_score'],
            hover_data={'Student_ID': filtered_df['student_id']},
            color_continuous_scale=color_scheme,
        )
        
        fig.update_coloraxes(colorbar_title='Exam Score')
        
        # Highlight selected group
        if selected_group != 'All' and selected_group in ['High (≥80%)', 'Medium (50-79%)', 'Low (<50%)']:
            for trace in fig.data:
                if trace.name != selected_group:
                    trace.marker.opacity = 0.3
                else:
                    trace.marker.opacity = 1.0
                    trace.marker.line = dict(width=2, color='black')
    
    # Highlight searched student
    if search_value and search_value.strip():
        matched = filtered_df[filtered_df['student_id'].astype(str).str.contains(search_value.strip(), case=False)]
        if not matched.empty:
            student = matched.iloc[0]
            if plot_type == 'scatter':
                fig.add_trace(go.Scatter(
                    x=[student[x_axis]],
                    y=[student['exam_score']],
                    mode='markers',
                    marker=dict(color='purple', size=15, symbol='star'),
                    name=f"ID: {student['student_id']}",
                    showlegend=False
                ))
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=10),
        margin=dict(l=30, r=30, t=20, b=30),
        legend=dict(font=dict(size=8), orientation="h", y=-0.1),
    )

    return fig

@app.callback(
    Output('performance-donut-plot', 'figure'),
    Input('student-count-slider', 'value'),
    Input('color-scheme-dropdown', 'value')
)
def update_performance_donut_plot(student_count, color_scheme):
    sample_df = df.sample(n=min(student_count, len(df)), random_state=42)
    sample_df['Performance_Group'] = sample_df['exam_score'].apply(categorize_performance)
    
    counts = sample_df['Performance_Group'].value_counts()
    labels = counts.index.tolist()

    # Define color maps for each color scheme (3 colors: [Low, Medium, High])
    color_schemes = {
        'viridis': ['#440154', '#21908d', '#fde725'],
        'plasma':  ['#0d0887', '#cc4778', '#f0f921'],
        'cividis': ['#00224e', '#7c7b78', '#fde636'],
        'inferno': ['#000004', '#b53679', '#fcffa4']
    }

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


@app.callback(
    Output('parallel-plot', 'figure'),
    Input('student-count-slider', 'value'),
    Input('selected-performance-group', 'children'),
    Input('color-scheme-dropdown', 'value')
)
def update_parallel_plot(student_count, selected_group, color_scheme):
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

@app.callback(
    Output('categorical-bar-plot', 'figure'),
    Input('student-count-slider', 'value'),
    Input('selected-performance-group', 'children'),
    Input('color-scheme-dropdown', 'value')
)
def update_categorical_bar_plot(student_count, selected_group, color_scheme):
    sample_df = df.sample(n=min(student_count, len(df)), random_state=42).copy()
    sample_df['Performance_Group'] = sample_df['exam_score'].apply(categorize_performance)
    
    # Filter by selected group
    if selected_group != 'All' and selected_group in ['High (≥80%)', 'Medium (50-79%)', 'Low (<50%)']:
        filtered_df = sample_df[sample_df['Performance_Group'] == selected_group]
        title_suffix = f" - {selected_group}"
    else:
        filtered_df = sample_df
        title_suffix = " - All Students"
    
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
    color_schemes = {
        'viridis': ['#440154', '#414487', '#2a788e', '#22a884', '#7ad151', '#fde725'],
        'plasma':  ['#0d0887', '#6a00a8', '#b12a90', '#e16462', '#fca636', '#f0f921'],
        'cividis': ['#00224e', '#475f6b', '#88897a', '#c7b36c', '#fde636', '#ffeb00'],
        'inferno': ['#000004', '#2d1152', '#721f81', '#b5367a', '#fb8761', '#fcffa4'],
        'default': ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
    }

    colors = color_schemes.get(color_scheme, color_schemes['default'])
    
    for i, feature in enumerate(available_features):
        if feature in filtered_df.columns:
            counts = filtered_df[feature].value_counts()
            fig.add_trace(go.Bar(
                name=feature.replace('_', ' ').title(),
                x=counts.index,
                y=counts.values,
                marker_color=colors[i % len(colors)],
                opacity=0.8,
                visible=True if i == 0 else 'legendonly'  # Show first feature by default
            ))
    
    fig.update_layout(
        title=f"Categorical Features{title_suffix}",
        title_font_size=12,
        xaxis_title="Categories",
        yaxis_title="Count",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=9),
        margin=dict(l=30, r=30, t=40, b=30),
        legend=dict(font=dict(size=8), orientation="h", y=-0.2)
    )
    
    return fig

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

    prediction = min(prediction_model.predict(input_data)[0], 100)
    
    # Use dynamic color based on prediction bucket
    color_schemes = {
        'viridis': ['#440154', '#21908d', '#fde725'],
        'plasma':  ['#0d0887', '#cc4778', '#f0f921'],
        'cividis': ['#00224e', '#7c7b78', '#fde636'],
        'inferno': ['#000004', '#b53679', '#fcffa4'],
        'default': ['#2ecc71', '#f39c12', '#e74c3c']
    }

    scheme = color_schemes.get(color_scheme, color_schemes['default'])

    colors = [scheme[0] if prediction >= 80 else scheme[1] if prediction >= 50 else scheme[2], '#ecf0f1']

    
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

if __name__ == '__main__':
    app.run(debug=True)