from dash import html, dcc
from utils.data_processing import get_selectable_features

# Custom CSS styles
small_tile_style = {
    'backgroundColor': 'white',
    'borderRadius': '12px',
    'boxShadow': '0 3px 10px rgba(0, 0, 0, 0.1)',
    'padding': '15px',
    'margin': '8px',
    'border': '1px solid #e0e0e0',
    'height': '335px',
    'transition': 'all 0.3s ease'
}

prediction_tile_style = {
    'backgroundColor': 'white',
    'borderRadius': '12px',
    'boxShadow': '0 3px 10px rgba(0, 0, 0, 0.1)',
    'padding': '20px',
    'margin': '8px',
    'border': '1px solid #e0e0e0',
    'height': '700px',
    'transition': 'all 0.3s ease'
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

def create_top_bar():
    """Create the top navigation bar"""
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
                    max=1000,  # Will be updated dynamically
                    step=50,
                    value=800,
                    marks={i: {'label': f'{i}', 'style': {'color': 'white', 'fontSize': '10px'}} 
                           for i in range(100, 1001, 200)},
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
                        html.Button("Reset Selection", id='reset-button', style={
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

def create_scatter_tsne_tile():
    """Create scatter/t-SNE plot tile"""
    selectable_features = get_selectable_features()
    
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
        
        dcc.Graph(
            id='scatter-tsne-plot', 
            style={'height': '240px'},
            config={'displayModeBar': False}
        )
    ], style=small_tile_style)

def create_performance_donut_tile():
    """Create performance distribution donut chart tile"""
    return html.Div([
        html.H4("🎯 Performance Distribution", style={
            'margin': '0 0 10px 0',
            'color': '#2c3e50',
            'fontSize': '16px',
            'fontWeight': '600'
        }),
        
        dcc.Graph(
            id='performance-donut-plot', 
            style={'height': '275px'},
            config={'displayModeBar': False}
        )
    ], style=small_tile_style)

def create_parallel_tile():
    """Create parallel coordinates plot tile"""
    return html.Div([
        html.H4("📊 Student Profiles", style={
            'margin': '0 0 10px 0',
            'color': '#2c3e50',
            'fontSize': '16px',
            'fontWeight': '600'
        }),
        
        dcc.Graph(
            id='parallel-plot', 
            style={'height': '275px'},
            config={'displayModeBar': False}
        )
    ], style=small_tile_style)

def create_categorical_bar_tile():
    """Create categorical analysis bar chart tile"""
    return html.Div([
        html.H4("📋 Categorical Analysis", style={
            'margin': '0 0 10px 0',
            'color': '#2c3e50',
            'fontSize': '16px',
            'fontWeight': '600'
        }),
        
        dcc.Graph(
            id='categorical-bar-plot', 
            style={'height': '275px'},
            config={'displayModeBar': False}
        )
    ], style=small_tile_style)

def create_prediction_tile():
    """Create prediction model tile with sliders"""
    return html.Div([
        html.H3("🎯 Score Prediction", style={
            'textAlign': 'center',
            'margin': '0 0 15px 0',
            'color': '#2c3e50',
            'fontSize': '18px',
            'fontWeight': '600'
        }),
        
        dcc.Graph(
            id='donut-chart', 
            style={'height': '160px'},
            config={'displayModeBar': False}
        ),
        
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

def create_layout():
    """Create the main dashboard layout"""
    return html.Div([
        create_top_bar(),
        
        # Hidden storage components for state management
        dcc.Store(id='selected-students-store', data=[]),
        dcc.Store(id='selected-performance-group-store', data='All'),
        dcc.Store(id='current-sample-size-store', data=800),
        dcc.Store(id='bar-chart-visibility-store', data=None),
        
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
            
            # Right section - prediction model
            html.Div([
                create_prediction_tile(),
            ], style={'width': '25%', 'display': 'inline-block', 'verticalAlign': 'top'})
        ], style={'padding': '5px 5px', 'height': 'calc(100vh - 60px)', 'overflow': 'hidden'})
    ], style=main_container_style)