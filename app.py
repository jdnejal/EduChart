from dash import Dash, html, dcc, callback, Output, Input, State
import plotly.express as px
import pandas as pd
import joblib
import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

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
    'height': '520px'
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
                    max=min(3000, len(df)),
                    step=100,
                    value=800,
                    marks={i: {'label': f'{i}', 'style': {'color': 'white', 'fontSize': '10px'}} 
                           for i in range(100, min(3001, len(df)+1), 500)},
                    tooltip={"placement": "bottom", "always_visible": True}
                )
            ], style={'width': '400px', 'marginRight': '15px'}),
            
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
                    'backgroundColor': 'white'
                }
            )
        ], style={'display': 'flex', 'alignItems': 'center'})
    ], style=top_bar_style)

# Small visualization tiles
def create_scatter_tile():
    return html.Div([
        html.H4("📈 Performance Scatter", style={
            'margin': '0 0 10px 0',
            'color': '#2c3e50',
            'fontSize': '16px',
            'fontWeight': '600'
        }),
        
        html.Div([
            dcc.Dropdown(
                id='x-axis-dropdown',
                options=[{'label': f.replace('_', ' ').replace(' (%)', ''), 'value': f} for f in selectable_features],
                value='Study_Hours_per_Week',
                clearable=False,
                style={'width': '48%', 'fontSize': '11px', 'display': 'inline-block', 'marginRight': '4%'}
            ),
            dcc.Dropdown(
                id='y-axis-dropdown',
                options=[{'label': f.replace('_', ' ').replace(' (%)', ''), 'value': f} for f in selectable_features],
                value='Exam_Score (%)',
                clearable=False,
                style={'width': '48%', 'fontSize': '11px', 'display': 'inline-block'}
            )
        ], style={'marginBottom': '8px'}),
        
        dcc.Graph(id='scatter-plot', style={'height': '240px'})
    ], style=small_tile_style)

def create_dimensionality_tile():
    return html.Div([
        html.H4("🎯 Dimensionality Reduction", style={
            'margin': '0 0 10px 0',
            'color': '#2c3e50',
            'fontSize': '16px',
            'fontWeight': '600'
        }),
        
        html.Div([
            dcc.Dropdown(
                id='dim-reduction-method',
                options=[
                    {'label': 'PCA', 'value': 'pca'},
                    {'label': 't-SNE', 'value': 'tsne'}
                ],
                value='pca',
                clearable=False,
                style={'width': '100%', 'fontSize': '11px'}
            )
        ], style={'marginBottom': '8px'}),
        
        dcc.Graph(id='dim-reduction-plot', style={'height': '240px'})
    ], style=small_tile_style)

def create_parallel_tile():
    return html.Div([
        html.H4("📊 Parallel Coordinates", style={
            'margin': '0 0 10px 0',
            'color': '#2c3e50',
            'fontSize': '16px',
            'fontWeight': '600'
        }),
        
        html.Div([
            dcc.Dropdown(
                id='parallel-features',
                options=[{'label': col.replace('_', ' '), 'value': col} 
                        for col in df.columns if col not in ['student_id', 'Student_ID']],
                value=[
                    "Assignment_Completion_Rate (%)",
                    "Attendance_Rate (%)",
                    "Study_Hours_per_Week",
                    "Exam_Score (%)"
                ],
                multi=True,
                style={'fontSize': '11px'}
            )
        ], style={'marginBottom': '8px'}),
        
        dcc.Graph(id='parallel-plot', style={'height': '240px'})
    ], style=small_tile_style)

def create_radar_tile():
    return html.Div([
        html.H4("🎯 Student Radar", style={
            'margin': '0 0 10px 0',
            'color': '#2c3e50',
            'fontSize': '16px',
            'fontWeight': '600'
        }),
        
        html.Div([
            dcc.Dropdown(
                id='radar-features',
                options=[{'label': col.replace('_', ' '), 'value': col} 
                        for col in df.columns if col not in ['student_id', 'Student_ID', 'Exam_Score (%)']],
                value=[
                    "Assignment_Completion_Rate (%)",
                    "Attendance_Rate (%)",
                    "Study_Hours_per_Week",
                    "Online_Courses_Completed"
                ],
                multi=True,
                style={'fontSize': '11px'}
            )
        ], style={'marginBottom': '8px'}),
        
        dcc.Graph(id='radar-plot', style={'height': '240px'})
    ], style=small_tile_style)

# Prediction tile
def create_prediction_tile():
    return html.Div([
        html.H3("🎯 Score Prediction", style={
            'textAlign': 'center',
            'margin': '0 0 15px 0',
            'color': '#2c3e50',
            'fontSize': '18px',
            'fontWeight': '600'
        }),
        
        dcc.Graph(id='donut-chart', style={'height': '150px'}),
        
        html.Div([
            html.Div([
                html.Label('Assignment (%)', style={'fontWeight': '500', 'fontSize': '12px'}),
                dcc.Slider(id='assignment_completion', min=0, max=100, step=1, value=85,
                          marks={0: '0', 50: '50', 100: '100'}, tooltip={"always_visible": True})
            ], style={'marginBottom': '12px'}),
            
            html.Div([
                html.Label('Attendance (%)', style={'fontWeight': '500', 'fontSize': '12px'}),
                dcc.Slider(id='attendance_rate', min=0, max=100, step=1, value=90,
                          marks={0: '0', 50: '50', 100: '100'}, tooltip={"always_visible": True})
            ], style={'marginBottom': '12px'}),
            
            html.Div([
                html.Label('Social Media (h/w)', style={'fontWeight': '500', 'fontSize': '12px'}),
                dcc.Slider(id='social_media_hours', min=0, max=40, step=1, value=10,
                          marks={0: '0', 20: '20', 40: '40'}, tooltip={"always_visible": True})
            ], style={'marginBottom': '12px'}),
            
            html.Div([
                html.Label('Study Hours/Week', style={'fontWeight': '500', 'fontSize': '12px'}),
                dcc.Slider(id='study_hours_per_week', min=0, max=40, step=1, value=15,
                          marks={0: '0', 20: '20', 40: '40'}, tooltip={"always_visible": True})
            ], style={'marginBottom': '12px'}),
            
            html.Div([
                html.Label('Online Courses', style={'fontWeight': '500', 'fontSize': '12px'}),
                dcc.Slider(id='online_courses', min=0, max=20, step=1, value=2,
                          marks={0: '0', 10: '10', 20: '20'}, tooltip={"always_visible": True})
            ])
        ])
    ], style=prediction_tile_style)

# Chatbot tile
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

# Main layout
app.layout = html.Div([
    create_top_bar(),
    
    html.Div([
        # Left section - 4 small visualization tiles in 2x2 grid
        html.Div([
            html.Div([
                html.Div([create_scatter_tile()], style={'width': '50%', 'display': 'inline-block'}),
                html.Div([create_dimensionality_tile()], style={'width': '50%', 'display': 'inline-block', 'float': 'right'})
            ]),
            html.Div([
                html.Div([create_parallel_tile()], style={'width': '50%', 'display': 'inline-block'}),
                html.Div([create_radar_tile()], style={'width': '50%', 'display': 'inline-block', 'float': 'right'})
            ])
        ], style={'width': '75%', 'display': 'inline-block', 'verticalAlign': 'top'}),
        
        # Right section - prediction model and chatbot
        html.Div([
            create_prediction_tile(),
            create_chatbot_tile()
        ], style={'width': '25%', 'display': 'inline-block', 'verticalAlign': 'top'})
    ], style={'padding': '5px 5px', 'height': 'calc(100vh - 60px)', 'overflow': 'hidden'})
], style=main_container_style)

# Callbacks
@app.callback(
    Output('scatter-plot', 'figure'),
    Input('search-input', 'value'),
    Input('x-axis-dropdown', 'value'),
    Input('y-axis-dropdown', 'value'),
    Input('student-count-slider', 'value')
)
def update_scatter_plot(search_value, x_axis, y_axis, student_count):
    filtered_df = df.sample(n=min(student_count, len(df)), random_state=42).copy()

    fig = px.scatter(
        filtered_df,
        x=x_axis,
        y=y_axis,
        size="Attendance_Rate (%)",
        color="Exam_Score (%)",
        size_max=8,
        hover_data=["Student_ID"],
        color_continuous_scale="viridis"
    )
    
    if search_value and search_value.strip():
        matched = filtered_df[filtered_df['Student_ID'].astype(str).str.contains(search_value.strip(), case=False)]
        if not matched.empty:
            student = matched.iloc[0]
            fig.add_trace(go.Scatter(
                x=[student[x_axis]],
                y=[student[y_axis]],
                mode='markers',
                marker=dict(color='red', size=12, symbol='star'),
                name=f"ID: {student['Student_ID']}",
                showlegend=False
            ))
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=10),
        margin=dict(l=30, r=30, t=20, b=30),
        showlegend=False
    )

    return fig

@app.callback(
    Output('dim-reduction-plot', 'figure'),
    Input('dim-reduction-method', 'value'),
    Input('student-count-slider', 'value')
)
def update_dim_reduction_plot(method, student_count):
    sample_df = df.sample(n=min(student_count, len(df)), random_state=42)
    
    # Select numeric features for dimensionality reduction
    numeric_features = [
        "Assignment_Completion_Rate (%)",
        "Attendance_Rate (%)",
        "Time_Spent_on_Social_Media (hours/week)",
        "Study_Hours_per_Week",
        "Online_Courses_Completed"
    ]
    
    X = sample_df[numeric_features].fillna(0)
    
    if method == 'pca':
        reducer = PCA(n_components=2, random_state=42)
        X_reduced = reducer.fit_transform(X)
        title = f"PCA (Explained Variance: {sum(reducer.explained_variance_ratio_):.2%})"
    else:  # tsne
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X)-1))
        X_reduced = reducer.fit_transform(X)
        title = "t-SNE Visualization"
    
    fig = px.scatter(
        x=X_reduced[:, 0],
        y=X_reduced[:, 1],
        color=sample_df['Exam_Score (%)'],
        color_continuous_scale="viridis",
        title=title
    )
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=10),
        margin=dict(l=30, r=30, t=30, b=30),
        showlegend=False,
        title_font_size=12
    )
    
    return fig

@app.callback(
    Output('parallel-plot', 'figure'),
    Input('parallel-features', 'value'),
    Input('student-count-slider', 'value')
)
def update_parallel_plot(selected_features, student_count):
    if not selected_features or len(selected_features) < 3:
        return go.Figure().add_annotation(
            text="Select at least 3 features",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=12, color="gray")
        )
    
    sample_df = df.sample(n=min(student_count, len(df)), random_state=42)
    
    fig = px.parallel_coordinates(
        sample_df,
        dimensions=selected_features[:5],
        color='Exam_Score (%)',
        color_continuous_scale='viridis'
    )
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=9),
        margin=dict(l=20, r=20, t=20, b=20)
    )
    
    return fig

@app.callback(
    Output('radar-plot', 'figure'),
    Input('radar-features', 'value'),
    Input('search-input', 'value'),
    Input('student-count-slider', 'value')
)
def update_radar_plot(selected_features, search_student, student_count):
    if not selected_features or len(selected_features) < 3:
        return go.Figure().add_annotation(
            text="Select at least 3 features",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=12, color="gray")
        )
    
    sample_df = df.sample(n=min(student_count, len(df)), random_state=42)
    
    # Get student data
    if search_student and search_student.strip():
        matched = sample_df[sample_df['Student_ID'].astype(str).str.contains(search_student.strip(), case=False)]
        student_row = matched.iloc[0] if not matched.empty else sample_df.iloc[0]
    else:
        student_row = sample_df.iloc[0]
    
    # Get top performers average
    top_performers = sample_df[sample_df['Exam_Score (%)'] >= sample_df['Exam_Score (%)'].quantile(0.8)]
    top_avg = top_performers[selected_features].mean()
    
    # Normalize data
    features = selected_features[:5]
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
        name='Student',
        line_color='#3498db'
    ))
    fig.add_trace(go.Scatterpolar(
        r=top_avg_norm + [top_avg_norm[0]],
        theta=theta_labels + [theta_labels[0]],
        fill='toself',
        name='Top Avg',
        line_color='#e74c3c',
        opacity=0.6
    ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=9),
        margin=dict(l=20, r=20, t=20, b=20),
        legend=dict(font=dict(size=8))
    )
    
    return fig

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