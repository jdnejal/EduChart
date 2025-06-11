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
                style={'width': '48%', 'fontSize': '11px', 'display': 'inline-block', 'marginRight': '4%'}
            ),
            dcc.Dropdown(
                id='x-axis-dropdown',
                options=[{'label': f.replace('_', ' ').replace(' (%)', ''), 'value': f} for f in selectable_features],
                value='Study_Hours_per_Week',
                clearable=False,
                style={'width': '48%', 'fontSize': '11px', 'display': 'inline-block'}
            )
        ], style={'marginBottom': '8px'}),
        
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

# Prediction tile (unchanged)
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
                html.Div([create_scatter_tsne_tile()], style={'width': '50%', 'display': 'inline-block'}),
                html.Div([create_performance_donut_tile()], style={'width': '50%', 'display': 'inline-block', 'float': 'right'})
            ]),
            html.Div([
                html.Div([create_parallel_tile()], style={'width': '50%', 'display': 'inline-block'}),
                html.Div([create_categorical_bar_tile()], style={'width': '50%', 'display': 'inline-block', 'float': 'right'})
            ])
        ], style={'width': '75%', 'display': 'inline-block', 'verticalAlign': 'top'}),
        
        # Right section - prediction model and chatbot
        html.Div([
            create_prediction_tile(),
            create_chatbot_tile()
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
    Output('selected-performance-group', 'children'),
    Input('performance-donut-plot', 'clickData')
)
def update_selected_group(clickData):
    if clickData is None:
        return 'All'
    return clickData['points'][0]['label']

@app.callback(
    Output('scatter-tsne-plot', 'figure'),
    Input('plot-type-dropdown', 'value'),
    Input('x-axis-dropdown', 'value'),
    Input('search-input', 'value'),
    Input('student-count-slider', 'value'),
    Input('selected-performance-group', 'children')
)
def update_scatter_tsne_plot(plot_type, x_axis, search_value, student_count, selected_group):
    filtered_df = df.sample(n=min(student_count, len(df)), random_state=42).copy()
    filtered_df['Performance_Group'] = filtered_df['Exam_Score (%)'].apply(categorize_performance)
    
    if plot_type == 'scatter':
        fig = px.scatter(
            filtered_df,
            x=x_axis,
            y='Exam_Score (%)',
            color='Performance_Group',
            size="Attendance_Rate (%)",
            size_max=8,
            hover_data=["Student_ID"],
            color_discrete_map={
                'High (≥80%)': '#2ecc71',
                'Medium (50-79%)': '#f39c12', 
                'Low (<50%)': '#e74c3c'
            }
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
        numeric_features = [
            "Assignment_Completion_Rate (%)",
            "Attendance_Rate (%)",
            "Time_Spent_on_Social_Media (hours/week)",
            "Study_Hours_per_Week",
            "Online_Courses_Completed"
        ]
        
        X = filtered_df[numeric_features].fillna(0)
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X)-1), learning_rate=200, n_iter=1000)
        X_reduced = reducer.fit_transform(X)
        
        fig = px.scatter(
            x=X_reduced[:, 0],
            y=X_reduced[:, 1],
            color=filtered_df['Performance_Group'],
            color_discrete_map={
                'High (≥80%)': '#2ecc71',
                'Medium (50-79%)': '#f39c12', 
                'Low (<50%)': '#e74c3c'
            },
            hover_data={'Student_ID': filtered_df['Student_ID']}
        )
        
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
        matched = filtered_df[filtered_df['Student_ID'].astype(str).str.contains(search_value.strip(), case=False)]
        if not matched.empty:
            student = matched.iloc[0]
            if plot_type == 'scatter':
                fig.add_trace(go.Scatter(
                    x=[student[x_axis]],
                    y=[student['Exam_Score (%)']],
                    mode='markers',
                    marker=dict(color='purple', size=15, symbol='star'),
                    name=f"ID: {student['Student_ID']}",
                    showlegend=False
                ))
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=10),
        margin=dict(l=30, r=30, t=20, b=30),
        legend=dict(font=dict(size=8), orientation="h", y=-0.1)
    )

    return fig

@app.callback(
    Output('performance-donut-plot', 'figure'),
    Input('student-count-slider', 'value')
)
def update_performance_donut_plot(student_count):
    sample_df = df.sample(n=min(student_count, len(df)), random_state=42)
    sample_df['Performance_Group'] = sample_df['Exam_Score (%)'].apply(categorize_performance)
    
    counts = sample_df['Performance_Group'].value_counts()
    
    fig = go.Figure(data=[go.Pie(
        labels=counts.index,
        values=counts.values,
        hole=0.4,
        marker=dict(
            colors=['#f39c12', '#2ecc71', '#e74c3c'],
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
    Input('selected-performance-group', 'children')
)
def update_parallel_plot(student_count, selected_group):
    sample_df = df.sample(n=min(student_count, len(df)), random_state=42).copy()
    sample_df['Performance_Group'] = sample_df['Exam_Score (%)'].apply(categorize_performance)
    
    # Filter by selected group
    if selected_group != 'All' and selected_group in ['High (≥80%)', 'Medium (50-79%)', 'Low (<50%)']:
        filtered_df = sample_df[sample_df['Performance_Group'] == selected_group]
    else:
        filtered_df = sample_df
    
    # Fixed features for parallel coordinates
    fixed_features = [
        "Assignment_Completion_Rate (%)",
        "Attendance_Rate (%)",
        "Study_Hours_per_Week",
        "Exam_Score (%)"
    ]
    
    if len(filtered_df) == 0:
        return go.Figure().add_annotation(
            text="No data for selected group",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=12, color="gray")
        )
    
    fig = px.parallel_coordinates(
        filtered_df,
        dimensions=fixed_features,
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
    Output('categorical-bar-plot', 'figure'),
    Input('student-count-slider', 'value'),
    Input('selected-performance-group', 'children')
)
def update_categorical_bar_plot(student_count, selected_group):
    sample_df = df.sample(n=min(student_count, len(df)), random_state=42).copy()
    sample_df['Performance_Group'] = sample_df['Exam_Score (%)'].apply(categorize_performance)
    
    # Filter by selected group
    if selected_group != 'All' and selected_group in ['High (≥80%)', 'Medium (50-79%)', 'Low (<50%)']:
        filtered_df = sample_df[sample_df['Performance_Group'] == selected_group]
        title_suffix = f" - {selected_group}"
    else:
        filtered_df = sample_df
        title_suffix = " - All Students"
    
    categorical_features = [
        'Preferred_Learning_Style',
        'Participation_in_Discussions', 
        'Use_of_Educational_Tech'
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
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
    
    for i, feature in enumerate(available_features):
        if feature in filtered_df.columns:
            counts = filtered_df[feature].value_counts()
            fig.add_trace(go.Bar(
                name=feature.replace('_', ' '),
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