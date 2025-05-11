from dash import Dash, html, dcc, callback, Output, Input
import plotly.express as px
import pandas as pd
import joblib
import numpy as np
import plotly.graph_objects as go

df = pd.read_csv("./data/student-habits/student_habits_performance.csv")

# Load the pre-trained prediction model
prediction_model = joblib.load('models/prediction_model.pkl')

app = Dash()

# Top bar
top_bar = html.Div([
    html.Div("Edu Chart", style={
        'fontSize': '28px',
        'fontWeight': 'bold',
        'color': 'white',
        'marginLeft': '20px',
        'font-family': 'arial'
    }),
    dcc.Input(
        id='search-input',
        placeholder='Search by student ID...',
        type='text',
        debounce=True,  # triggers callback only when typing stops
        style={
            'marginRight': '20px',
            'padding': '5px 10px',
            'fontSize': '16px',
            'borderRadius': '5px',
            'border': '1px solid #ccc'
        }
    )
], style={
    'display': 'flex',
    'justifyContent': 'space-between',
    'alignItems': 'center',
    'backgroundColor': '#007BFF',
    'padding': '10px 0'
})

app.layout = html.Div([
    top_bar,
    html.Div([
        # Left: Scatter plot
        html.Div([
            dcc.Graph(id='scatter-plot'),
            html.Div([
                html.Label('Select Chart:'),
                dcc.Tabs(
                    id='chart-tabs',
                    value='parallel-coords',
                    children=[
                        dcc.Tab(label='Parallel', value='parallel-coords'),
                        dcc.Tab(label='Radar Chart', value='radar'),
                        dcc.Tab(label='Box Plot', value='box-plot'),                       
                    ],
                    colors={
                        "border": "white",
                        "primary": "#888",
                        "background": "#f9f9f9"
                    },
                    style={'fontWeight': 'bold'}
                ),
            ], style={'margin': '20px 0 10px 0'}),

            # html.Div(id='boxplot-controls'),

            # Parallel coordinates plot (shown by default)
            dcc.Graph(id='parallel-coords-plot'),
        ], style={'width': '70%', 'display': 'inline-block', 'verticalAlign': 'top'}),

        # Right: Controls and prediction output
        html.Div([
            html.H3('Exam Score Prediction', style={'textAlign': 'center'}),

            # Donut chart for predicted score
            dcc.Graph(id='donut-chart'),

            # Sliders for input values
            html.Div([
                html.Label('Study Hours per Day'),
                dcc.Slider(id='study_hours_per_day', min=0, max=12, step=0.5, value=5,
                           marks={i: str(i) for i in range(0, 13, 1)}),

                html.Label('Mental Health Rating'),
                dcc.Slider(id='mental_health_rating', min=1, max=10, step=0.1, value=7,
                           marks={i: str(i) for i in range(1, 11)}),

                html.Label('Social Media Hours'),
                dcc.Slider(id='social_media_hours', min=0, max=10, step=0.5, value=2,
                           marks={i: str(i) for i in range(0, 11, 2)}),

                html.Label('Netflix Hours'),
                dcc.Slider(id='netflix_hours', min=0, max=10, step=0.5, value=1,
                           marks={i: str(i) for i in range(0, 11, 2)}),

                html.Label('Sleep Hours'),
                dcc.Slider(id='sleep_hours', min=0, max=12, step=0.5, value=7,
                           marks={i: str(i) for i in range(0, 13, 1)}),

                html.Label('Exercise Frequency (per week)'),
                dcc.Slider(id='exercise_frequency', min=0, max=7, step=1, value=3,
                           marks={i: str(i) for i in range(0, 8)}),

                html.Label('Attendance Percentage'),
                dcc.Slider(id='attendance_percentage', min=0, max=100, step=1, value=90,
                           marks={i: str(i) for i in range(0, 101, 20)}),
            ], style={'padding': '10px 20px'}),

        ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top', 'font-family': 'arial'})
    ])
])

@app.callback(
    Output('scatter-plot', 'figure'),
    Input('search-input', 'value')
)
def update_scatter_plot(search_value):
    fig = px.scatter(
        df,
        x='study_hours_per_day',
        y='exam_score',
        size="attendance_percentage",
        color="gender",
        size_max=10,
        hover_data=["student_id"]
    )

    if search_value and search_value.strip():
        matched = df[df['student_id'].astype(str).str.contains(search_value.strip(), case=False)]
        if not matched.empty:
            student = matched.iloc[0]
            # Zoom around the student's point
            fig.update_layout(
                xaxis_range=[student['study_hours_per_day'] - 1, student['study_hours_per_day'] + 1],
                yaxis_range=[student['exam_score'] - 5, student['exam_score'] + 5]
            )
            # Add marker
            fig.add_trace({
                'type': 'scatter',
                'x': [student['study_hours_per_day']],
                'y': [student['exam_score']],
                'mode': 'markers+text',
                'marker': {'color': 'red', 'size': 15},
                'text': [f"ID: {student['student_id']}"],
                'textposition': 'top center',
                'showlegend': False
            })

    return fig


@app.callback(
    Output('donut-chart', 'figure'),
    Input('study_hours_per_day', 'value'),
    Input('mental_health_rating', 'value'),
    Input('social_media_hours', 'value'),
    Input('netflix_hours', 'value'),
    Input('sleep_hours', 'value'),
    Input('exercise_frequency', 'value'),
    Input('attendance_percentage', 'value')
)
def update_donut_chart(study_hours_per_day, mental_health_rating, social_media_hours,
                       netflix_hours, sleep_hours, exercise_frequency, attendance_percentage):

    # Prepare input for prediction
    input_data = pd.DataFrame({
        'study_hours_per_day': [study_hours_per_day],
        'mental_health_rating': [mental_health_rating],
        'social_media_hours': [social_media_hours],
        'netflix_hours': [netflix_hours],
        'sleep_hours': [sleep_hours],
        'exercise_frequency': [exercise_frequency],
        'attendance_percentage': [attendance_percentage]
    })

    # Make prediction
    prediction = min(prediction_model.predict(input_data)[0], 100)

    # Donut chart figure
    donut_fig = {
        'data': [{
            'values': [prediction, 100 - prediction],
            'labels': ['Predicted Score', 'Remaining'],
            'type': 'pie',
            'hole': 0.6,
            'direction': 'clockwise',
            'marker': {'colors': ['#007BFF', '#f0f0f0']}
        }],
        'layout': {
            'showlegend': False,
            'height': 250,
            'margin': {'t': 30, 'b': 30, 'l': 0, 'r': 0}
        }
    }

    return donut_fig

@app.callback(
    Output('parallel-coords-plot', 'figure'),
    Input('chart-tabs', 'value'),
    Input('search-input', 'value'),
    # Input('boxplot-metric', 'value'),
    # Input('performance-threshold', 'value')
)
def update_parallel_coords(selected_chart,selected_student):
    features = [
        'study_hours_per_day', 'social_media_hours',
        'sleep_hours', 'attendance_percentage','mental_health_rating'
    ]

    if selected_chart == 'parallel-coords':
        fig = px.parallel_coordinates(
            df,
            dimensions=['study_hours_per_day', 'mental_health_rating', 'social_media_hours', 
                        'netflix_hours', 'sleep_hours', 'exercise_frequency', 'attendance_percentage'],
            color='exam_score',
            color_continuous_scale=px.colors.diverging.Tealrose,
            labels={col: col.replace('_', ' ').title() for col in df.columns}
        )
        return fig
    
    elif selected_chart == 'radar':  # Radar chart

        if selected_student and selected_student in df['student_id'].values:
            student_row = df[df['student_id'] == selected_student].iloc[0]
        else:
            student_row = df.iloc[0]  # Default to first student if not found

        # Calculate averages for top performers (e.g., exam_score >= 90)
        top_performers = df[df['exam_score'] >= 90]
        top_avg = top_performers[features].mean()

        # Get student data
        student_data = [student_row[feat] for feat in features]
        top_avg_data = [top_avg[feat] for feat in features]

        # Define min and max for each feature
        feature_min = [0, 0, 0, 0]      # Minimum possible values
        feature_max = [12, 12, 10, 100] # Maximum possible values

        def normalize(values, min_vals, max_vals):
            return [(v - min_v) / (max_v - min_v) if max_v > min_v else 0
                    for v, min_v, max_v in zip(values, min_vals, max_vals)]

        student_norm = normalize(student_data, feature_min, feature_max)
        top_avg_norm = normalize(top_avg_data, feature_min, feature_max)

        # Close the loop for radar chart
        radar_labels = [label.replace('_', ' ').title() for label in features]
        radar_labels += [radar_labels[0]]
        student_data += [student_data[0]]
        top_avg_data += [top_avg_data[0]]

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=student_norm,
            theta=radar_labels,
            fill='toself',
            name='Student'
        ))
        fig.add_trace(go.Scatterpolar(
            r=top_avg_norm,
            theta=radar_labels,
            fill='toself',
            name='Top Performers Avg'
        ))
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            showlegend=True,
            title="Student vs. Top Performers: Normalized Radar Chart"
        )

        return fig

    # Box Plot
    elif selected_chart == 'box-plot':

        # Default values if none selected
        # metric = boxplot_metric if boxplot_metric else 'sleep_hours'
        # threshold = performance_threshold if performance_threshold is not None else 90

        # # Define high and low performer groups dynamically
        # df['performance_group'] = df['exam_score'].apply(lambda x: 'High' if x >= threshold else 'Low')

        # Example: Compare sleep hours for high vs. low performers
        # Define high and low performer groups
        df['performance_group'] = df['exam_score'].apply(lambda x: 'High' if x >= 90 else 'Low')
        
        fig = px.box(
            df,
            x='performance_group',
            y='sleep_hours',
            color='performance_group',
            points='all',  # Show all points
            labels={'sleep_hours': 'Sleep Hours', 'performance_group': 'Performance Group'},
            title='Sleep Hours: High vs. Low Performers'
        )

        return fig

if __name__ == '__main__':
    app.run(debug=True)
