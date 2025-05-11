from dash import Dash, html, dcc, callback, Output, Input
import plotly.express as px
import pandas as pd
import joblib
import numpy as np

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
            dcc.Graph(id='scatter-plot')
        ], style={'width': '70%', 'display': 'inline-block'}),

        # Right: Controls and prediction output
        html.Div([
            html.H3('Exam Score Prediction', style={'textAlign': 'center'}),

            # Donut chart for predicted score
            dcc.Graph(id='donut-chart'),

            # Sliders for input values
            html.Div([
                html.Label('Study Hours per Day'),
                dcc.Slider(id='study_hours_per_day', min=0, max=24, step=0.5, value=5,
                           marks={i: str(i) for i in range(0, 25, 4)}),

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
                dcc.Slider(id='sleep_hours', min=0, max=24, step=0.5, value=7,
                           marks={i: str(i) for i in range(0, 25, 4)}),

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

if __name__ == '__main__':
    app.run(debug=True)
