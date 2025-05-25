from dash import Dash, html, dcc, callback, Output, Input
import plotly.express as px
import pandas as pd
import joblib
import numpy as np
import plotly.graph_objects as go

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

# Top bar
top_bar = html.Div([
    html.Div("Edu Chart", style={
        'fontSize': '34px',
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
    'padding': '10px 0',
})

app.layout = html.Div([
    top_bar,
    html.Div([
        # Left: Scatter plot
        html.Div([
            # Controls above the scatter plot
            html.Div([
                html.Span(id='visible-count', style={'fontWeight': 'bold', 'marginRight': '20px'}),

                html.Label("X-axis:"),
                dcc.Dropdown(
                    id='x-axis-dropdown',
                    options=[{'label': f, 'value': f} for f in selectable_features],
                    value='Study_Hours_per_Week',
                    clearable=False,
                    style={'width': '200px', 'display': 'inline-block', 'marginRight': '20px'}
                ),
                html.Label("Y-axis:"),
                dcc.Dropdown(
                    id='y-axis-dropdown',
                    options=[{'label': f, 'value': f} for f in selectable_features],
                    value='Exam_Score (%)',
                    clearable=False,
                    style={'width': '200px', 'display': 'inline-block'}
                )
            ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '10px'}),
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

            dcc.Dropdown(
                id='feature-selector',
                options=[{'label': col, 'value': col} for col in df.columns if col != 'student_id' and col != 'exam_score'],
                value=[
                    "Assignment_Completion_Rate (%)",
                    "Attendance_Rate (%)",
                    "Time_Spent_on_Social_Media (hours/week)",
                    "Study_Hours_per_Week",
                    "Online_Courses_Completed"
                ],
                multi=True,
                placeholder="Select up to 5 features",
                maxHeight=200,
                style={'margin-bottom': '10px'}
            ),


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
                html.Label('Assignment Completion Rate (%)'),
                dcc.Slider(id='assignment_completion', min=0, max=100, step=1, value=85,
                        marks={i: str(i) for i in range(0, 101, 10)}),

                html.Label('Attendance Rate (%)'),
                dcc.Slider(id='attendance_rate', min=0, max=100, step=1, value=90,
                        marks={i: str(i) for i in range(0, 101, 10)}),

                html.Label('Time Spent on Social Media (hours/week)'),
                dcc.Slider(id='social_media_hours', min=0, max=40, step=1, value=10,
                        marks={i: str(i) for i in range(0, 41, 5)}),

                html.Label('Study Hours per Week'),
                dcc.Slider(id='study_hours_per_week', min=0, max=40, step=1, value=15,
                        marks={i: str(i) for i in range(0, 41, 5)}),

                html.Label('Online Courses Completed'),
                dcc.Slider(id='online_courses', min=0, max=20, step=1, value=2,
                        marks={i: str(i) for i in range(0, 21, 2)}),
            ], style={'padding': '10px 20px'}),


        ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top', 'font-family': 'arial'})
    ])
])

@app.callback(
    Output('scatter-plot', 'figure'),
    Output('visible-count', 'children'),
    Input('search-input', 'value'),
    Input('x-axis-dropdown', 'value'),
    Input('y-axis-dropdown', 'value')
)
def update_scatter_plot(search_value, x_axis, y_axis):
    filtered_df = df.copy()

    if search_value and search_value.strip():
        matched = filtered_df[filtered_df['Student_ID'].astype(str).str.contains(search_value.strip(), case=False)]
        if not matched.empty:
            student = matched.iloc[0]
            zoom_x = [student[x_axis] - 2, student[x_axis] + 2]
            zoom_y = [student[y_axis] - 10, student[y_axis] + 10]
        else:
            zoom_x = zoom_y = None
    else:
        zoom_x = zoom_y = None

    fig = px.scatter(
        filtered_df,
        x=x_axis,
        y=y_axis,
        size="Attendance_Rate (%)",
        color="gender" if "gender" in df.columns else None,
        size_max=10,
        hover_data=["Student_ID"]
    )

    if zoom_x and zoom_y:
        fig.update_layout(
            xaxis_range=zoom_x,
            yaxis_range=zoom_y
        )
        fig.add_trace(go.Scatter(
            x=[student[x_axis]],
            y=[student[y_axis]],
            mode='markers+text',
            marker=dict(color='red', size=15),
            text=[f"ID: {student['Student_ID']}"],
            textposition='top center',
            showlegend=False
        ))

    return fig, f"Students shown: {len(filtered_df)}"



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
    Input('feature-selector', 'value')
)
def update_parallel_coords(selected_chart, selected_student, selected_features):
    if not selected_features or len(selected_features) < 3:
        return go.Figure().update_layout(title="Please select at least 3 features.")

    features = selected_features[:5]  # Limit to 5 for visual clarity

    if selected_chart == 'parallel-coords':
        fig = px.parallel_coordinates(
            df,
            dimensions=features,
            color='Exam_Score (%)',
            color_continuous_scale=px.colors.diverging.Tealrose,
            labels={col: col.replace('_', ' ').title() for col in df.columns}
        )
        return fig

    elif selected_chart == 'radar':
        if selected_student and selected_student in df['Student_ID'].values:
            student_row = df[df['Student_ID'] == selected_student].iloc[0]
        else:
            student_row = df.iloc[0]  # Default to first student

        top_performers = df[df['Exam_Score (%)'] >= 90]
        top_avg = top_performers[features].mean()

        student_data = [student_row[feat] for feat in features]
        top_avg_data = [top_avg[feat] for feat in features]

        feature_min = [df[feat].min() * 0.9 for feat in features]
        feature_max = [df[feat].max() * 1.1 for feat in features]

        def normalize(values, min_vals, max_vals):
            return [(v - min_v) / (max_v - min_v) if max_v > min_v else 0
                    for v, min_v, max_v in zip(values, min_vals, max_vals)]

        student_norm = normalize(student_data, feature_min, feature_max)
        top_avg_norm = normalize(top_avg_data, feature_min, feature_max)

        radar_labels = [label.replace('_', ' ').replace(' (%)', '').title() for label in features]
        student_norm += [student_norm[0]]
        top_avg_norm += [top_avg_norm[0]]
        radar_labels += [radar_labels[0]]

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
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            title="Student vs. Top Performers: Normalized Radar Chart"
        )
        return fig

    elif selected_chart == 'box':
        df_long = df.melt(id_vars=["Student_ID"], value_vars=features, var_name="Feature", value_name="Value")
        fig = px.box(df_long, x="Feature", y="Value", points="all", title="Distribution of Selected Features")
        fig.update_layout(xaxis_title="Feature", yaxis_title="Value")
        return fig


if __name__ == '__main__':
    app.run(debug=True)
