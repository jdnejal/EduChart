from dash import Dash, html, dcc, callback, Output, Input
import plotly.express as px
import pandas as pd

df = pd.read_csv("./data/student-habits/student_habits_performance.csv")

app = Dash()

app.layout = [
    html.H1(children='EduClimb', style={'textAlign':'center', 'font-family':'arial', 'color':'blue'}),
    dcc.Graph(figure=px.scatter(df, x='study_hours_per_day', y='exam_score', size="attendance_percentage",size_max=10, color="gender", hover_data="student_id"))
]

if __name__ == '__main__':
    app.run(debug=True)
