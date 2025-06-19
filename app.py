from dash import Dash
from components.layout import create_layout
from components.callbacks import register_callbacks
from utils.data_processing import load_and_prepare_data
from components.chat_callbacks import register_chat_callbacks  # New chat callbacks
from sklearn.preprocessing import StandardScaler

# Initialize the app
app = Dash(__name__, suppress_callback_exceptions=True)

# Load data and model
df, prediction_model = load_and_prepare_data()

selected_features = [
    'study_hours_per_day',
    'mental_health_rating',
    'social_media_hours',
    'sleep_hours',
    'netflix_hours',
    'exercise_frequency',
    'attendance_percentage'
]

X = df[selected_features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Store data globally for callbacks
app.df = df
app.prediction_model = prediction_model
app.scaler = scaler

# Create layout
app.layout = create_layout()

# Register all callbacks
register_callbacks(app)
register_chat_callbacks(app) 

if __name__ == '__main__':
    app.run(debug=False)