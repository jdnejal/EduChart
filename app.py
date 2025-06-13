from dash import Dash
from components.layout import create_layout
from components.callbacks import register_callbacks
from utils.data_processing import load_and_prepare_data

# Initialize the app
app = Dash(__name__, suppress_callback_exceptions=True)

# Load data and model
df, prediction_model = load_and_prepare_data()

# Store data globally for callbacks
app.df = df
app.prediction_model = prediction_model

# Create layout
app.layout = create_layout()

# Register all callbacks
register_callbacks(app)

if __name__ == '__main__':
    app.run(debug=True)