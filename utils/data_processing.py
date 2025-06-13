import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_and_prepare_data():
    """Load and prepare the dataset and model"""
    df = pd.read_csv("./data/student_habits_performance.csv")
    prediction_model = joblib.load('models/prediction_model_small_data.pkl')
    return df, prediction_model

def categorize_performance(score):
    """Categorize performance based on exam score"""
    if score >= 80:
        return 'High (≥80%)'
    elif score >= 50:
        return 'Medium (50-79%)'
    else:
        return 'Low (<50%)'

def get_selectable_features():
    """Return list of selectable features"""
    return [
        "study_hours_per_day", 
        "attendance_percentage",   
        "social_media_hours", 
        "netflix_hours", 
        "sleep_hours",
        "exercise_frequency",
        "mental_health_rating",
        "exam_score"
    ]

def get_categorical_features():
    """Return list of categorical features"""
    return [
        'gender',
        'part_time_job', 
        'diet_quality',
        'parental_education_level',
        'internet_quality',
        'extracurricular_participation'
    ]

def encode_data_for_tsne(df):
    """Encode data for t-SNE visualization"""
    categorical_cols = ['gender', 'part_time_job', 'diet_quality',
                       'parental_education_level', 'internet_quality',
                       'extracurricular_participation', 'Performance_Group']

    df_encoded = df.copy()
    for col in categorical_cols:
        if col in df_encoded.columns:
            df_encoded[col] = LabelEncoder().fit_transform(df_encoded[col])

    # Drop student_id (not useful for prediction)
    if 'student_id' in df_encoded.columns:
        df_encoded = df_encoded.drop(columns=['student_id'])

    # Separate features and target
    X = df_encoded.drop(columns=['exam_score'])

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled

def filter_data_by_selection(df, selected_students=None, performance_group=None, sample_size=None):
    """Filter dataframe based on various selection criteria"""
    filtered_df = df.copy()
    
    # Add performance group
    filtered_df['Performance_Group'] = filtered_df['exam_score'].apply(categorize_performance)
    
    # Sample data if needed
    if sample_size and sample_size < len(filtered_df):
        filtered_df = filtered_df.sample(n=sample_size, random_state=42)
    
    # Filter by performance group
    if performance_group and performance_group != 'All':
        filtered_df = filtered_df[filtered_df['Performance_Group'] == performance_group]
    
    # Filter by selected students
    if selected_students and len(selected_students) > 0:
        filtered_df = filtered_df[filtered_df['student_id'].isin(selected_students)]
    
    return filtered_df

def get_color_schemes():
    """Return available color schemes"""
    return {
        'viridis': ['#440154', '#414487', '#2a788e', '#22a884', '#7ad151', '#fde725'],
        'plasma':  ['#0d0887', '#6a00a8', '#b12a90', '#e16462', '#fca636', '#f0f921'],
        'cividis': ['#00224e', '#475f6b', '#88897a', '#c7b36c', '#fde636', '#ffeb00'],
        'inferno': ['#000004', '#2d1152', '#721f81', '#b5367a', '#fb8761', '#fcffa4'],
        'default': ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
    }