from dash import Input, Output, State, html, callback_context
import json
from datetime import datetime
import pandas as pd
from utils.data_processing import categorize_performance


def register_chat_callbacks(app):
    """Register all chat-related callbacks for the dashboard"""
    
    # Access the globally stored data
    df = app.df
    prediction_model = app.prediction_model

    @app.callback(
        Output('chat-container', 'style'),
        Output('chat-visible-store', 'data'),
        Input('chat-toggle-button', 'n_clicks'),
        Input('chat-close-button', 'n_clicks'),
        State('chat-visible-store', 'data'),
        prevent_initial_call=True
    )
    def toggle_chat_visibility(toggle_clicks, close_clicks, is_visible):
        """Toggle chat container visibility"""
        ctx = callback_context
        if not ctx.triggered:
            return {'display': 'none'}, False
        
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        if button_id == 'chat-toggle-button':
            # Toggle visibility
            new_visibility = not is_visible
        elif button_id == 'chat-close-button':
            # Always close
            new_visibility = False
        else:
            new_visibility = is_visible
        
        # Define the base chat container style
        base_style = {
            'position': 'fixed',
            'bottom': '80px',
            'right': '20px',
            'width': '350px',
            'height': '500px',
            'backgroundColor': '#2c3e50',
            'border': '1px solid #34495e',
            'borderRadius': '10px',
            'boxShadow': '0 4px 20px rgba(0,0,0,0.3)',
            'zIndex': '1002',
            'display': 'flex',
            'flexDirection': 'column'
        }
        
        if new_visibility:
            base_style['display'] = 'flex'
        else:
            base_style['display'] = 'none'
        
        return base_style, new_visibility

    @app.callback(
        Output('chat-messages', 'children'),
        Output('chat-input', 'value'),
        Output('chat-history-store', 'data'),
        Input('chat-send-button', 'n_clicks'),
        Input('chat-input', 'n_submit'),
        State('chat-input', 'value'),
        State('chat-history-store', 'data'),
        State('selected-students-store', 'data'),
        State('selected-performance-group-store', 'data'),
        State('student-count-slider', 'value'),
        prevent_initial_call=True
    )
    def handle_chat_interaction(send_clicks, input_submit, user_input, chat_history, 
                               selected_students, selected_group, student_count):
        """Handle chat input and generate AI responses"""
        
        if not user_input or user_input.strip() == "":
            return chat_history or [], "", chat_history or []
        
        # Initialize chat history if None
        if chat_history is None:
            chat_history = []
        
        # Add user message
        timestamp = datetime.now().strftime("%H:%M")
        user_message = {
            'type': 'user',
            'content': user_input.strip(),
            'timestamp': timestamp
        }
        chat_history.append(user_message)
        
        # Generate AI response based on context
        ai_response = generate_ai_response(
            user_input.strip(), 
            df, 
            selected_students, 
            selected_group, 
            student_count,
            prediction_model
        )
        
        ai_message = {
            'type': 'ai',
            'content': ai_response,
            'timestamp': timestamp
        }
        chat_history.append(ai_message)
        
        # Convert to display format
        chat_display = format_chat_messages(chat_history)
        
        return chat_display, "", chat_history

    @app.callback(
        Output('chat-context-info', 'children'),
        Input('selected-students-store', 'data'),
        Input('selected-performance-group-store', 'data'),
        Input('student-count-slider', 'value')
    )
    def update_chat_context(selected_students, selected_group, student_count):
        """Update the context info displayed in chat"""
        
        context_parts = []
        
        # Dataset size
        context_parts.append(f"Dataset: {len(df)} students")
        
        # Current view
        if student_count < len(df):
            context_parts.append(f"Viewing: {student_count} students")
        
        # Performance group filter
        if selected_group and selected_group != 'All':
            context_parts.append(f"Group: {selected_group}")
        
        # Selection info
        if selected_students:
            context_parts.append(f"Selected: {len(selected_students)} students")
        
        context_text = " | ".join(context_parts)
        
        return html.Div([
            html.Small(context_text, style={'color': '#666', 'fontSize': '10px'})
        ])


def generate_ai_response(user_input, df, selected_students=None, selected_group=None, 
                        student_count=None, prediction_model=None):
    """Generate contextual AI responses based on user input and current data state"""
    
    user_input_lower = user_input.lower()
    
    # Get current data context
    current_df = df.head(student_count) if student_count else df
    
    # Filter by performance group if selected
    if selected_group and selected_group != 'All':
        performance_categories = categorize_performance(current_df['gpa'])
        current_df = current_df[performance_categories == selected_group]
    
    # Filter by selected students if any
    analysis_df = current_df
    if selected_students:
        analysis_df = current_df[current_df['student_id'].isin(selected_students)]
        context_prefix = f"Looking at your selection of {len(selected_students)} students: "
    else:
        context_prefix = f"Based on the current dataset of {len(analysis_df)} students: "
    
    # Pattern matching for different types of questions
    if any(word in user_input_lower for word in ['summary', 'overview', 'describe', 'tell me about']):
        return generate_summary_response(analysis_df, context_prefix)
    
    elif any(word in user_input_lower for word in ['gpa', 'grade', 'performance', 'academic']):
        return generate_performance_response(analysis_df, context_prefix)
    
    elif any(word in user_input_lower for word in ['study', 'hours', 'time']):
        return generate_study_response(analysis_df, context_prefix)
    
    elif any(word in user_input_lower for word in ['mental', 'health', 'wellbeing', 'stress']):
        return generate_mental_health_response(analysis_df, context_prefix)
    
    elif any(word in user_input_lower for word in ['sleep', 'rest']):
        return generate_sleep_response(analysis_df, context_prefix)
    
    elif any(word in user_input_lower for word in ['social', 'media', 'screen']):
        return generate_social_media_response(analysis_df, context_prefix)
    
    elif any(word in user_input_lower for word in ['exercise', 'physical', 'fitness', 'activity']):
        return generate_exercise_response(analysis_df, context_prefix)
    
    elif any(word in user_input_lower for word in ['attendance', 'class', 'present']):
        return generate_attendance_response(analysis_df, context_prefix)
    
    elif any(word in user_input_lower for word in ['correlation', 'relationship', 'related']):
        return generate_correlation_response(analysis_df, context_prefix)
    
    elif any(word in user_input_lower for word in ['compare', 'difference', 'vs', 'versus']):
        return generate_comparison_response(analysis_df, context_prefix)
    
    elif any(word in user_input_lower for word in ['predict', 'prediction', 'forecast']):
        return generate_prediction_response(analysis_df, context_prefix, prediction_model)
    
    elif any(word in user_input_lower for word in ['help', 'what can', 'how to']):
        return generate_help_response()
    
    else:
        # Default response with basic statistics
        return generate_default_response(analysis_df, context_prefix, user_input)


def generate_summary_response(df, context_prefix):
    """Generate a summary of the current data"""
    if len(df) == 0:
        return "No students in the current selection to analyze."
    
    avg_gpa = df['gpa'].mean()
    avg_study_hours = df['study_hours_per_day'].mean()
    avg_mental_health = df['mental_health_rating'].mean()
    
    performance_dist = categorize_performance(df['gpa']).value_counts()
    
    response = f"{context_prefix}\n\n"
    response += f"📊 **Key Metrics:**\n"
    response += f"• Average GPA: {avg_gpa:.2f}\n"
    response += f"• Average study hours/day: {avg_study_hours:.1f}\n"
    response += f"• Average mental health rating: {avg_mental_health:.1f}/10\n\n"
    response += f"🎯 **Performance Distribution:**\n"
    for category, count in performance_dist.items():
        percentage = (count / len(df)) * 100
        response += f"• {category}: {count} students ({percentage:.1f}%)\n"
    
    return response


def generate_performance_response(df, context_prefix):
    """Generate GPA/performance related insights"""
    if len(df) == 0:
        return "No students in the current selection to analyze."
    
    avg_gpa = df['gpa'].mean()
    min_gpa = df['gpa'].min()
    max_gpa = df['gpa'].max()
    
    # Find factors correlated with GPA
    correlations = df.select_dtypes(include=['float64', 'int64']).corr()['gpa'].abs().sort_values(ascending=False)
    top_factors = correlations[1:4]  # Exclude GPA itself
    
    response = f"{context_prefix}\n\n"
    response += f"🎓 **Academic Performance Analysis:**\n"
    response += f"• Average GPA: {avg_gpa:.2f}\n"
    response += f"• GPA Range: {min_gpa:.2f} - {max_gpa:.2f}\n\n"
    response += f"🔍 **Top factors correlated with GPA:**\n"
    for factor, correlation in top_factors.items():
        response += f"• {factor.replace('_', ' ').title()}: {correlation:.3f}\n"
    
    return response


def generate_study_response(df, context_prefix):
    """Generate study habits insights"""
    if len(df) == 0:
        return "No students in the current selection to analyze."
    
    avg_study_hours = df['study_hours_per_day'].mean()
    study_gpa_corr = df['study_hours_per_day'].corr(df['gpa'])
    
    # Study hours distribution
    study_bins = pd.cut(df['study_hours_per_day'], bins=3, labels=['Low', 'Medium', 'High'])
    study_dist = study_bins.value_counts()
    
    response = f"{context_prefix}\n\n"
    response += f"📚 **Study Habits Analysis:**\n"
    response += f"• Average study hours per day: {avg_study_hours:.1f}\n"
    response += f"• Correlation with GPA: {study_gpa_corr:.3f}\n\n"
    response += f"📊 **Study Time Distribution:**\n"
    for category, count in study_dist.items():
        percentage = (count / len(df)) * 100
        response += f"• {category} study time: {count} students ({percentage:.1f}%)\n"
    
    return response


def generate_mental_health_response(df, context_prefix):
    """Generate mental health insights"""
    if len(df) == 0:
        return "No students in the current selection to analyze."
    
    avg_mental_health = df['mental_health_rating'].mean()
    mental_gpa_corr = df['mental_health_rating'].corr(df['gpa'])
    
    # Mental health categories
    mental_health_bins = pd.cut(df['mental_health_rating'], 
                               bins=[0, 4, 7, 10], 
                               labels=['Low (1-4)', 'Medium (5-7)', 'High (8-10)'])
    mental_dist = mental_health_bins.value_counts()
    
    response = f"{context_prefix}\n\n"
    response += f"🧠 **Mental Health Analysis:**\n"
    response += f"• Average mental health rating: {avg_mental_health:.1f}/10\n"
    response += f"• Correlation with GPA: {mental_gpa_corr:.3f}\n\n"
    response += f"📊 **Mental Health Distribution:**\n"
    for category, count in mental_dist.items():
        percentage = (count / len(df)) * 100
        response += f"• {category}: {count} students ({percentage:.1f}%)\n"
    
    return response


def generate_sleep_response(df, context_prefix):
    """Generate sleep-related insights"""
    if len(df) == 0:
        return "No students in the current selection to analyze."
    
    avg_sleep = df['sleep_hours'].mean()
    sleep_gpa_corr = df['sleep_hours'].corr(df['gpa'])
    
    response = f"{context_prefix}\n\n"
    response += f"😴 **Sleep Pattern Analysis:**\n"
    response += f"• Average sleep hours: {avg_sleep:.1f} hours\n"
    response += f"• Correlation with GPA: {sleep_gpa_corr:.3f}\n"
    
    # Sleep recommendations
    if avg_sleep < 7:
        response += f"\n💡 **Insight:** Students are getting less than the recommended 7-9 hours of sleep."
    elif avg_sleep > 9:
        response += f"\n💡 **Insight:** Students are getting more sleep than typically recommended."
    else:
        response += f"\n✅ **Insight:** Students are getting a healthy amount of sleep."
    
    return response


def generate_social_media_response(df, context_prefix):
    """Generate social media usage insights"""
    if len(df) == 0:
        return "No students in the current selection to analyze."
    
    avg_social_media = df['social_media_hours'].mean()
    social_gpa_corr = df['social_media_hours'].corr(df['gpa'])
    
    response = f"{context_prefix}\n\n"
    response += f"📱 **Social Media Usage Analysis:**\n"
    response += f"• Average social media hours per day: {avg_social_media:.1f}\n"
    response += f"• Correlation with GPA: {social_gpa_corr:.3f}\n"
    
    if social_gpa_corr < -0.1:
        response += f"\n📉 **Insight:** Higher social media usage appears to be associated with lower GPA."
    elif social_gpa_corr > 0.1:
        response += f"\n📈 **Insight:** Interestingly, social media usage shows a positive correlation with GPA."
    else:
        response += f"\n➡️ **Insight:** Social media usage shows little correlation with academic performance."
    
    return response


def generate_exercise_response(df, context_prefix):
    """Generate exercise-related insights"""
    if len(df) == 0:
        return "No students in the current selection to analyze."
    
    avg_exercise = df['exercise_frequency'].mean()
    exercise_gpa_corr = df['exercise_frequency'].corr(df['gpa'])
    
    response = f"{context_prefix}\n\n"
    response += f"🏃 **Exercise Frequency Analysis:**\n"
    response += f"• Average exercise frequency: {avg_exercise:.1f} times per week\n"
    response += f"• Correlation with GPA: {exercise_gpa_corr:.3f}\n"
    
    if exercise_gpa_corr > 0.1:
        response += f"\n💪 **Insight:** Regular exercise appears to be positively associated with academic performance."
    else:
        response += f"\n🤔 **Insight:** Exercise frequency shows minimal correlation with GPA in this dataset."
    
    return response


def generate_attendance_response(df, context_prefix):
    """Generate attendance-related insights"""
    if len(df) == 0:
        return "No students in the current selection to analyze."
    
    avg_attendance = df['attendance_percentage'].mean()
    attendance_gpa_corr = df['attendance_percentage'].corr(df['gpa'])
    
    response = f"{context_prefix}\n\n"
    response += f"🎓 **Attendance Analysis:**\n"
    response += f"• Average attendance: {avg_attendance:.1f}%\n"
    response += f"• Correlation with GPA: {attendance_gpa_corr:.3f}\n"
    
    if attendance_gpa_corr > 0.3:
        response += f"\n✅ **Strong Insight:** Class attendance shows a strong positive correlation with academic performance."
    elif attendance_gpa_corr > 0.1:
        response += f"\n📈 **Insight:** Class attendance is positively associated with better grades."
    else:
        response += f"\n🤔 **Insight:** Attendance shows weaker correlation with GPA than expected."
    
    return response


def generate_correlation_response(df, context_prefix):
    """Generate correlation analysis"""
    if len(df) == 0:
        return "No students in the current selection to analyze."
    
    # Calculate correlations with GPA
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    correlations = df[numeric_cols].corr()['gpa'].abs().sort_values(ascending=False)
    
    response = f"{context_prefix}\n\n"
    response += f"🔗 **Correlation Analysis with GPA:**\n"
    for col, corr in correlations.items():
        if col != 'gpa':
            strength = "Strong" if corr > 0.5 else "Moderate" if corr > 0.3 else "Weak"
            response += f"• {col.replace('_', ' ').title()}: {corr:.3f} ({strength})\n"
    
    return response


def generate_comparison_response(df, context_prefix):
    """Generate comparison between performance groups"""
    if len(df) == 0:
        return "No students in the current selection to analyze."
    
    performance_categories = categorize_performance(df['gpa'])
    
    response = f"{context_prefix}\n\n"
    response += f"⚖️ **Performance Group Comparison:**\n\n"
    
    for category in performance_categories.unique():
        group_data = df[performance_categories == category]
        if len(group_data) > 0:
            response += f"**{category} Performers ({len(group_data)} students):**\n"
            response += f"• Avg Study Hours: {group_data['study_hours_per_day'].mean():.1f}\n"
            response += f"• Avg Mental Health: {group_data['mental_health_rating'].mean():.1f}\n"
            response += f"• Avg Attendance: {group_data['attendance_percentage'].mean():.1f}%\n\n"
    
    return response


def generate_prediction_response(df, context_prefix, prediction_model):
    """Generate prediction-related insights"""
    if prediction_model is None:
        return "Prediction model is not available for analysis."
    
    response = f"{context_prefix}\n\n"
    response += f"🔮 **Prediction Insights:**\n"
    response += f"Use the prediction panel on the right to explore how different factors affect academic performance.\n\n"
    response += f"You can adjust variables like:\n"
    response += f"• Study hours per day\n"
    response += f"• Mental health rating\n"
    response += f"• Sleep hours\n"
    response += f"• Exercise frequency\n"
    response += f"• And more...\n\n"
    response += f"The model will show you the predicted performance probability!"
    
    return response


def generate_help_response():
    """Generate help information"""
    response = "🤖 **I can help you analyze student data! Here are some things you can ask:**\n\n"
    response += "📊 **General Analysis:**\n"
    response += "• 'Give me a summary' or 'Describe the data'\n"
    response += "• 'What's the correlation between study time and GPA?'\n\n"
    response += "🎓 **Academic Performance:**\n"
    response += "• 'Tell me about GPA distribution'\n"
    response += "• 'How do high performers differ from low performers?'\n\n"
    response += "💡 **Specific Factors:**\n"
    response += "• Ask about study habits, mental health, sleep, exercise, attendance\n"
    response += "• 'How does social media usage affect grades?'\n\n"
    response += "🔍 **Interactive Features:**\n"
    response += "• Select students in the scatter plot to analyze specific groups\n"
    response += "• Click on performance groups in the donut chart\n"
    response += "• Use the prediction panel to explore scenarios\n"
    
    return response


def generate_default_response(df, context_prefix, user_input):
    """Generate a default response with basic statistics"""
    if len(df) == 0:
        return "No students in the current selection to analyze."
    
    response = f"{context_prefix}\n\n"
    response += f"I analyzed your question about: '{user_input}'\n\n"
    response += f"📊 **Quick Stats:**\n"
    response += f"• Students in analysis: {len(df)}\n"
    response += f"• Average GPA: {df['gpa'].mean():.2f}\n"
    response += f"• Average study hours: {df['study_hours_per_day'].mean():.1f}/day\n\n"
    response += f"💬 Ask me more specific questions about study habits, performance, or correlations!"
    
    return response


def format_chat_messages(chat_history):
    """Format chat messages for display"""
    if not chat_history:
        return []
    
    messages = []
    for msg in chat_history:
        if msg['type'] == 'user':
            messages.append(
                html.Div([
                    html.Div([
                        html.Span(msg['content'], className='user-message-text'),
                        html.Span(msg['timestamp'], className='message-timestamp')
                    ], className='user-message')
                ], className='message-container user-container')
            )
        else:  # AI message
            # Convert markdown-style formatting to HTML
            content = msg['content'].replace('**', '')  # Remove markdown bold
            content = content.replace('•', '•')  # Ensure bullet points
            
            messages.append(
                html.Div([
                    html.Div([
                        html.Pre(content, className='ai-message-text'),
                        html.Span(msg['timestamp'], className='message-timestamp')
                    ], className='ai-message')
                ], className='message-container ai-container')
            )
    
    return messages