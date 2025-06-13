from dash import Input, Output, State, html, callback_context
import json
from datetime import datetime
import pandas as pd
from utils.data_processing import categorize_performance
from utils.AI_Agent import initialize_ai_agent, get_ai_response


def register_chat_callbacks(app):
    """Register all chat-related callbacks for the dashboard"""
    
    # Access the globally stored data
    df = app.df
    
    # Initialize the AI agent
    ai_agent = initialize_ai_agent("data/student_performance_large_dataset.csv")

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
        
        # Generate AI response using the AI agent
        ai_response = get_ai_response(user_input.strip())
        
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