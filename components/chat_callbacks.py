from dash import Input, Output, State, html, callback_context
import json
from datetime import datetime
import pandas as pd
from utils.data_processing import categorize_performance
from utils.AI_Agent import initialize_ai_agent, get_ai_response
from components.chat_styles import (
    user_message_style,
    ai_message_style,
    message_container_user,
    message_container_ai,
    timestamp_style
)

def register_chat_callbacks(app):
    """Register all chat-related callbacks for the dashboard"""
    
    # Access the globally stored data
    df = app.df
    
    # Initialize the AI agent
    ai_agent = initialize_ai_agent("data/student_habits_performance.csv")

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
        Output('ai-student-selection-store', 'data'),
        Output('chat-loading-output', 'children'),  # NEW: Loading indicator output
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
            return chat_history or [], "", chat_history or [], None, ""
        
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
        
        # Show loading indicator while processing
        loading_display = format_chat_messages(chat_history)
        
        # Generate AI response using the AI agent (now returns structured data)
        ai_response_data = get_ai_response(user_input.strip())
        
        # Extract analysis text and student IDs
        analysis_text = ai_response_data.get('analysis', 'No response generated')
        student_ids = ai_response_data.get('student_ids', [])
        
        # Add analysis info if student IDs were found
        if student_ids:
            analysis_text += f"\n\n📊 Found {len(student_ids)} relevant students - visualizations updated!"
        
        ai_message = {
            'type': 'ai',
            'content': analysis_text,
            'timestamp': timestamp,
            'student_ids': student_ids  # Store student IDs in message for reference
        }
        chat_history.append(ai_message)
        
        # Convert to display format
        chat_display = format_chat_messages(chat_history)
        
        # Return student IDs for visualization updates, empty loading indicator
        return chat_display, "", chat_history, student_ids, ""

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
        timestamp = html.Span(msg['timestamp'], style=timestamp_style)

        if msg['type'] == 'user':
            messages.append(
                html.Div([
                    html.Div([
                        html.Div(msg['content'], style=user_message_style),
                        timestamp
                    ])
                ], style=message_container_user)
            )
        else:  # AI message
            content = msg['content'].replace('**', '')  # Optional: strip markdown bold
            if msg.get('student_ids'):
                content = f"🎯 {content}"
            
            messages.append(
                html.Div([
                    html.Div([
                        html.Div(content, style=ai_message_style),
                        timestamp
                    ])
                ], style=message_container_ai)
            )
    
    return messages