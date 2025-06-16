# Custom CSS styles
small_tile_style = {
    'backgroundColor': 'white',
    'borderRadius': '12px',
    'boxShadow': '0 3px 10px rgba(0, 0, 0, 0.1)',
    'padding': '15px',
    'margin': '8px',
    'border': '1px solid #e0e0e0',
    'height': '335px',
    'transition': 'all 0.3s ease'
}

prediction_tile_style = {
    'backgroundColor': 'white',
    'borderRadius': '12px',
    'boxShadow': '0 3px 10px rgba(0, 0, 0, 0.1)',
    'padding': '20px',
    'margin': '8px',
    'border': '1px solid #e0e0e0',
    'height': '700px',
    'transition': 'all 0.3s ease'
}

top_bar_style = {
    'display': 'flex',
    'justifyContent': 'space-between',
    'alignItems': 'center',
    'backgroundColor': '#2c3e50',
    'padding': '12px 30px',
    'boxShadow': '0 2px 8px rgba(0, 0, 0, 0.15)',
    'position': 'sticky',
    'top': '0',
    'zIndex': '1000'
}

main_container_style = {
    'backgroundColor': '#f8f9fa',
    'height': '100vh',
    'fontFamily': 'Segoe UI, Tahoma, Geneva, Verdana, sans-serif',
    'overflow': 'hidden'
}

# AI Chat specific styles Added new
chat_button_style = {
    'position': 'fixed',
    'bottom': '20px',
    'right': '20px',
    'width': '50px',
    'height': '50px',
    'backgroundColor': '#3498db',
    'color': 'white',
    'border': 'none',
    'borderRadius': '50%',
    'cursor': 'pointer',
    'fontSize': '24px',
    'boxShadow': '0 4px 12px rgba(0,0,0,0.3)',
    'zIndex': '1001',
    'display': 'flex',
    'alignItems': 'center',
    'justifyContent': 'center',
    'transition': 'all 0.3s ease'
}

chat_container_style = {
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
    'display': 'none',  # Initially hidden
    'flexDirection': 'column'
}

chat_header_style = {
    'padding': '15px',
    'backgroundColor': '#34495e',
    'color': 'white',
    'borderRadius': '10px 10px 0 0',
    'display': 'flex',
    'justifyContent': 'space-between',
    'alignItems': 'center',
    'borderBottom': '1px solid #4a5f7a'
}

chat_messages_style = {
    'flex': '1',
    'padding': '10px',
    'overflowY': 'auto',
    'backgroundColor': '#ecf0f1',
    'display': 'flex',
    'flexDirection': 'column',
    'gap': '10px',
    'maxHeight': '100%',          
    'overflowWrap': 'break-word'   
}

chat_input_container_style = {
    'padding': '10px',
    'backgroundColor': '#34495e',
    'borderRadius': '0 0 10px 10px',
    'display': 'flex',
    'gap': '5px'
}

chat_input_style = {
    'flex': '1',
    'padding': '8px 12px',
    'border': '1px solid #4a5f7a',
    'borderRadius': '5px',
    'backgroundColor': '#2c3e50',
    'color': 'white',
    'fontSize': '14px'
}

chat_send_button_style = {
    'padding': '8px 12px',
    'backgroundColor': '#3498db',
    'color': 'white',
    'border': 'none',
    'borderRadius': '5px',
    'cursor': 'pointer',
    'fontSize': '14px'
}

message_style_ai = {
    'padding': '8px 12px',
    'backgroundColor': '#3498db',
    'color': 'white',
    'borderRadius': '10px',
    'marginBottom': '5px',
    'fontSize': '12px',
    'lineHeight': '1.4',
    'whiteSpace': 'pre-wrap',
    'maxWidth': '80%',
    'wordWrap': 'break-word',
    'alignSelf': 'flex-start',  # Ensure AI messages align left
    'overflowWrap': 'break-word',  # Extra safety for long text
}


message_style_user = {
    'padding': '8px 12px',
    'backgroundColor': '#95a5a6',
    'color': 'white',
    'borderRadius': '10px',
    'marginBottom': '5px',
    'fontSize': '12px',
    'lineHeight': '1.4',
    'alignSelf': 'flex-end',
    'maxWidth': '80%',
    'whiteSpace': 'pre-wrap',
    'wordWrap': 'break-word',
    'overflowWrap': 'break-word'
}