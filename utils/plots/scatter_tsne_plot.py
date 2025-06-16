import plotly.express as px
import plotly.graph_objects as go
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd

def create_scatter_tsne_plot(df, plot_type, x_axis, y_axis, search_value, student_count,
                            selected_group, color_scheme, categorize_performance):
    """Create scatter plot or t-SNE visualization"""
    
    # Sample and prepare data
    filtered_df = df.sample(n=min(student_count, len(df)), random_state=42).copy()
    filtered_df['Performance_Group'] = filtered_df['exam_score'].apply(categorize_performance)
    
    # Filter by selected group if not 'All'
    if selected_group != 'All' and selected_group in ['High (≥80%)', 'Medium (50-79%)', 'Low (<50%)']:
        display_df = filtered_df[filtered_df['Performance_Group'] == selected_group].copy()
    else:
        display_df = filtered_df.copy()
    
    # Check if we have data to display
    if len(display_df) == 0:
        return go.Figure().add_annotation(
            text="No data for selected group",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=12, color="gray")
        )
    
    if plot_type == 'scatter':
        # Create scatter plot with proper custom_data
        fig = px.scatter(
            display_df,
            x=x_axis,
            y=y_axis,
            color='exam_score',
            size="attendance_percentage",
            size_max=8,
            hover_data={"student_id": True},
            color_continuous_scale=color_scheme,
        )
        
        # Manually set customdata after creation
        fig.update_traces(
            customdata=display_df[['student_id']].values,
            hovertemplate="<b>Student ID: %{customdata[0]}</b><br>" +
                         f"{x_axis}: %{{x}}<br>" +
                         f"{y_axis}: %{{y}}<br>" +
                         "Exam Score: %{marker.color}<br>" +
                         "Attendance: %{marker.size}<br>" +
                         "<extra></extra>"
        )
        
        # Highlight searched student
        if search_value and search_value.strip():
            matched = display_df[display_df['student_id'].astype(str).str.contains(search_value.strip(), case=False)]
            if not matched.empty:
                student = matched.iloc[0]
                fig.add_trace(go.Scatter(
                    x=[student[x_axis]],
                    y=[student[y_axis]],
                    mode='markers',
                    marker=dict(color='purple', size=15, symbol='star'),
                    name=f"ID: {student['student_id']}",
                    showlegend=True,
                    customdata=[[student['student_id']]]
                ))
    
    else:  # t-SNE
        def encodedData(df):
            categorical_cols = ['gender', 'part_time_job', 'diet_quality',
                              'parental_education_level', 'internet_quality',
                              'extracurricular_participation', 'Performance_Group']
            
            df_encoded = df.copy()
            
            # Encode categorical columns
            for col in categorical_cols:
                if col in df_encoded.columns:
                    le = LabelEncoder()
                    df_encoded[col] = le.fit_transform(df_encoded[col])
            
            # Drop non-numeric columns
            df_encoded = df_encoded.drop(columns=['student_id'], errors='ignore')
            
            # Separate features and target
            X = df_encoded.drop(columns=['exam_score'], errors='ignore')
            
            # Scale the features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            return X_scaled
        
        try:
            X = encodedData(display_df)
            
            # Perform t-SNE
            reducer = TSNE(n_components=2, perplexity=min(30, len(display_df)-1), random_state=42)
            X_reduced = reducer.fit_transform(X)
            
            # FIXED: Create t-SNE plot with simplified colorbar
            fig = go.Figure()
            
            # Get color values
            color_values = display_df['exam_score'].values
            
            fig.add_trace(go.Scatter(
                x=X_reduced[:, 0],
                y=X_reduced[:, 1],
                mode='markers',
                marker=dict(
                    color=color_values,
                    colorscale=color_scheme,
                    showscale=False,  # Show colorbar
                    colorbar=dict(title='Exam Score'),
                    size=8
                ),
                customdata=display_df[['student_id']].values,
                hovertemplate="<b>Student ID: %{customdata[0]}</b><br>" +
                             "t-SNE 1: %{x:.2f}<br>" +
                             "t-SNE 2: %{y:.2f}<br>" +
                             "Exam Score: %{marker.color}<br>" +
                             "<extra></extra>",
                name='Students'
            ))
            
            # Update layout for t-SNE
            fig.update_layout(
                xaxis_title='t-SNE Dimension 1',
                yaxis_title='t-SNE Dimension 2'
            )
            
            # Highlight searched student in t-SNE
            if search_value and search_value.strip():
                matched_indices = display_df[display_df['student_id'].astype(str).str.contains(search_value.strip(), case=False)].index
                original_indices = display_df.index
                
                for matched_idx in matched_indices:
                    try:
                        pos_in_reduced = list(original_indices).index(matched_idx)
                        if pos_in_reduced < len(X_reduced):
                            fig.add_trace(go.Scatter(
                                x=[X_reduced[pos_in_reduced, 0]],
                                y=[X_reduced[pos_in_reduced, 1]],
                                mode='markers',
                                marker=dict(color='purple', size=15, symbol='star'),
                                name=f"ID: {display_df.loc[matched_idx, 'student_id']}",
                                showlegend=True,
                                customdata=[[display_df.loc[matched_idx, 'student_id']]]
                            ))
                    except (ValueError, IndexError):
                        continue  # Skip if student not found in reduced data
        
        except Exception as e:
            return go.Figure().add_annotation(
                text=f"t-SNE calculation failed: {str(e)}",
                xref="paper", yref="paper", x=0.5, y=0.5,
                showarrow=False, font=dict(size=12, color="red")
            )
    
    # Update layout
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=10),
        margin=dict(l=30, r=30, t=20, b=30),
        legend=dict(font=dict(size=8), orientation="h", y=-0.1),
        dragmode='select',  # Enable box selection by default
        # Improve overall appearance
        hovermode='closest'
    )
    
    return fig