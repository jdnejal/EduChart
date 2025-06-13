import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.agents import AgentType
import json
import threading
import time

class StudentDataAIAgent:
    """Enhanced AI Agent for student data analysis with Dash integration"""
    
    def __init__(self, csv_path="data/student_performance_large_dataset.csv"):
        load_dotenv()
        
        # Load the dataset
        self.df = pd.read_csv(csv_path)
        
        # Enhanced custom prefix with more context
        self.custom_prefix = f"""
        You are an AI assistant specialized in analyzing student performance data. 

        IMPORTANT RULES:
        1. Use ONLY the provided DataFrame columns: {', '.join(self.df.columns.tolist())}
        2. The dataset contains {len(self.df)} student records
        3. If a question cannot be answered with the available data, respond: "I cannot determine that from the available data."
        4. Don't use any external knowledge - only analyze the provided DataFrame
        5. When showing student IDs, limit to 10 examples unless specifically asked for more
        6. For statistical queries, provide clear numbers and percentages
        7. Always be helpful and explain your findings in a user-friendly way

        DATASET OVERVIEW:
        - Total students: {len(self.df)}
        - Key metrics available: academic performance, study habits, lifestyle factors
        - You can analyze correlations, patterns, and provide insights
        
        When users ask about "interesting students" or patterns, focus on:
        - Outliers in performance vs study habits
        - Unusual combinations of lifestyle factors
        - Students with unexpected results given their inputs
        """
        
        # Initialize the LLM
        self.api_key = os.getenv("GOOGLE_API_KEY", "AIzaSyCPGdkMQlTioIB1gLjZaqAHFbv7NQtK78w")
        self.llm = ChatGoogleGenerativeAI(
            google_api_key=self.api_key,
            model="models/gemini-1.5-flash",
            temperature=0.1  # Lower temperature for more consistent responses
        )
        
        # Create the agent
        self.agent = create_pandas_dataframe_agent(
            self.llm,
            self.df,
            verbose=False,
            allow_dangerous_code=True,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            prefix=self.custom_prefix
        )
        
        # Cache for performance
        self.response_cache = {}
        
    def get_response(self, question, timeout=30):
        """
        Get AI response with timeout and error handling
        
        Args:
            question (str): User's question
            timeout (int): Timeout in seconds
            
        Returns:
            str: AI response or error message
        """
        # Check cache first
        if question in self.response_cache:
            return self.response_cache[question]
        
        try:
            # Use threading for timeout functionality
            result = {"response": None, "error": None}
            
            def get_agent_response():
                try:
                    response = self.agent.invoke(question)
                    result["response"] = response.get('output', 'No response generated')
                except Exception as e:
                    result["error"] = str(e)
            
            thread = threading.Thread(target=get_agent_response)
            thread.start()
            thread.join(timeout=timeout)
            
            if thread.is_alive():
                return "⏱️ The query is taking too long to process. Please try a simpler question."
            
            if result["error"]:
                return f"❌ Error processing your question: {result['error']}"
            
            if result["response"]:
                # Cache successful responses
                self.response_cache[question] = result["response"]
                return result["response"]
            else:
                return "❌ No response was generated. Please try rephrasing your question."
                
        except Exception as e:
            return f"❌ Unexpected error: {str(e)}"
    
    def get_dataset_summary(self):
        """Get a quick summary of the dataset"""
        summary = {
            "total_students": len(self.df),
            "columns": list(self.df.columns),
            "sample_data": self.df.head(3).to_dict('records')
        }
        return summary
    
    def suggest_questions(self):
        """Provide suggested questions users can ask"""
        suggestions = [
            "What are the top 5 factors that correlate with high academic performance?",
            "Show me students with unusual study patterns",
            "Which students have high social media usage but still perform well?",
            "What's the average study time for students with different performance levels?",
            "Find students who sleep less than 6 hours but have high grades",
            "What patterns do you see in exercise frequency vs academic performance?",
            "Show me the distribution of mental health ratings across the dataset",
            "Which students have the most balanced lifestyle and academic performance?"
        ]
        return suggestions

# Global instance for the Dash app
ai_agent = None

def initialize_ai_agent(csv_path="data/student_performance_large_dataset.csv"):
    """Initialize the global AI agent instance"""
    global ai_agent
    if ai_agent is None:
        ai_agent = StudentDataAIAgent(csv_path)
    return ai_agent

def get_ai_response(question):
    """Get AI response - main function for Dash callbacks"""
    global ai_agent
    if ai_agent is None:
        return "❌ AI Agent not initialized. Please refresh the page."
    
    return ai_agent.get_response(question)

def get_suggested_questions():
    """Get suggested questions for the chat interface"""
    global ai_agent
    if ai_agent is None:
        return []
    return ai_agent.suggest_questions()

# Test function
def test_agent():
    """Test function to verify agent functionality"""
    agent = StudentDataAIAgent()
    
    test_questions = [
        "How many students are in the dataset?",
        "What are the column names?",
        "Show me 3 interesting students and explain why they are interesting",
        "What's the correlation between study hours and performance?"
    ]
    
    print("Testing AI Agent...")
    for question in test_questions:
        print(f"\nQ: {question}")
        response = agent.get_response(question)
        print(f"A: {response}")
        print("-" * 50)

if __name__ == "__main__":
    test_agent()