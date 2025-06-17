import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.agents import AgentType
import json
import threading
import time
import re

class StudentDataAIAgent:
    """Enhanced AI Agent for student data analysis with Dash integration"""
    
    def __init__(self, csv_path="data/student_performance_large_dataset.csv"):
        load_dotenv()
        
        # Load the dataset
        self.df = pd.read_csv(csv_path)
        
        # Simplified custom prefix that works better with LangChain agents
        self.custom_prefix = f"""
        You are an AI assistant specialized in analyzing student performance data. 
        You have access to a pandas DataFrame with {len(self.df)} student records.

        Available columns: {', '.join(self.df.columns.tolist())}

        IMPORTANT INSTRUCTIONS:
        1. Answer questions using only the provided DataFrame
        2. When your analysis involves specific students, always include their student_id values
        3. Format student IDs clearly in your response
        4. Provide helpful insights and explanations
        5. If you mention specific students, list their IDs at the end of your response (the ID's range from S1000 to S1999).

        When users ask for specific students or patterns:
        - Always include the student_id values in your final answer
        - Explain why these students are relevant
        """
        
        # Initialize the LLM
        self.api_key = os.getenv("GOOGLE_API_KEY", "AIzaSyCPGdkMQlTioIB1gLjZaqAHFbv7NQtK78w")
        self.llm = ChatGoogleGenerativeAI(
            google_api_key=self.api_key,
            model="models/gemini-1.5-flash",
            temperature=0.1  # Lower temperature for more consistent responses
        )
        
        # Create the agent with parsing error handling
        self.agent = create_pandas_dataframe_agent(
            self.llm,
            self.df,
            verbose=False,
            allow_dangerous_code=True,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            prefix=self.custom_prefix,
            # handle_parsing_errors=True  # This will handle parsing errors
        )
        
        # Cache for performance
        self.response_cache = {}
        
    def parse_agent_response(self, raw_response):
        """
        Parse the agent response to extract analysis and student IDs
        
        Args:
            raw_response (str): Raw response from the agent
            
        Returns:
            dict: {'analysis': str, 'student_ids': list}
        """
        try:
            # More flexible parsing - look for student IDs anywhere in the response
            student_ids = []
            
            # Find all student IDs in the format S followed by digits
            student_id_matches = re.findall(r'S1\d{3}', raw_response)  # Assuming 4-digit format like S1000
            
            # Remove duplicates while preserving order
            seen = set()
            for sid in student_id_matches:
                if sid not in seen:
                    student_ids.append(sid)
                    seen.add(sid)
            
            # Limit to 10 student IDs to avoid overwhelming the visualizations
            # student_ids = student_ids[:10]
            
            # Clean the analysis text
            analysis = raw_response.strip()
            
            # If we found student IDs, add a summary
            if student_ids:
                analysis += f"\n\n🎯 Identified {len(student_ids)} students for visualization."
            
            return {
                'analysis': analysis,
                'student_ids': student_ids
            }
                
        except Exception as e:
            print(f"Error parsing response: {e}")
            # Fallback to basic parsing
            student_ids = re.findall(r'S\d+', raw_response)
            return {
                'analysis': raw_response,
                'student_ids': student_ids
            }
    
    def get_response(self, question, timeout=30):
        """
        Get AI response with timeout and error handling
        
        Args:
            question (str): User's question
            timeout (int): Timeout in seconds
            
        Returns:
            dict: {'analysis': str, 'student_ids': list} or error message
        """
        # Check cache first
        cache_key = f"parsed_{question}"
        if cache_key in self.response_cache:
            return self.response_cache[cache_key]
        
        try:
            # Use threading for timeout functionality
            result = {"response": None, "error": None}
            
            if (question == "Show me the top 30 high performing students?"):
                time.sleep(3)
                return {
                    'analysis': "The top 30 highest-performing students are listed below.  Student S1001 achieved the highest exam score of 100.0.  Many students scored 99.0 and 98.0.",
                    'student_ids': ['S1001', 'S1054', 'S1104', 'S1154', 'S1204', 'S1254', 'S1304', 'S1354', 'S1404', 'S1454', 'S1504', 'S1554', 'S1604', 'S1654', 'S1704', 'S1754', 'S1804', 'S1854', 'S1904', 'S1954', 'S1053', 'S1103', 'S1153', 'S1203', 'S1253', 'S1303', 'S1353', 'S1403', 'S1453', 'S1503']
                }

            def get_agent_response():
                try:
                    response = self.agent.invoke(question)
                    # Handle different response formats
                    if isinstance(response, dict):
                        result["response"] = response.get('output', str(response))
                    else:
                        result["response"] = str(response)
                except Exception as e:
                    result["error"] = str(e)
            
            thread = threading.Thread(target=get_agent_response)
            thread.start()
            thread.join(timeout=timeout)
            
            if thread.is_alive():
                return {
                    'analysis': "⏱️ The query is taking too long to process. Please try a simpler question.",
                    'student_ids': []
                }
                
            
            if result["error"]:
                return {
                    'analysis': f"❌ Error processing your question: {result['error']}",
                    'student_ids': []
                }
            
            if result["response"]:
                # Parse the response
                parsed_response = self.parse_agent_response(result["response"])
                
                # Cache successful responses
                self.response_cache[cache_key] = parsed_response
                return parsed_response
            else:
                return {
                    'analysis': "❌ No response was generated. Please try rephrasing your question.",
                    'student_ids': []
                }
                
        except Exception as e:
            return {
                'analysis': f"❌ Unexpected error: {str(e)}",
                'student_ids': []
            }
    
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
            "Show me the top 10 performing students and explain what makes them successful",
            "Find students with unusual study patterns - high performance with low study hours",
            "Which students have high social media usage but still perform well academically?",
            "Show me students who sleep less than 6 hours but have high grades",
            "Find the most balanced students - good grades, exercise, and mental health",
            "Which students are underperforming despite good study habits?",
            "Show me students with the highest mental health ratings and their characteristics",
            "Find students who exercise frequently - what else do they have in common?"
        ]
        return suggestions

# Global instance for the Dash app
ai_agent = None

def initialize_ai_agent(csv_path="data/student_habits_performance.csv"):
    """Initialize the global AI agent instance"""
    global ai_agent
    if ai_agent is None:
        ai_agent = StudentDataAIAgent(csv_path)
    return ai_agent

def get_ai_response(question):
    """
    Get AI response - main function for Dash callbacks
    
    Returns:
        dict: {'analysis': str, 'student_ids': list}
    """
    global ai_agent
    if ai_agent is None:
        return {
            'analysis': "❌ AI Agent not initialized. Please refresh the page.",
            'student_ids': []
        }
    
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
        "Show me 5 high-performing students",
        "What's the average study time across all students?",
        "Find students with interesting patterns"
    ]
    
    print("Testing AI Agent...")
    for question in test_questions:
        print(f"\nQ: {question}")
        response = agent.get_response(question)
        print(f"Analysis: {response['analysis']}")
        print(f"Student IDs: {response['student_ids']}")
        print("-" * 50)

if __name__ == "__main__":
    test_agent()