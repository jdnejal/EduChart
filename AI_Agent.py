import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.agents import AgentType

load_dotenv()




df = pd.read_csv("data/student_performance_large_dataset.csv")

custom_prefix = """Follow these rules:
1. Use ONLY the provided DataFrame columns: {', '.join(df.columns.tolist())}
2. If question can't be answered with data, respond: "Cannot determine answer"
3. Don't use any external knowledge or data
4. Answer questions based on the DataFrame content only
"""


api_key="AIzaSyCPGdkMQlTioIB1gLjZaqAHFbv7NQtK78w"

llm = ChatGoogleGenerativeAI(    
            google_api_key=api_key, 
            model="models/gemini-1.5-flash"
            )

agent = create_pandas_dataframe_agent(
    llm,
    df,
    verbose=False,
    allow_dangerous_code=True ,
    custom_prefix=custom_prefix,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    )       

# question ="Summarize the data?"
# response=agent.invoke(question) 
# print(response['output'])


