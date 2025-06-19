# EduChart

The EduChart dashboard is a tool for instructors and student advisors to analyze data about students.
The tool allows the users to filter data, predict exam scores and analyze using a custom AI-powered chatbot.
The tool is built in Python with Dash and has uses the sklearn RandomForestRegressor for its predictions. In addition, the tool includes an AI-driven chatbot which is built upon Langchains framework using the Gemini LLM.

## Running the tool

1. Download the packages in `requirements.txt`
2. Run python `app.py` in the terminal
3. The file will now run on a local machine and can be accessed by opening the `http://127.0.0.1:8050` path (might differ per machine).
4. Enjoy!

# Folder Structure

## EDA and Prediction Model

In the preparation folder `EDA_Pred.ipynb`, the exploratory data analysis, of the dataset can be found.
This includes finding the most important features using SHAP and finding correlation between features.
In addition, the prediction model to predict exam scores used in EduChart can be found. The actual prediction model weights can be found in the models folder.

## Dataset

The dataset can be found in the data folder and is retrieved from Kaggle.

Dataset: https://www.kaggle.com/code/jayaantanaath/student-habits-vs-academic-performance-ml-90

## Components and layout

The layout, styling and actual callback code for running each visualizations in the dashboard can be found in the components folder.
The component folders handles interactions between visualizations.
For each plot there is an additional unique file in utils/plots which contains the individual working for each plot.

## Other utilities

The other files in utils are `AI_Agent.py` and `data_processing.py`.
The first file contains the internal structure of the AI agent and how it handles calls to the Gemini API. The latter contains general reuable functions for handling data and contains code for retrieving the right color schemes.
