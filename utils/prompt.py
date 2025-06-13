import json
from typing import Tuple, List

def createPrompt(question: str) -> str:
    return (
        "You are given a question based on a dataset."
        "Return the list of student ids of the students used in the answer or return an empty list if a selection of student ids will not support reaching the answer.\n\n"
        "Answer the question and return the result in the following JSON format:\n\n"
        "{\n"
        '  "answerText": string,\n'
        '  "studentId": string[]\n'
        "}\n\n"
        f"Question:\n{question}"
    )

def parse_prompt_response(response: str) -> Tuple[str, List[str]]:
    """
    Parses the response string returned by the API prompt call.

    Args:
        response (str): The raw response string (should contain a JSON object).

    Returns:
        Tuple[str, List[str]]: A tuple containing the answerText and a list of studentId strings.

    Raises:
        ValueError: If the response is not valid JSON or required fields are missing.
    """
    try:
        data = json.loads(response)
        answer_text = data["answerText"]
        student_ids = data["studentId"]
        if not isinstance(answer_text, str) or not isinstance(student_ids, list):
            raise ValueError("Invalid types for answerText or studentId")
        return answer_text, student_ids
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
        raise ValueError(f"Failed to parse response: {e}")