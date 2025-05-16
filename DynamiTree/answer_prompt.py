ANSWER_QUERY = '''
You must answer the following question strictly and only based on the provided contexts. Do not invent or infer any information that is not explicitly present in the contexts. Make sure to include all the dates relevant to the information.
Do not add extra details, generalizations, or guesses. Copy the relevant information exactly as it appears where appropriate. The answer should be a string.

Contexts:
{contexts}

Question:
{question}

Provide your answer in a JSON object with the following format:
{{
    "answer": "Your answer here"
}}

'''