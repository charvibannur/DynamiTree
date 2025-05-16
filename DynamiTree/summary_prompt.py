SUMMARIZE_FACTS = '''
Given a numbered list of facts, create a clear 1-4 sentence summary that captures the key message and essential details. Reference fact indices in [brackets] to show which facts support each statement. The summary should be concise while including the most important information that can be directly supported by the provided facts. Return only a JSON object with a "summary" key.

Example:
Facts from Climate Report:
[1] Global temperatures rose 1.1°C since pre-industrial times
[2] Arctic sea ice decreased 13% per decade
[3] 2020-2023 were the warmest years on record
[4] Greenhouse gas emissions reached record levels in 2023
[5] Sea levels are rising at twice the rate of the 20th century

Output:
{{
    "summary": "According to the Climate Report, global temperatures have risen 1.1°C since pre-industrial times [1], leading to rapid Arctic ice loss of 13% per decade [2] and record-breaking warm years from 2020-2023 [3]. The climate crisis has accelerated, with greenhouse gas emissions hitting new highs in 2023 [4] and sea levels rising at double the 20th century rate [5]."
}}

Now summarize these facts:
{facts}
'''