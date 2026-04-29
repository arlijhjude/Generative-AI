import pandas as pd
import openai
import os
import json
import time
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- 1. SETUP: INITIALIZE API CLIENT ---
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OpenAI API key not found. Please set it in a .env file.")

client = openai.OpenAI(api_key=api_key)

# --- 2. DATA: LOAD YOUR DATAFRAME ---
# Using relative paths makes the script work on any computer
input_file = 'Survey_Data.csv'
output_file = 'Categorization_GenAI.csv'

if not os.path.exists(input_file):
    print(f"Error: {input_file} not found. Please place the file in the script directory.")
    exit()

raw_data = pd.read_csv(input_file).reset_index()
for_input = raw_data[['index', 'Review Comment']]

# --- 3. PROMPT ENGINEERING ---
system_role_prompt = """
You are an expert data analyst specializing in customer feedback.
Analyze the user's review and provide the following information:
1. 'Sentiment': 'Positive' or 'Negative'.
2. 'Main_Reason': Primary reason (e.g., 'Staff Friendliness').
3. 'Sub_Reason': Specific detail (e.g., 'Cashier was polite').
4. 'Sentiment Category': "PEOPLE", "PROCESS", "SERVICE", or "TECHNOLOGY".

Respond ONLY with a single, valid JSON object.
"""

# --- 4. FUNCTION: API CALL ---
def get_review_analysis(review_text, model="gpt-4o-mini"):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_role_prompt},
                {"role": "user", "content": str(review_text)}
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return {"Sentiment": "Error", "Main_Reason": str(e), "Sub_Reason": "Error", "Sentiment Category": "Error"}

# --- 5. EXECUTION ---
results = []
print("Analyzing customer reviews...")

for index, row in tqdm(for_input.iterrows(), total=for_input.shape[0]):
    analysis_result = get_review_analysis(row['Review Comment'])
    results.append(analysis_result)
    # Adjusted sleep time - 30s is very high; consider 1s unless you have strict rate limits
    time.sleep(1) 

# --- 6. INTEGRATION & OUTPUT ---
results_df = pd.DataFrame(results)
final_df = pd.concat([for_input, results_df], axis=1)

print(f"\n--- Analysis Complete. Saving to {output_file} ---")
final_df.to_csv(output_file, index=False)