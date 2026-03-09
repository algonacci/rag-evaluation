import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
import os
from tqdm import tqdm

load_dotenv()

client = OpenAI(
    base_url=os.getenv('LLM_BASE_URL'),
    api_key=os.getenv('LLM_API_KEY')
)

model = os.getenv('LLM_MODEL_NAME')
if not model:
    raise ValueError("LLM_MODEL_NAME not found in .env file")

df = pd.read_csv('02_data_evaluation.csv')

df_test = df

def generate_answer(question):
    try:
        response = client.chat.completions.create(
            model=str(model),
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Always answer in clear English. Answer the question accurately and concisely based on the provided context or your knowledge."},
                {"role": "user", "content": question}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error generating answer for question: {question[:50]}... Error: {e}")
        return None

print(f"Generating answers for {len(df_test)} questions...")

generated_answers = []
for question in tqdm(df_test['question'], desc="Generating answers"):
    answer = generate_answer(question)
    generated_answers.append(answer)

df_test['generated_answer'] = generated_answers

df_test.to_csv('03_with_answers.csv', index=False)

print(f"Answers saved to 03_with_answers.csv")
