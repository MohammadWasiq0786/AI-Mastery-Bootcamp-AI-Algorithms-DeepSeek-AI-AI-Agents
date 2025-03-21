from fastapi import FastAPI
import requests

app = FastAPI()
OLLAMA_URL = "http://localhost:11434/api/generate"

@app.post("/complete_code/")
def complete_code(data: dict):
    payload = {"model": "deepseek-r1", "prompt": f"Complete code:\n\n{data['code_snippet']}", "stream": False}
    response = requests.post(OLLAMA_URL, json=payload)
    return response.json().get("response", "No completion available.")

# Run with: uvicorn app:app --reload
