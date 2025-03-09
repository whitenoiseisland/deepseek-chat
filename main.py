import os
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

app = FastAPI()

MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"

# Lazy loading to avoid startup crashes
tokenizer = None
model = None

def load_model():
    global tokenizer, model
    if tokenizer is None or model is None:
        print("Loading model and tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype=torch.float16
        ).cuda()
        print("Model loaded successfully!")

class ChatRequest(BaseModel):
    prompt: str

@app.post("/chat")
async def chat(request: ChatRequest):
    load_model()  # Ensure the model is loaded before processing

    inputs = tokenizer(request.prompt, return_tensors="pt").to("cuda")
    try:
        output = model.generate(**inputs, max_length=512)
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "DeepSeek-R1-Distill-Qwen-32B API is running!"}

if __name__ == "__main__":
    # Dynamically get the port Render assigned
    port = int(os.getenv("PORT", 10000))  # PORT is set by Render during deployment
    uvicorn.run(app, host="0.0.0.0", port=port)  # Bind to 0.0.0.0 to accept external traffic
