import os
import gdown
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from model_utils import generate_answer

app = FastAPI()

# Add CORS middleware to allow cross-origin requests (for frontend communication)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow requests from all origins (frontend)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Path to save your model
MODEL_PATH = "./my_finetuned_model/model.safetensors"

# Google Drive File ID (replace with your actual Google Drive File ID)
GOOGLE_DRIVE_FILE_ID = "1uwKZIQ-Rw6pyr4M8p6EJvuXufcb7TqTw"

# Function to download the model if it doesn't already exist
def download_model():
    if not os.path.exists(MODEL_PATH):
        print(f"Downloading model from Google Drive...")
        url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)
        print("Model downloaded successfully.")

# Download the model when the app starts
download_model()

# Define your Pydantic model for handling the request
class Query(BaseModel):
    question: str

# Define the endpoint that queries the model
@app.post("/query")
async def query_model(query: Query):
    # Call the generate_answer function to get the response from the model
    answer = generate_answer(query.question)
    return {"answer": answer}
