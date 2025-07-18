# backend/main.py
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from cat_model import predict_cat_breed
from PIL import Image
import io

app = FastAPI()

# Enable CORS so frontend can access this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with frontend URL in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Cat Breed Predictor API is running"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    breed = predict_cat_breed(contents)
    return {"breed": breed}
