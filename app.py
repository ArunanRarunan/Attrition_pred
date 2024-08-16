from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

model = joblib.load("KNN.joblib")

class PredictionInput(BaseModel):
    CAMP:str
    GENDER:str
    RELIGION:str
    MARITAL_STATUS:str
    HOSTEL:str
    PREVIOUS_EXPERIENCE:str
    NOF_YEARS:str
    EXPERIENCE_FIELD:str
    BASIC_EARNED:int
    GRADE:str
    BRANCH:str
    SECTION:str
    DESIGNATION:str
    CURRENT_BASIC:int
    AGE:int

@app.post("/predict")
def predict(data:PredictionInput):
    try:
        input_df = pd.DataFrame([data.dict()])
        input_df["NOF_YEARS"] = input_df["NOF_YEARS"].replace("-","0").astype(float)
        input_df["EXPERIENCE_FIELD"]=input_df["EXPERIENCE_FIELD"].replace("-","NO_EXPERIENCE")

        label_encoders = {}
        for col in ["CAMP", "GENDER", "RELIGION", "MARITAL_STATUS", "HOSTEL", "PREVIOUS_EXPERIENCE", "EXPERIENCE_FIELD", "GRADE","BRANCH","SECTION","DESIGNATION"]:
            if input_df[col].dtype == object:
                le = LabelEncoder()
                le.fit(input_df[col])  # Fit on the input column
                input_df[col] = le.transform(input_df[col])  # Transform the input column
                label_encoders[col] = le  # Save encoder for future use (optional)

        input_data = input_df.values

        prediction = model.predict(input_data)
        prediction_label = "MIGHT LEAVE" if prediction[0] == 1 else "WILL WITHSTAND"

        return {"prediction": prediction_label}
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/")
def read_root():
       # Serve a custom HTML file as the root page
       return FileResponse("static/base.html")

app.mount("/static", StaticFiles(directory="static"), name="static")