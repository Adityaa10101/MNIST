import torch
from fastapi import FastAPI
from mnist_cnn import CNN
from pydantic import BaseModel

app=FastAPI()
model=CNN()
checkpoint =torch.load("./model/chkpoint.pth")
model.load_state_dict(checkpoint["model_state"])
model.eval()

@app.get("/")
def read_root():
    return {"message":"hello"}

class PredictionRequest(BaseModel):
    pixel_values:list[float]

@app.post("/predict")
def predict(request: PredictionRequest):
    input_tensor=torch.tensor(request.pixel_values)
    input_tensor=input_tensor.view(1,1,28,28)
    with torch.no_grad():
        output=model(input_tensor)
        probabilities=torch.softmax(output,dim=1)
        confidence,predicted=torch.max(probabilities,dim=1)
        predicted_class = predicted.item()
        confidence_val=confidence.item()
        all_probs = probabilities[0].tolist()
        return {"Predicted_Class": predicted_class,
                "Prob":confidence_val,
                "All_Probs": all_probs
            }