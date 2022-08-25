from fastapi import FastAPI
from prediction_service import predict, schemas
app = FastAPI()

@app.post("/")
def post_preds(request_dict: schemas.PredRequest, response_model=schemas.PredResponse):
    toxic, pred_proba = predict.api_response(request_dict)
    if not toxic:
        pred_proba = 1 - pred_proba

    response = {"toxic": toxic, "probable": pred_proba}
    return response