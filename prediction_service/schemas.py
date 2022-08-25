from pydantic import BaseModel

class PredRequest(BaseModel):
    text: str

class PredResponse(BaseModel):
    toxic: bool
    probale: float
    class Config:
        orm_mode = True