from fastapi import FastAPI
from hardware.laser_simulator import router as laser_router


app = FastAPI()
app.include_router(laser_router)

@app.get("/")
def root():
    return {"Hello": "World"}
