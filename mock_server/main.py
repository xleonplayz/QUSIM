from fastapi import FastAPI
from hardware.laser_simulator import router as laser_router
from hardware.fastcounter_simulator import router as fastcounter_router
from hardware.pulser_simulator import router as pulser_router


app = FastAPI()
app.include_router(laser_router)
app.include_router(fastcounter_router)
app.include_router(pulser_router)

@app.get("/")
def root():
    return {"Hello": "World"}
