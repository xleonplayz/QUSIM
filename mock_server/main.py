# -*- coding: utf-8 -*-
"""
Mock Server for NV Center API. 
Start the server with:
    >> uvicorn main:app --reload --ws-max-size 1073741824
Note: We need ws-max-size >1MB to transfer waveform data

After starting the server you can start qudi with: 
    >> qudi

For installation of qudi follow 
    https://qudi-core-testing.readthedocs.io/en/george/setup/installation.html

A short guide on how to setup this mock server with qudi: 
First we need to install the qudi-core package.
    >> python3.10 -m venv qudi-env
    >> source qudi-env/bin/activate
    >> python -m pip install qudi-core
    >> qudi- install-kernel

Then we need to install additional dependencies:
    >> python -m pip install numpy fastapi uvicorn asyncio websockets requests

Now we can install the qudi iqo modules: 
Note: We install it here in developer mode, so we can edit the modules
      This is not required, but helpful for development.
    >> git clone https://github.com/Ulm-IQO/qudi-iqo-modules.git
    >> cd qudi-iqo-modules
    >> python -m pip install -e .
    
Finally we install the mock server:
    >> git clone https://github.com/xleonplayz/QUSIM.git
    >> cd QUSIM/qudi-addon
    >> python -m pip install -e .
"""


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
