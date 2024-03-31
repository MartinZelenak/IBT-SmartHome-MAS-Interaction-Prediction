# SmartHome-Inhabitant-Simulation
run simulation (from outside the project folder):  
<code>python3 -m \<folder name>.Scenario</code>  
Simulation parameters can be adjusted via constants in Scenario.py

## Files  
utils.py: Helper functions  
inhabitantModel.py: Inhabitant base class  
environment.py: Environment with Timeslots, Home model and event system (derived from simpy.environment)  
homeModel.py: classes for modeling a smart home (rooms with devices)  
deviceModels.py: classes modeling smart devices
event.py: event system (publish, subscribe)  
stateLogger.py: Logs the environment state and inhabitants location and actions in given time intervals (logs used for machine learning)  
Scenario.py: Simulation experiment. Inhabitant with concrete behavior (derived from inhabitantModel.Inhabitant)  
