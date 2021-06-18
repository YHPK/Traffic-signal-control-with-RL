# Traffic signal control with RL

This work based on [**Reinforcement learning-based multi-agent
system for network traffic signal control**](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.232.9789&rep=rep1&type=pdf)

## Training Code



## Environment Setting
### Five-Intersection Traffic Network
<img src="./fig/Intersection.png" width="400" height="400">

#### centrally connected vehicular traffic network

- outbound intersection: Operate based on local information.
- inbound intersection: RL-based agent controls traffic signaling.

#### vehicles generation

- during each simulation time step, new vehicles are generated, as governed by a 
  **poisson process**, outside each outbound intersection.
  
- They are placed at the end of the queue of their respective destination lanes.
- no vehicles are generated at the central intersection.

#### intersection config

- each intersection have 8 lanes.
- the capacity of each lane is 40.
- the traffic lights change once every 20 time steps.


## RL Setting
### State
#### relative traffic flow

<img src="https://render.githubusercontent.com/render/math?math=\text{relative traffic flow} = \frac{\text{the total delay of vehicles in a lane}}{\text{the average delay at all lanes in the intersection}}">

### Action
<img src="./fig/Actions.png">

### Reward

## Result


