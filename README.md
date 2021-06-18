# Traffic signal control with RL

This work based on [**Reinforcement learning-based multi-agent
system for network traffic signal control**](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.232.9789&rep=rep1&type=pdf)

## Training Code
### Environment prepare
this code verified on python=3.6 and cuda=10.2 setting.  

```bash
conda env create -f environment.yml
```
or
```bash
pip install -r requirements.txt
```

### Run
```bash
python main.py <optins>
```
#### options
| options                  | type    | help                                                                         |
|--------------------------|:-------:|------------------------------------------------------------------------------|
| `--episode`              | `int`   | number of maximum episode                                                    |
| `--max_step`             | `int`   | number of maximum step in single episode                                     |
| `--batch_size`           | `int`   | [for dqn] batch size                                                         |
| `--gpu_num`              | `int`   | GPU number to use                                                            |
| `--lr`                   | `float` | learning rate                                                                |
| `--gamma`                | `float` | discount factor                                                              |
| `--epsilon`              | `float` | [for dqn] e-greedy parameter                                                 |
| `--traffic_change_time`  | `int`   | the traffic light changes every `n` steps.                                   |
| `--average_arrival_rate` | `int`   | vehicle generation ratio                                                     |
| `--algo_type`            | `str`   | The type of algorithm used by the agent.  choice from `dqn`, `a2c` and `lqf` |
| `--render_type`          | `str`   | visualize the state of intersection. choice from `print` and `reder`         |
| `--log_save_path`        | `str`   | path to save log                                                             |


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

- the state is represented by an eight-dimensional feature vector 
  with each element representing the **relative traffic flow** at one of the lanes.

- the state space size of five-intersection environment is 40(# of lanes in intersections * # of intersections)

#### relative traffic flow

<img src="https://render.githubusercontent.com/render/math?math=\text{relative traffic flow} = \frac{\text{the total delay of vehicles in a lane}}{\text{the average delay at all lanes in the intersection}}">



### Action
<img src="./fig/Actions.png">

- the maximal number of applicable, compatible and non-conflicting phase combinations is 
  eight for each isolated intersection presented **{(1,5), (1,6), (2,5), (2,6), (3,7), (3,8), (4,7), (4,8)}**.

- the action space size is 8


### Reward
<img src="https://render.githubusercontent.com/render/math?math=r = D_{last} - D_{currnet}">

## Result


