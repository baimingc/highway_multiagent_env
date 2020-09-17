# highway-multiagent-env

Multi-agent version of highway_env/intersection at https://github.com/eleurent/highway-env


<img src="misc/multiagent_intersection.gif?raw=true" width="33%"> <img src="misc/multiagent_intersection1.gif?raw=true" width="33%"> 

## Installation

cd into this directory then `pip install -e .`

## Usage

```python
import gym
import highway_env

env = gym.make("intersection-multiagent-v0")

done = False
while not done:
    action = ... # Your agent code here
    obs, reward, done, info = env.step(action)
    env.render()
```
