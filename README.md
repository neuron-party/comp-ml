# computational machine learning lab

## how to run
`eval_failure_sta.py`: same agent failure rate evaluation for STA agents <br>
`eval_failure.py`: same agent failure rate evaluation for regular agents <br>
`eval_relay_sta.py`: relay failure rate evaluation for STA agents <br>
`eval_relay.py`: relay failure rate evaluation for regular agents <br>
`train_procgen.py`: training script for regular PPO agents on procgen environments <br>
`train_sta.py`: training script for STA PPO agents on procgen environments 

**example**
```
python train_procgen.py --env-name=starpilot --num-envs=64 --log=True --project-name=ppo_starpilot --instance-name=agent01 --save-path=weights/ppo_starpilot_1 --device=7
```

### argument-parsers and parameters
**train_procgen and train_sta**
```
# agent parameters (defaults):
gamma: 0.999 (discount rate)
lr: 5e-4 (learning rate for optimizer)
epsilon: 0.2 (parameter for probability ratio clipping)
vf-clip: 0.2 (value function clipping for loss function)
device: 0 (cuda device)
num-steps: 256 (how many steps to play through before gradient updates)
num-epochs: 3 (number of epochs for optimization)
num-batches: 8 (number of batches to use per epoch)
lambd: 0.95 (lambda parameter for calculating advantages)
c1: 0.5 (scaling parameter for value function loss)
c2: 0.01 (scaling parameter for entropy loss)
max-grad-norm: 0.5 (gradient clipping)
norm-adv: True (normalize advantages)

# environment args:
env-name: environment name i.e starpilot, jumper, heist, ...
num-envs: number of environments for vectorization
num-levels: procgen argument, number of levels to use for generation
start-level: procgen argument, starting level for generation
distribution-mode: procgen argument, difficulty level
max-global-steps: how many steps to train for 

# logging args:
log: True/False whether to track the training or not. ensure that you are signed into wandb in the command line if True
project-name: name of the project in wandb
instance-name: instance name of your run (goes into your project)
save-path: save path for model weights. do not need to add .pth
```
**eval_failure and eval_relay, regular and sta**
```
procgen arguments

save-path: directory to save the output to. output is a list of lists, where each inner list contains 2000 failures/successes

# regular agents
weights-path: directory to load the model weights from. should be of form ppo_jumper_ and assumes there are multiple

# sta agents
ra-weights-path: directory to load the regular agent model weights from. should be of form ppo_jumper_ and assumes there are multiple
sta-weights-path: directory to load the sta agent model weights from.

i should probably remove some of the hardcodes in the scripts
```
**note:** currently, eval_failure_sta and eval_relay_sta only work for single sta agents (does not iterate through multiple). need to update these scripts


## current work
* training 4 regular and 4 STA PPO agents for starpilot to see if the STA algorithm has more of an effect on continuous environments
* adapting STA implementation details and parameters for procgen 
    * STA1: upper bound on the best threshold so that states can be added to the buffer throughout the entire training process
    * STA2: all 50 states preceding a successful trajectory should be controllable, not just the 50th one from the completion
    * STA3: no upper bound on the best threshold so that only states closer and closer to the completion step get added into the buffer throughout training
    * STAC: no upper bound on the best threshold but for continuous environments like starpilot (iterative rewarding process)
    * need to find a parameter for the sampling ratio
    * changed the control statement if average_return > 1 to if average_return > 0 since reward is much lower for procgen compared to mujoco
    * ram memory is a current issue as the states are large numpy arrays and the STA set grows considerably large

## to do
* 
* evaluate same agent and relay failure rates for PPO leaper and PPO ninja agents
* evaluate returns for PPO leaper and PPO ninja agents
* reevaluate the same agent and relay failure rates for PPO heist agents using redefined success
* reevaluate returns for PPO heist agents
* evaluate same agent and relay failures rates for regular PPO and STA PPO starpilot agents
* evaluate returns for regular PPO and STA PPO starpilot agents
* soft actor critic?


## finished
* 4 regular agents for PPO jumper, PPO heist, PPO ninja, PPO leaper
* same agent and relay evaluations for regular PPO jumper agents
* returns for regular PPO jumper agents