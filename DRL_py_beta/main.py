"""
    Main file to execute PPO algorithm
"""

import torch.optim as optim
from ppo import *
from network import *

# tolerance for std
EPS = 1e-6

# discount function
gamma = 0.99
# TD_lambda method factor
lambda_ = 0.97
# no of patches at the surface of cylinder
n_sensor = 400

# coefficients for reward function
r_1 = 3
r_2 = 0.1
r_3 = 0
r_4 = 0

# max and min actions
action_bounds = [-10,10]

# policy model and value model instances
policy_model = FCCA(n_sensor, 64, action_bounds)
value_model = FCV(n_sensor, 64)


# no of workers
n_worker = 12
# no of total buffer size
buffer_size = 10
# range to randomly start control
control_between = [2.5, 4.0]
# env instance
env = env(n_worker, buffer_size, control_between)

# learning rate for policy model and value model
policy_lr = 0.0015
value_lr = 0.00075

# policy optimizer
policy_optimizer = optim.Adam(policy_model.parameters(), policy_lr)
# no of epochs for value model
policy_optimization_epochs = 80
# ration for no of trajectory to take for training (1 = 100%)
policy_sample_ratio = 1
# clipping parameter of policy loss
policy_clip_range = 0.1
# maximum norm tolerance of policy optimization
policy_model_max_grad_norm = float('inf')
# tolerance for training of policy net
policy_stopping_kl = 0.2
# factor for entropy loss
entropy_loss_weight = 0.01

# value optimizer
value_optimizer = optim.Adam(value_model.parameters(), policy_lr)
# no of epochs for value model
value_optimization_epochs = 80
# ration for no of trajectory to take for training (1 = 100%)
value_sample_ratio = 1
# clipping parameter of value model loss
value_clip_range = float('inf')
# maximum norm tolerance of value optimization
value_model_max_grad_norm = float('inf')
# tolerance for trainig of value net
value_stopping_mse = 25

# to retrain model from checkpoint
retrain_models = False
begin_from_sample = 0

if retrain_models:
    begin_from_sample = 16
    path = "results/models/"
    sample_str = f"{begin_from_sample-1}"
    policy_model = torch.jit.load(path+"policy_"+sample_str+".pt")
    policy_model.eval()
    print(f"policy model model loaded from {path}")
    path = "results/value_models/"
    value_model.load_state_dict(torch.load(path+"value_"+sample_str+".pt"))
    print(f"value model model loaded from {path}")
    value_model.eval()

# main PPO algorithm iteration
main_ppo_iteration = 100

evaluation_score = []

# iteration for PPO algorithm
for i in range(main_ppo_iteration - begin_from_sample):
    sample = i + begin_from_sample
    train_model(value_model,
                policy_model,
                env,
                policy_optimizer,
                policy_optimization_epochs,
                policy_sample_ratio,
                policy_clip_range,
                policy_model_max_grad_norm,
                policy_stopping_kl,
                entropy_loss_weight,
                value_optimization_epochs,
                value_optimizer,
                value_sample_ratio,
                value_clip_range,
                value_model_max_grad_norm,
                value_stopping_mse,
                gamma,
                lambda_,
                r_1,
                r_2,
                r_3,
                r_4,
                sample,
                n_sensor,
                EPS,
                evaluation_score,
                action_bounds)
    print(f'The Iteration {sample} is completed \n \n')
