"""
Project: LR Scheduler Visualizations
File: visualizer.py
Authors: Drew Meyer
Create On: 7/15/2024
Description: Takes an LR config as input and creates LR schedulers visualizations
"""

import json
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
import math

def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

def create_scheduler(scheduler_config):
    name = scheduler_config['name']
    params = scheduler_config['params']
    initial_lr = scheduler_config['initial_lr']
    num_epochs = scheduler_config['num_epochs']

    dummy_model = torch.nn.Linear(1, 1)
    optimizer = optim.SGD(dummy_model.parameters(), lr=initial_lr)

    if name == 'StepLR':
        return lr_scheduler.StepLR(optimizer, **params)
    elif name == 'ExponentialLR':
        return lr_scheduler.ExponentialLR(optimizer, **params)

def generate_lr_schedule(scheduler_config):
    name = scheduler_config['name']
    params = scheduler_config.get('params', {})
    initial_lr = scheduler_config['initial_lr']
    num_epochs = scheduler_config['num_epochs']

    lr_schedule = [initial_lr]

    if name == "LambdaLR":
        lambda_func = lambda epoch: 0.75 ** epoch
        for epoch in range(1, num_epochs):
            lr_schedule.append(initial_lr * lambda_func(epoch))

    elif name == "LinearLR":
        start_factor = params.get('start_factor', 0.5)
        total_iters = params.get('total_iters', 5)
        for epoch in range(1, num_epochs):
            if epoch < total_iters:
                factor = start_factor + (1 - start_factor) * (epoch / total_iters)
            else:
                factor = 1.0
            lr_schedule.append(initial_lr * factor)

    elif name == "ConstantLR":
        factor = params.get('factor', 0.5)
        total_iters = params.get('total_iters', 20)
        for epoch in range(1, num_epochs):
            if epoch < total_iters:
                lr_schedule.append(initial_lr * factor)
            else:
                lr_schedule.append(initial_lr)

    elif name == "MultiplicativeLR":
        lambda_func = lambda epoch: 0.95
        for epoch in range(1, num_epochs):
            lr_schedule.append(lr_schedule[-1] * lambda_func(epoch))

    elif name == "PolynomialLR":
        total_iters = params.get('total_iters', 5)
        power = params.get('power', 2.0)
        for epoch in range(1, num_epochs):
            if epoch < total_iters:
                factor = (1 - epoch / total_iters) ** power
            else:
                factor = 0
            lr_schedule.append(initial_lr * factor)

    elif name == "ExponentialLR":
        gamma = params.get('gamma', 0.80)
        for epoch in range(1, num_epochs):
            lr_schedule.append(lr_schedule[-1] * gamma)

    elif name == "StepLR":
        step_size = params.get('step_size', 5)
        gamma = params.get('gamma', 0.1)
        for epoch in range(1, num_epochs):
            if epoch % step_size == 0:
                lr_schedule.append(lr_schedule[-1] * gamma)
            else:
                lr_schedule.append(lr_schedule[-1])

    elif name == "MultiStepLR":
        milestones = params.get('milestones', [5, 10])
        gamma = params.get('gamma', 0.1)
        for epoch in range(1, num_epochs):
            if epoch in milestones:
                lr_schedule.append(lr_schedule[-1] * gamma)
            else:
                lr_schedule.append(lr_schedule[-1])

    elif name == "ChainedScheduler":
        phase1_iters = 5
        phase2_iters = 15
        for epoch in range(1, num_epochs):
            if epoch <= phase1_iters:
                factor = 0.5 + (1 - 0.5) * (epoch / phase1_iters)
                lr_schedule.append(initial_lr * factor)
            elif epoch <= phase1_iters + phase2_iters:
                lr_schedule.append(lr_schedule[-1])
            else:
                lr_schedule.append(lr_schedule[-1] * 0.9)

    elif name == "OneCycleLR":
        max_lr = params.get('max_lr', 0.1)
        total_steps = params.get('total_steps', 50)
        for step in range(1, num_epochs):
            cycle = math.floor(1 + step / (total_steps / 2))
            x = abs(step / (total_steps / 2) - 2 * cycle + 1)
            lr = initial_lr + (max_lr - initial_lr) * max(0, (1 - x))
            lr_schedule.append(lr)

    elif name == "CyclicLR":
        base_lr = params.get('base_lr', 0.01)
        max_lr = params.get('max_lr', 0.1)
        step_size_up = params.get('step_size_up', 2.5)
        for step in range(1, num_epochs):
            cycle = math.floor(1 + step / (2 * step_size_up))
            x = abs(step / step_size_up - 2 * cycle + 1)
            lr = base_lr + (max_lr - base_lr) * max(0, (1 - x))
            lr_schedule.append(lr)

    elif name == "CosineAnnealingLR":
        T_max = params.get('T_max', 5)
        eta_min = params.get('eta_min', 0)
        for epoch in range(1, num_epochs):
            lr = eta_min + (initial_lr - eta_min) * (1 + math.cos(math.pi * epoch / T_max)) / 2
            lr_schedule.append(lr)

    elif name == "CosineAnnealingWarmRestarts":
        T_0 = params.get('T_0', 10)
        for epoch in range(1, num_epochs):
            lr = initial_lr * (1 + math.cos(math.pi * (epoch % T_0) / T_0)) / 2
            lr_schedule.append(lr)

    else:
        print(f"Warning: Unknown scheduler type '{name}'. Using constant learning rate.")
        lr_schedule = [initial_lr] * num_epochs

    if len(lr_schedule) != num_epochs:
        print(f"Warning: LR schedule length ({len(lr_schedule)}) does not match num_epochs ({num_epochs}) for {name}")
        lr_schedule = lr_schedule[:num_epochs] if len(lr_schedule) > num_epochs else lr_schedule + [lr_schedule[-1]] * (num_epochs - len(lr_schedule))

    return lr_schedule

def plot_lr_schedule(name, lr_schedule, num_epochs):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), lr_schedule)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title(f'Learning Rate Schedule - {name}')
    plt.savefig(f'../visualizations/learning_rate_{name}.png')
    plt.close()

def main(config_path):
    config = load_config(config_path)
    
    for scheduler_config in config['schedulers']:
        name = scheduler_config['name']
        num_epochs = scheduler_config['num_epochs']
        
        lr_schedule = generate_lr_schedule(scheduler_config)
        
        if len(lr_schedule) != num_epochs:
            print(f"Warning: LR schedule length ({len(lr_schedule)}) does not match num_epochs ({num_epochs}) for {name}")
            continue
        
        plot_lr_schedule(name, lr_schedule, num_epochs)
        
        print(f"Generated visualization for {name}")

if __name__ == "__main__":
    main('config.json')