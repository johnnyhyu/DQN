# **DQN: CartPole-v1**

This repository contains a PyTorch implementation of a Deep Q-Network (DQN) that achieves an average score of 193 on the CartPole-v1 environment.

## **Performance**
Score: 193

Environment: CartPole-v1

Framework: PyTorch

## **Model Architecture**

Input: 5 (State observations + termination flag)

Hidden Layer 1: 64 units (ReLU)

Hidden Layer 2: 32 units (ReLU)

Output: 1 (Mapped to actions 0 and 1)

## **Hyperparameters**

Batch Size: 32

Replay Buffer: 1,000 transitions

Epsilon: 0.1 (Fixed exploration)

Gamma: 0.99

Target Update: Every 100 timesteps

Optimizer: Adam (Default learning rate)

## **Key Details**

The model includes the termination status as a 5th input feature, which helps the agent identify terminal states more effectively. It uses a standard experience replay buffer to sample batches of 32 transitions, stabilizing the training compared to online updates. The target network is synchronized every 100 steps to ensure consistent Q-value targets during the optimization process.

## **Usage**

Install Gymnasium and PyTorch.

Run the training script to observe the agent converge toward the 193+ score.
