# FR5_RL_Grasping

This project is Project 2 of ME5406, built on the PyBullet simulation platform. It applies deep reinforcement learning methods to perform a collision-aware grasping task using a gripper with degrees of freedom. The implemented learning algorithm is PPO, and the framework supports multi-process accelerated training. The training environment is a tabletop grasping scene with both obstacles and target objects.
This project is based on the reinforcement learning grasping training code from [https://github.com/WangZY233/FR5_Reinforcement-learning](https://github.com/WangZY233/FR5_Reinforcement-learning), with additional implementations of obstacles and grasping actions.

## Hardware
- OS Image: PyTorch 2.0.0 + Python 3.8 (Ubuntu 20.04), with CUDA 11.8 support  
- GPU: NVIDIA RTX 4090 (24GB VRAM) ×1  
- CPU: 16-core vCPU, model Intel(R) Xeon(R) Platinum 8352V, 2.10GHz

## Scene Deployment Instructions
This project builds the training environment using the PyBullet simulation platform. The controlled object is a six-axis industrial robotic arm, which offers high precision and stability. A two-finger gripper is mounted at the end of the arm.

In the simulation, the workspace is a 40cm × 40cm × 20cm area in front of the robot. The grasping target is a cylindrical object that appears randomly within this area. The gripper must complete the grasping action without unintended collisions (e.g., touching obstacles or the platform). A grasp is considered successful if the gripper firmly holds the target and lifts it to a height greater than 40cm. Conversely, a failure is determined if any of the following occurs:
- The gripper unintentionally touches the target object or the table before making a controlled contact;
- A collision with an obstacle occurs during the grasping process;
- The total number of steps exceeds the set limit (e.g., 100 steps).

In each episode, the environment is reset with random positions for the target object and two red cube obstacles, ensuring the policy has strong generalization capabilities. In this scenario, the agent learns the control strategy for the gripper through state inputs to achieve accurate and reliable grasps.

## Main code
- fr5_description: Stores the URDF model files of the robotic arm.
- Fr5_env.py: Constructs the reinforcement learning environment.
- Fr5_train.py: Contains the reinforcement learning training code.
- FR5_test.py: Used for reinforcement learning testing.


## How to use
```bash
pip install -r requirments.txt
```
### Train
```bash
python -m FR_Gym.Fr5_train --timesteps 30000 --gui False
```
### Test
```bash
python test.py --model_path ./models/PPO-run-episode9.zip
```
