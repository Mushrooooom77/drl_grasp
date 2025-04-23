# FR5_RL_Grasping

本项目为ME5406的Project2，基于 PyBullet 仿真平台，使用深度强化学习方法实现带夹爪自由度的碰撞检测抓取任务。项目实现的学习算法为PPO、SAC、A2C、TD3，支持多进程加速训练，训练环境为一个带有障碍物与目标物的桌面抓取场景。
该项目基于https://github.com/WangZY233/FR5_Reinforcement-learning 的强化学习抓取训练代码，并在其基础上添加了障碍物和抓取动作。

## Hardware
- 操作系统镜像：PyTorch 2.0.0 + Python 3.8（Ubuntu 20.04），支持 CUDA 11.8
- GPU：NVIDIA RTX 4090（24GB 显存）×1
- CPU：16 核 vCPU，型号为 Intel(R) Xeon(R) Platinum 8352V，主频 2.10GHz

## Scene Deployment Instructions
本项目采用仿真平台 PyBullet 搭建训练环境，控制对象为一台六轴工业机械臂，具备较高的控制精度和稳定性。我们在其末端安装了二指夹爪
在仿真中，工作区域为前方 40cm × 40cm × 20cm 的空间，抓取目标为一个圆柱形物体，该物体随机出现在该区域内。夹爪需在不发生非预期碰撞（例如触碰障碍物或平台）情况下完成抓取动作。我们定义抓取成功的标准为：夹爪对目标物体夹紧后，物体被成功提起至高度大于 40cm。相反，若出现以下任一情况将被判定为失败：
夹爪在接触前误碰目标物体或桌面；抓取过程中发生与障碍物的碰撞；总步数超过设定上限（例如 100 步）。
实验中，每次重置环境时，目标物体与场景中两个红色立方体障碍物的位置都会进行随机初始化，确保策略具备较强的泛化能力。在该场景中，智能体通过状态输入学习夹爪控制策略，实现对目标的精准抓取。

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
