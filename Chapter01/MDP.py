# 马尔科夫决策过程(MDP) - 对Q学习适配的贝尔曼方程
# 通过Q动作值（回报）函数的强化学习
# Copyright 2018 Denis Rothman MIT License. See LICENSE.
import numpy as ql

# R是每个状态的回报矩阵（Reward Matrix）
R = ql.matrix([[0, 0, 0, 0, 1, 0],
               [0, 0, 0, 1, 0, 1],
               [0, 0, 100, 1, 0, 0],
               [0, 1, 1, 0, 1, 0],
               [1, 0, 0, 1, 0, 0],
               [0, 1, 0, 0, 0, 0]])

# Q是学习矩阵，其中每个回报会被学习/存储
Q = ql.matrix(ql.zeros([6, 6]))

# Gamma : 它是一种惩罚形式或学习的不确定性
# 如果值是1，回报会过高
# 通过这种方式系统知道它正在学习
gamma = 0.8

# agent_s_state. 系统计算的代理名称
# s是代理的起始状态，s'是它要到达的状态
# 这一状态可以随机的或者是选定的，只要未确定剩余选项即可
# 随机性是随机过程的一部分
agent_s_state = 1


# 在代理处于给定状态时可能的动作"a"
def possible_actions(state):
    current_state_row = R[state,]
    possible_act = ql.where(current_state_row > 0)[1]
    return possible_act


# 获取当前状态的可用动作
PossibleAction = possible_actions(agent_s_state)


# 该函数随机选择可用动作范围内可以执行的动作
def ActionChoice(available_actions_range):
    if (sum(PossibleAction) > 0):
        next_action = int(ql.random.choice(PossibleAction, 1))
    if (sum(PossibleAction) <= 0):
        next_action = int(ql.random.choice(5, 1))
    return next_action


# 对要执行的下一个动作取样
action = ActionChoice(PossibleAction)


# 使用Q函数强化学习的贝尔曼方程的一个版本
# 该强化学习算法是一个无记忆的过程
# 从一个状态到另一个状态的转换函数T不在以下方程中。T由以上的随机选择完成

def reward(current_state, action, gamma):
    Max_State = ql.where(Q[action,] == ql.max(Q[action,]))[1]

    if Max_State.shape[0] > 1:
        Max_State = int(ql.random.choice(Max_State, size=1))
    else:
        Max_State = int(Max_State)
    MaxValue = Q[action, Max_State]

    # 基于MDP的贝尔曼Q函数
    Q[current_state, action] = R[current_state, action] + gamma * MaxValue


# 回报Q矩阵
reward(agent_s_state, action, gamma)

# 根据系统收敛n次迭代学习
# 通过对比上一部分的Q矩阵的总和Q矩阵n-1，收敛函数可以替代该过程的系统化重复
for i in range(50000):
    current_state = ql.random.randint(0, int(Q.shape[0]))
    PossibleAction = possible_actions(current_state)
    action = ActionChoice(PossibleAction)
    reward(current_state, action, gamma)

# 在对Q阶段归一化前显示Q
print("Q  :")
print(Q)

# Q的范数
print("Normed Q :")
print(Q / ql.max(Q) * 100)