import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from env import Environment
from student import *


def main():

    CTR_dict = {
        'AI Engineer':{
            'RL Course by RC': 0.3,
            'Optimization course at Amazon': 0.2,
            'Trading Course by TikTok Guru': 0.03,
        },
        'Management Engineer':{
            'RL Course by RC': 0.08,
            'Optimization course at Amazon': 0.3,
            'Trading Course by TikTok Guru': 0.05,
        },
        'High School Kid':{
            'RL Course by RC': 0.05,
            'Optimization course at Amazon': 0.1,
            'Trading Course by TikTok Guru': 0.1,
        }
    }

    CTR_df = pd.DataFrame(CTR_dict).T

    CTR = CTR_df.to_numpy()

    env = Environment(CTR)

    methods = [explore_and_commit, epsilon_greedy]

    logs = []
    avg_rewards = []
    regrets = []
    for method in methods:
        log = []
        Qs, total_reward, regret = method(env, iters=200)
        avg_rewards.append(total_reward/200)
        regrets.append(regret)
        logs.append(Qs)

    for avg_reward, method, regret in zip(avg_rewards, methods, regrets):
        print(method.__name__)
        print('Reward:', avg_reward)
        print('Regret:', regret)
        print()

    plt.figure(figsize=(20,10))
    i = 0
    for s in range(env.n_states):
        for a in range(env.n_actions):
            i=i+1
            ax = plt.subplot(env.n_states, env.n_actions, i)
            ax.set_title(f"s='{CTR_df.index[s]}', a='{CTR_df.columns[a]}'")
            for j, log in enumerate(logs):
                ax.plot([l[s,a] for l in log], label=methods[j].__name__)
            ax.plot([CTR[s,a]]*len(logs[0]), label='True CTR')

            ax.legend()
    plt.show()


if __name__ == '__main__':
    main()


