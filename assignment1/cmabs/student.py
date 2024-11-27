import numpy as np

def explore_and_commit(env, explore_steps = 50, iters = 200):
    clicks = np.zeros((env.n_states, env.n_actions))
    views = np.zeros((env.n_states, env.n_actions))
    Q = np.zeros((env.n_states, env.n_actions))
    Qs = []
    total_reward = 0.
    regret = 0.

    # Explore phase
    for i in range(explore_steps):
        state = env.observe()
        action = np.random.randint(env.n_actions)
        click = env.step(action)
        
        views[state, action] += 1
        clicks[state, action] += click
        
        Q[state, action] = clicks[state, action] / views[state, action]
        total_reward += click
        best_action = env.CTR[state, :].argmax()
        regret += env.CTR[state,best_action] - click
        Qs.append(Q.copy())

    # Commit phass 
    for i in range(iters-explore_steps):
        state = env.observe()
        action = Q[state, :].argmax()
        click = env.step(action)
        
        total_reward += click
        best_action = env.CTR[state, :].argmax()
        regret += env.CTR[state,best_action] - click
    return Qs, total_reward, regret

def epsilon_greedy(env, epsilon = 0.1, null_epsilon_after = 50, iters = 200):
    clicks = np.zeros((env.n_states, env.n_actions))
    views = np.zeros((env.n_states, env.n_actions))
    Q = np.zeros((env.n_states, env.n_actions))
    Qs = []
    total_reward = 0.
    regret = 0.

    # Explore with epsilon > 0
    for i in range(null_epsilon_after):
        state = env.observe()
        
        if np.random.random() < (1-epsilon):
            action = np.random.randint(env.n_actions)
        else:
            action = np.argmax(Q[state, :])
            
        click = env.step(action)
        
        views[state, action] += 1
        clicks[state, action] += click
        Q[state, action] = clicks[state, action] / views[state, action]
        
        total_reward = click
        best_action = env.CTR[state, :].argmax()
        regret += env.CTR[state,best_action] - click
        
        Qs.append(Q.copy())

    # Commit (epsilon = 0)
    for i in range(iters-null_epsilon_after):
        state = env.observe()
        action = np.argmax(Q[state, :])
        click = env.step(action)
        #print("best_action", action)
        total_reward += click
        best_action = env.CTR[state, :].argmax()
        regret += env.CTR[state,best_action] - click

    return Qs, total_reward, regret