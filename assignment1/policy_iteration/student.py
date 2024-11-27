import numpy as np

def reward_function(state, env_size):
    r = 0.0
    if (state == np.array([env_size-1, env_size-1])).all():
        r = 1.0
    return r

def transition_probabilities(env, s, a, env_size, directions, holes):
   
    prob_next_state = np.zeros((env_size, env_size))
    
    correct_direction = directions[a]
    if(a == 0 ):
        wrong_action = 3
    else:
        wrong_action = (a - 1)
    wrong_direction = directions[wrong_action]
    
    def is_valid_position(s):
        if not (0 <= s[0] < env_size and 0 <= s[1] < env_size):
            return False
        if holes[s[0], s[1]] == 1:
            return False
        return True
    
    correct_next_state = s + correct_direction
    wrong_next_state = s + wrong_direction
    
    if is_valid_position(correct_next_state):
        prob_next_state[correct_next_state[0], correct_next_state[1]] = 0.5
    else:
        prob_next_state[s[0], s[1]] = 0.5
    
    if is_valid_position(wrong_next_state):
        prob_next_state[wrong_next_state[0], wrong_next_state[1]] = 0.5
    else:
        prob_next_state[s[0], s[1]] = 0.5
    
    return prob_next_state