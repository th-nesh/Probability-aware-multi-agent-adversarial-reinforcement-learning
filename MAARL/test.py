import numpy as np
def risk_calculation(action_index):
    # action_list= [-8,-4,0,4,8]
    # real_probability = [0.03,0.15,0.39,0.42,0.01]
    action_list= [-4,-2,0,2,4]
    real_probability = [0.055,0.098,0.306,0.225,0.183]
    real_probability = np.array(real_probability)
    final_probability = real_probability[np.argwhere(np.array(action_list)==action_index)]

    return final_probability[0,0]


print(risk_calculation(2))