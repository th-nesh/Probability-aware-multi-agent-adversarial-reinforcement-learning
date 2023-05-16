import numpy as np 


def real_to_pixel(action_index):

    x_max = 200
    y_max = 200
    x_room = 400
    y_room= 400
    pixel_value = (x_room*y_room)/(x_max* y_max)
    action_to_take = (action_index-1)*((pixel_value)/2)

    return action_to_take

def action_prob_cal(action_index):
    if(action_index== 0):
        return 0.17
    elif(action_index ==1):
        return 0.2
    elif(action_index ==2):
        return 0.3
    elif(action_index ==3):
        return 0.16
    elif(action_index == 4):
        return 0.12

def risk_calculation(action_index):
    action_list= [-2,0,2,4,6]
    real_probability = [0.17,0.2,0.3,0.16,0.04]
    probaility_1 = []
    for items in action_list:
        res = action_index.count(items)
        probaility_1.append(res)
    fact = 1
    for items in probaility_1:
        fact = fact *np.math.factorial(items)
    action_len = len(action_index)
    advance_func = np.math.factorial(action_len)/fact

    res= [pow(a,b) for a,b in zip(real_probability,probaility_1)]
    result = 1
    for items in res:
        result = result *items
    final_probability = result *advance_func
    return final_probability

def cumm_risk(action_taken):
    i = 0
    cummulative_prob = 0
    cummulative_prob += risk_calculation(action_taken)
    while(i<len(action_taken)):
        if(action_taken[i]== 0):
            action_taken[i] = -2
            cummulative_prob += risk_calculation(action_taken)
            action_taken[i] = 2
            cummulative_prob += risk_calculation(action_taken)
        elif(action_taken[i]== 2):
            action_taken[i] = 4
            cummulative_prob += risk_calculation(action_taken)
        elif(action_taken[i]== 4):
            action_taken[i] = 6
            cummulative_prob += risk_calculation(action_taken)
        elif(action_taken[i]== 6):
            i= i+1
        elif(action_taken[i]== -2):
            i = i+1
    return cummulative_prob