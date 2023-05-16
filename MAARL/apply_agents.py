import time
import numpy as np
from ppo import Agent
from Environment import Environment
import config


# apply both agents: mode=0
# apply only prot: mode=1
# apply only adv: mode=2
mode = 0

ACTION_SPACE_STEP_ADVERSARY = 5
ACTION_SPACE_STEP_PROT = 4  # has to be even!


config_file = config.config_1

adv1 = Agent(n_actions=5, batch_size=config_file['batch_size'], gamma=config_file['gamma'],
                 alpha=config_file['lr'], n_epochs=config_file['n_epochs'],
                 model_arch=config_file['model_arch'], epsilon=config_file['epsilon'], entropy_coef=config_file['entropy_coef'], value_loss_coef=config_file['value_loss_coef'])

adv1.load_models(path='adv_1' + '_models/', name='best')

if mode == 0:   # apply both agents:
    env = Environment(map_path=config_file['map_path'], relevant_segments=0, done_after_collision=True, adversary=adv1, visualize=True)
elif mode == 1 or mode == 2: # apply only prot / apply only adv
    env = Environment(map_path=config_file['map_path'], relevant_segments=0, done_after_collision=True, adversary=None, visualize=True)

prot = Agent(n_actions=5, batch_size=config_file['batch_size'], gamma=config_file['gamma'],
                alpha=config_file['lr'], n_epochs=config_file['n_epochs'], model_arch=config_file['model_arch'], epsilon=config_file['epsilon'], entropy_coef=config_file['entropy_coef'], value_loss_coef=config_file['value_loss_coef'])

prot.load_models(path='prot_1' + '_models/', name='best')


# applying agents to a trajectory: ----------------------------------------------------------------------

if mode == 0 or mode == 1:  # apply both agents / apply only prot
    observation = env.reset('prot')

    action_index, _, _, _ = prot.choose_action(observation, test_mode=True)
    _, reward_adv, _, _, traj = env.step_prot(ACTION_SPACE_STEP_PROT*action_index - 1)
elif mode == 2:     # apply only adv
    observation = env.reset('adv1')

    traj = [env.trajectory_vanilla[0]]
    done = False
    while not done:
        action_index, _, _, _ = adv1.choose_action(observation, test_mode=True)
        action_angle_offset = np.deg2rad(ACTION_SPACE_STEP_ADVERSARY * action_index - (int(config_file['N_actions']/2)*ACTION_SPACE_STEP_ADVERSARY))
        observation, reward_adv, done, _, node_adv = env.step_adv1(action_angle_offset)
        print('reward:', reward_adv)
        if reward_adv == 1:
            print('collision')
        traj.append(node_adv)

        time.sleep(3)   # this is just to slowly see the trajectory building up for evaluation -> should be removed in application
# -------------------------------------------------------------------------------------------------------

# the list "traj" now contains the nodes (waypoints) after the manipulation of the agents
print('trajectory:', traj)
