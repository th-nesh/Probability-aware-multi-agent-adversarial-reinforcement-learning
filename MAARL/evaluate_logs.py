import numpy as np
import matplotlib.pyplot as plt


def plot_rewards(path, log_reader, avg_window=1000):

    rewards = np.array(log_reader.rewards)
    n_windows = int(len(rewards)/avg_window)
    rewards_avg = []

    # calculate average reward
    for i in range(0, n_windows):
        rewards_window = rewards[(i*avg_window):(i*avg_window+avg_window)]
        rewards_avg.append(np.mean(rewards_window))

    x = list(range(0, n_windows))
    x = np.array(x)*avg_window

    plt.close()

    plt.plot(x, rewards_avg)
    plt.xlabel('steps')
    plt.ylabel('average reward')
    plt.grid()

    plt.savefig(path+"_rewards.png")
    plt.savefig(path+"_rewards.pdf")


def plot_scores(path, log_reader, avg_window=500):

    rewards = np.array(log_reader.rewards)
    steps = np.array(log_reader.steps)

    scores = []
    score = 0
    for i in range(0, len(steps)-1):
        score += rewards[i]
        if steps[i+1] <= steps[i]:
            scores.append(score)
            score = 0
        if i == len(steps)-2:
            if steps[-1] <= steps[-2]:
                scores.append(score)
                scores.append(rewards[-1])
            else:
                scores.append(score+rewards[-1])

    n_windows = int(len(scores)/avg_window)
    scores_avg = []

    # calculate average reward
    for i in range(0, n_windows):
        scores_window = scores[(i*avg_window):(i*avg_window+avg_window)]
        scores_avg.append(np.mean(scores_window))

    plt.close()

    x = list(range(0, n_windows))
    x = np.array(x)*avg_window
    plt.plot(x, scores_avg)
    plt.xlabel('episode')
    plt.ylabel('average score')
    plt.grid()

    plt.savefig(path+"_scores.png")
    plt.savefig(path+"_scores.pdf")

def plot_scores_dual(path, log_reader_1,log_reader_2, avg_window=500):

    rewards_1 = np.array(log_reader_1.rewards)
    rewards_2 = np.array(log_reader_2.rewards)
    steps_1 = np.array(log_reader_1.steps)
    steps_2 = np.array(log_reader_2.steps)
    scores_1 = []
    score_1 = 0
    for i in range(0, len(steps_1)-1):
        score_1 += rewards_1[i]
        if steps_1[i+1] <= steps_1[i]:
            scores_1.append(score_1)
            score_1 = 0
        if i == len(steps_1)-2:
            if steps_1[-1] <= steps_1[-2]:
                scores_1.append(score_1)
                scores_1.append(rewards_1[-1])
            else:
                scores_1.append(score_1+rewards_1[-1])

    n_windows = int(len(scores_1)/avg_window)
    scores_avg_1 = []

    # calculate average reward
    for i in range(0, n_windows):
        scores_window_1 = scores_1[(i*avg_window):(i*avg_window+avg_window)]
        scores_avg_1.append(np.mean(scores_window_1))

    
    scores_2 = []
    score_2 = 0
    for i in range(0, len(steps_2)-1):
        score_2 += rewards_2[i]
        if steps_2[i+1] <= steps_2[i]:
            scores_2.append(score_2)
            score_2 = 0
        if i == len(steps_2)-2:
            if steps_2[-1] <= steps_2[-2]:
                scores_2.append(score_2)
                scores_2.append(rewards_2[-1])
            else:
                scores_2.append(score_2+rewards_2[-1])

    #n_windows = int(len(scores_1)/avg_window)
    scores_avg_2 = []

    # calculate average reward
    for i in range(0, n_windows):
        scores_window_2 = scores_2[(i*avg_window):(i*avg_window+avg_window)]
        scores_avg_2.append(np.mean(scores_window_2))
    plt.close()

    x = list(range(0, n_windows))
    x = np.array(x)*avg_window
    plt.plot(x, scores_avg_1,label= "Probability aware MAARL", color = 'g')
    plt.plot(x, scores_avg_2, label = "MAARL without probabilities",color =  'y')

    plt.xlabel('episode')
    plt.ylabel('average score')
    plt.grid()
    plt.legend()

    plt.savefig(path+"_scores.png")
    plt.savefig(path+"_scores.pdf")

def plot_collisions(path, log_reader, avg_window=500):

    collisions = np.array(log_reader.collisions)
    steps = np.array(log_reader.steps)

    collisions_scores = []
    collision = 0
    for i in range(0, len(steps)-1):
        collision += collisions[i]
        if steps[i+1] <= steps[i]:
            collisions_scores.append(collision)
            collision = 0
        if i == len(steps)-2:
            if steps[-1] <= steps[-2]:
                collisions_scores.append(collision)
                collisions_scores.append(collisions[-1])
            else:
                collisions_scores.append(collision+collisions[-1])

    n_windows = int(len(collisions_scores)/avg_window)
    collisions_scores_avg = []

    # calculate average reward
    for i in range(0, n_windows):
        collisions_window = collisions_scores[(i*avg_window):(i*avg_window+avg_window)]
        collisions_scores_avg.append(np.mean(collisions_window))

    plt.close()

    x = list(range(0, n_windows))
    x = np.array(x)*avg_window
    plt.plot(x, collisions_scores_avg)
    plt.xlabel('episode')
    plt.ylabel('average collision rate')
    plt.grid()

    plt.savefig(path+"_collisions.png")
    plt.savefig(path+"_collisions.pdf")


def plot_length_ratios(path, log_reader, avg_window=500):

    length_ratios = np.array(log_reader.length_ratios)
    steps = np.array(log_reader.steps)

    length_ratios_list = []
    length_ratio = 0
    for i in range(0, len(steps)-1):
        length_ratio += -(2*length_ratios[i]-1)
        if steps[i+1] <= steps[i]:
            length_ratios_list.append(length_ratio)
            length_ratio = 0
        if i == len(steps)-2:
            if steps[-1] <= steps[-2]:
                length_ratios_list.append(length_ratio)
                length_ratios_list.append(length_ratios[-1])
            else:
                length_ratios_list.append(length_ratio+length_ratios[-1])

    n_windows = int(len(length_ratios_list)/avg_window)
    length_ratios_avg = []

    # calculate average reward
    for i in range(0, n_windows):
        length_ratios_window = length_ratios_list[(i*avg_window):(i*avg_window+avg_window)]
        length_ratios_avg.append(np.mean(length_ratios_window))

    plt.close()

    x = list(range(0, n_windows))
    x = np.array(x)*avg_window
    plt.plot(x, length_ratios_avg)
    plt.xlabel('episode')
    plt.ylabel('average length ratio')
    plt.grid()

    plt.savefig(path+"_length_ratios.png")
    plt.savefig(path+"_length_ratios.pdf")


def plot_collisions_special(path, log_reader, avg_window=500):

    collisions = np.array(log_reader.rewards)
    steps = np.array(log_reader.steps)

    collisions_scores = []
    collision = 0
    for i in range(0, len(steps)-1):
        if collisions[i] != 1:
            collision += 0
        else:
            collision += 1
        if steps[i+1] <= steps[i]:
            collisions_scores.append(collision)
            collision = 0
        if i == len(steps)-2:
            if steps[-1] <= steps[-2]:
                collisions_scores.append(collision)
                collisions_scores.append(collisions[-1])
            else:
                collisions_scores.append(collision+collisions[-1])

    n_windows = int(len(collisions_scores)/avg_window)
    collisions_scores_avg = []

    # calculate average reward
    for i in range(0, n_windows):
        collisions_window = collisions_scores[(i*avg_window):(i*avg_window+avg_window)]
        collisions_scores_avg.append(np.mean(collisions_window))

    plt.close()

    x = list(range(0, n_windows))
    x = np.array(x)*avg_window
    plt.plot(x, collisions_scores_avg)
    plt.xlabel('episode')
    plt.ylabel('average collision rate')
    plt.grid()

    plt.savefig(path+"_collisions.png")
    plt.savefig(path+"_collisions.pdf")

def plot_collisions_dual(path, log_reader1,log_reader2, avg_window=500):

    collisions_1 = np.array(log_reader1.collisions)
    steps_1 = np.array(log_reader1.steps)

    collisions_2 = np.array(log_reader2.collisions)
    steps_2 = np.array(log_reader2.steps)

    collisions_scores_1 = []
    collision_1 = 0
    for i in range(0, len(steps_1)-1):
        collision_1 += collisions_1[i]
        if steps_1[i+1] <= steps_1[i]:
            collisions_scores_1.append(collision_1)
            collision_1 = 0
        if i == len(steps_1)-2:
            if steps_1[-1] <= steps_1[-2]:
                collisions_scores_1.append(collision_1)
                collisions_scores_1.append(collisions_1[-1])
            else:
                collisions_scores_1.append(collision_1+collisions_1[-1])

    n_windows = int(len(collisions_scores_1)/avg_window)
    collisions_scores_avg_1 = []

    # calculate average reward
    for i in range(0, n_windows):
        collisions_window_1 = collisions_scores_1[(i*avg_window):(i*avg_window+avg_window)]
        collisions_scores_avg_1.append(np.mean(collisions_window_1))

    # test
    collisions_scores_2 = []
    collision_2 = 0
    for i in range(0, len(steps_2)-1):
        collision_2 += collisions_2[i]
        if steps_2[i+1] <= steps_2[i]:
            collisions_scores_2.append(collision_2)
            collision_2 = 0
        if i == len(steps_2)-2:
            if steps_2[-1] <= steps_2[-2]:
                collisions_scores_2.append(collision_2)
                collisions_scores_2.append(collisions_2[-1])
            else:
                collisions_scores_2.append(collision_2+collisions_2[-1])

    #n_windows = int(len(collisions_scores_1)/avg_window)
    collisions_scores_avg_2 = []

    # calculate average reward
    for i in range(0, n_windows):
        collisions_window_2 = collisions_scores_2[(i*avg_window):(i*avg_window+avg_window)]
        collisions_scores_avg_2.append(np.mean(collisions_window_2))
    plt.close()

    x = list(range(0, n_windows))
    x = np.array(x)*avg_window
    plt.plot(x, collisions_scores_avg_1,label= "Probability aware MAARL", color = 'g')
    plt.plot(x, collisions_scores_avg_2, label = "MAARL without probabilities",color =  'y')
    plt.xlabel('episode')
    plt.ylabel('average collision rate')
    plt.grid()
    plt.legend()
 


    plt.savefig(path+"_collisions.png")
    plt.savefig(path+"_collisions.pdf")

def plot_rewards_dual(path, log_reader_1,log_reader_2, avg_window=1000):

    rewards_1 = np.array(log_reader_1.rewards)
    rewards_2 = np.array(log_reader_2.rewards)
    n_windows = int(len(rewards_1)/avg_window)
    rewards_avg_1 = []
    rewards_avg_2 = []

    # calculate average reward
    for i in range(0, n_windows):
        rewards_window_1 = rewards_1[(i*avg_window):(i*avg_window+avg_window)]
        rewards_avg_1.append(np.mean(rewards_window_1))

    for i in range(0, n_windows):
        rewards_window_2 = rewards_2[(i*avg_window):(i*avg_window+avg_window)]
        rewards_avg_2.append(np.mean(rewards_window_2))
    x = list(range(0, n_windows))
    x = np.array(x)*avg_window

    plt.close()

    plt.plot(x, rewards_avg_1,label= "Probability aware MAARL", color = 'g')
    plt.plot(x, rewards_avg_2, label = "MAARL without probabilities",color =  'y')
    plt.xlabel('steps')
    plt.ylabel('average reward')
    plt.legend()
    plt.grid()

    plt.savefig(path+"_rewards.png")
    plt.savefig(path+"_rewards.pdf")

def calc_probabilities(log_reader):
    actions_0 = np.array(log_reader.actions_0)
    actions_1 = np.array(log_reader.actions_1)
    actions_2 = np.array(log_reader.actions_2)
    actions_3 = np.array(log_reader.actions_3)
    actions_4 = np.array(log_reader.actions_4)

    prob_0 = 0
    prob_1 = 0
    prob_2 = 0
    prob_3 = 0
    prob_4 = 0

    probs = [prob_0, prob_1, prob_2, prob_3, prob_4]

    for i in range(0, len(actions_0)):
        a_chosen = np.argmax(np.array([actions_0[i], actions_1[i], actions_2[i], actions_3[i], actions_4[i]]))
        probs[a_chosen] += 1

    # print('probs anders:')
    # print(np.mean(np.array([actions_0])), np.mean(np.array([actions_1])), np.mean(np.array([actions_2])), np.mean(np.array([actions_3])), np.mean(np.array([actions_4])))

    print('probs:')
    print(np.array([probs])/len(actions_0))


#plot_rewards(path, log_reader)
#plot_scores(path, log_reader)
