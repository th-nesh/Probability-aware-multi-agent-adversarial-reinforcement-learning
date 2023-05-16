import time
import numpy as np
import evaluate_logs

from CSV_Logger import CSV_Reader
avg_window = 20   # determines how many episodes are averaged in plots

#log_reader = CSV_Reader('prot_1', test_mode=False)
#evaluate_logs.plot_collisions('prot_1', log_reader, avg_window=avg_window)
#evaluate_logs.plot_length_ratios('prot_1', log_reader, avg_window=avg_window)
#evaluate_logs.plot_scores('prot_1', log_reader, avg_window=avg_window)
#print('probs adv:')
#evaluate_logs.calc_probabilities(log_reader)  # for evaluation: calculates and prints the frequencies of chosen actions (for histogram)

log_reader = CSV_Reader('adv_1', test_mode=False)
#log_reader_1 = CSV_Reader('adv_6', test_mode=False)
evaluate_logs.plot_rewards('adv_1', log_reader, avg_window=3*avg_window)    # rewards (for adversary) are per step instead of per episode -> choose bigger average window
evaluate_logs.plot_scores('adv_1', log_reader, avg_window=avg_window)
evaluate_logs.plot_collisions('adv_1', log_reader, avg_window=avg_window)
#evaluate_logs.plot_collisions_dual('adv_5', log_reader,log_reader_1, avg_window=avg_window)
#evaluate_logs.plot_rewards_dual('adv_5', log_reader,log_reader_1, avg_window=3*avg_window)
#evaluate_logs.plot_scores_dual('adv_5', log_reader,log_reader_1,avg_window=avg_window)
#print('probs adv:')
evaluate_logs.calc_probabilities(log_reader)  # for evaluation: calculates and prints the frequencies of chosen actions (for histogram)
#[[0.69976786 0.0661594  0.00419139 0.04687903 0.18300232]]