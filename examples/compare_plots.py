import json
import numpy as np
import matplotlib.pyplot as plt

# Alternating gaps only
# Params used:
# n_training = 30
# n_val = 10
# sail_params['beta0'] = 0
# sail_params['k']     = 30
# sail_params['N']     = 3
# sail_params['T']     = 6000
# sail_params['Tv']    = 2000
# sail_params['m']     = 30
# sail_params['mv']    = 10
# 
# learner_params['output_size'] = 1
# learner_params['input_size'] = 17 
# learner_params['batch_size'] = 64
# learner_params['training_epochs'] = 20 
# learner_params['seed_val'] = 1234
# learner_params['mode'] = "cpu"
# learner_params['display_step'] = 1
def read_json(json_filename):
    json_file = open(json_filename)
    json_str = json_file.read()
    return json.loads(json_str)

parent_dir = './plot_comparison'
file1 = 'train_iter_3_features_17_num_train_envs_30_num_valid_envs_10.json'
file2 = None


file1_data = read_json(file1)
file1_train_loss_hist = np.array(file1_data['train_loss_hist'])

plt.plot(file1_train_loss_hist[0])
plt.show()