import json
import math
import numpy as np
import matplotlib.pyplot as plt


def f(v, alpha):
    return v**2*math.sin(math.radians(2*alpha))/9.78


def create_params_json(observations=10000, filename='params.json'):
    params_structure = {
        'v_normal': np.random.normal(loc=40, scale=5, size=observations).tolist(),
        'v_uniform': np.random.uniform(low=20, high=60, size=observations).tolist(),
        'alpha_normal': np.random.normal(loc=60, scale=7, size=observations).tolist(),
        'alpha_uniform': np.random.uniform(low=45, high=75, size=observations).tolist()
    }

    fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(15, 5))
    for i, (name, values) in enumerate(params_structure.items()):
        ax[i].set_title(name)
        ax[i].hist(x=values)
    plt.show()

    with open(filename, 'w') as file:
        params_structure['observations'] = observations
        json.dump(params_structure, file, indent=4)


def l_distributions(params_file='params.json'):
    with open(params_file, 'r') as file:
        params = json.load(file)

    dist_types = ['normal', 'uniform']
    fig, ax = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle(f'Observations = {params["observations"]}')
    for i, i_name in enumerate(dist_types):
        for j, j_name in enumerate(dist_types):
            lens = [f(params[f'v_{i_name}'][el], params[f'alpha_{j_name}'][el]) for el in range(len(params['v_normal']))]
            ax[i, j].set_title(f'v_{i_name} vs alpha_{j_name}')
            ax[i, j].hist(x=lens)
    plt.show()


def main():
    create_params_json()
    l_distributions()


if __name__ == "__main__":
    main()
