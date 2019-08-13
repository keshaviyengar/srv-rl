import numpy as np

import optuna
from optuna.pruners import SuccessiveHalvingPruner, MedianPruner
from optuna.samplers import RandomSampler, TPESampler

from rl_controller import RLController


def hyperparam_opt(env, n_trials=10, agent_hyperparams=None, training_hyperparams=None, sampler_method='random',
                   pruner_method='halving', seed=1):
    # Set hyperparams to dictionary to get updated by sample_params
    if agent_hyperparams is None:
        agent_hyperparams = {}
    if training_hyperparams is None:
        training_hyperparams = {}

    if sampler_method == 'random':
        sampler = RandomSampler(seed=seed)
    elif sampler_method == 'tpe':
        sampler = TPESampler(n_startup_trials=5, seed=seed)
    else:
        raise ValueError('Unknown sampler: {}'.format(sampler_method))

    if pruner_method == 'halving':
        pruner = SuccessiveHalvingPruner(min_resource=1, reduction_factor=4, min_early_stopping_rate=0)
    elif pruner_method == 'median':
        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    elif pruner_method == 'none':
        # Do not prune
        pruner = MedianPruner(n_startup_trials=n_trials, n_warmup_steps=10)
    else:
        raise ValueError('Unknown pruner: {}'.format(pruner_method))

    study = optuna.create_study(sampler=sampler, pruner=pruner)

    def objective(trial):
        agent_kwargs, training_kwargs = ddpg_her_sample_params(trial, agent_hyperparams, training_hyperparams)

        for key, value in trial.params.items():
            print('{}: {}'.format(key, value))

        # Create RL controller object
        rl_controller = RLController(env=env, agent_kwargs=agent_kwargs)
        try:
            eval_success_rate, eval_mean_reward_ep = rl_controller.train(**training_kwargs)
        except AssertionError:
            eval_mean_reward_ep = np.inf
        cost = -1 * eval_mean_reward_ep
        del rl_controller
        return cost

    try:
        study.optimize(objective, n_trials=n_trials)
    except KeyboardInterrupt:
        pass

    print('Number of finished trials: ', len(study.trials))

    print('Best trial:')
    trial = study.best_trial

    print('Value: ', trial.value)

    print('Params: ')
    for key, value in trial.params.items():
        print('{}: {}'.format(key, value))

    return study.trials_dataframe()


def ddpg_her_sample_params(trial, agent_hyperparams, training_hyperparams):
    if 'gamma' not in agent_hyperparams:
        gamma = trial.suggest_categorical('gamma', [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
    else:
        gamma = agent_hyperparams['gamma']
    if 'tau' not in agent_hyperparams:
        tau = trial.suggest_categorical('tau', [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001])
    else:
        tau = agent_hyperparams['tau']
    if 'actor_learning_rate' not in agent_hyperparams:
        actor_learning_rate = trial.suggest_categorical('actor_lr', [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001])
    else:
        actor_learning_rate = agent_hyperparams['actor_learning_rate']
    if 'critic_learning_rate' not in agent_hyperparams:
        critic_learning_rate = trial.suggest_categorical('critic_lr', [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001])
    else:
        critic_learning_rate = agent_hyperparams['critic_learning_rate']
    if 'batch_size' not in agent_hyperparams:
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128, 256])
    else:
        batch_size = agent_hyperparams['batch_size']
    if 'buffer_size' not in agent_hyperparams:
        buffer_size = trial.suggest_categorical('buffer_size', [int(1e4), int(1e5), int(1e6)])
    else:
        buffer_size = agent_hyperparams['buffer_size']

    if 'k' not in training_hyperparams:
        k = trial.suggest_categorical('k', [1, 2, 4, 4, 5, 6, 7, 8])
    else:
        k = training_hyperparams['k']
    if 'optimization_steps' not in training_hyperparams:
        optimization_steps = trial.suggest_categorical('optimization_steps', [5, 10, 15, 20, 25, 30])
    else:
        optimization_steps = training_hyperparams['optimization_steps']
    if 'action_noise' not in training_hyperparams:
        action_noise = trial.suggest_uniform('action_noise', 0, 1)
    else:
        action_noise = training_hyperparams['action_noise']

    new_agent_hyperparams = {
        'gamma': gamma,
        'actor_learning_rate': actor_learning_rate,
        'critic_learning_rate': critic_learning_rate,
        'tau': tau,
        'batch_size': batch_size,
        'buffer_size': buffer_size
    }

    new_agent_hyperparams.update(agent_hyperparams)

    new_training_hyperparams = {
        'k': k,
        'optimization_steps': optimization_steps,
        'action_noise': action_noise
    }

    new_training_hyperparams.update(training_hyperparams)

    return new_agent_hyperparams, new_training_hyperparams


from test_envs.bit_flipping_env import BitFlippingEnv

if __name__ == '__main__':
    env = BitFlippingEnv(continuous=True, max_steps=50)
    # Load json file of parameters
    #with open("ddpg_her_hyperparam_parameters.json", 'r') as f:
    #    params = json.load(f)
    #    agent_kwargs = params['agent_hyperparams']
    #    training_kwargs = params['training_hyperparams']
    agent_kwargs = None
    training_kwargs = None

    opt_params_dataframe = hyperparam_opt(env=env, n_trials=1000, agent_hyperparams=agent_kwargs, training_hyperparams=training_kwargs)
    opt_params_dataframe.to_csv('hyperparam_search.csv', index=False, header=True)
