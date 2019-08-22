import gym
import multiprocessing as mp

# This multi-processing tests how to collect data from multiple environment process


def run(q, env, i):
    done = True
    while True:
        if done:
            prev_obs = env.reset()
        else:
            prev_obs = obs
        obs, rew, done, info = env.step(env.action_space.sample())
        data = prev_obs, rew, done, i
        q.put(data)  # Send data to main function


def main():
    n_procs = 10  # number of processes
    q_size = 10
    q = mp.Queue(q_size)
    procs = []
    for i in range(n_procs):
        env = gym.make('CartPole-v0')
        proc = mp.Process(target=run, args=[q, env, i])
        procs.append(proc)  # Append all process to a list
        proc.start()  # Start the process
    while True:
        data = []
        print('new data queue')
        for i in range(q_size):
            data.append(q.get())  # Receive data from a run process and append to data object
            print('i: ', data[-1])


if __name__ == '__main__':
    main()
