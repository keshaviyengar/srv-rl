

# Logger is meant to take out printing and logging from the rl_controller and ddpg_her classes
class Logger:
    def __init__(self):
        self.epoch_data = {'Epoch': [], 'Success rate': [], 'Mean reward per episode': [], 'Actor losses': [],
                           'Critic losses': []}
        self.episode_data = {'Epoch': [], 'Episode': [], 'Step': [], 'State (joint)': [], 'State (achieved goal)': [],
                             'Action': [], 'Next state (joint)': [], 'Next state (achieved goal)': [], 'Reward': [],
                             'Success': []}
        self.eval_epoch_data = {'Epoch': [], 'Success rate': [], 'Mean reward per episode': []}

    def update_epoch_data(self, epoch, success_rate, mean_reward_per_ep, actor_losses, critic_losses):
        self.epoch_data['Epoch'].append(epoch)
        self.epoch_data['Success rate'].append(success_rate)
        self.epoch_data['Mean reward per episode'].append(mean_reward_per_ep)
        self.epoch_data['Actor losses'].append(actor_losses)
        self.epoch_data['Critic losses'].append(critic_losses)

    def update_episode_data(self, epoch, episode, step, state_joint, state_ag, action, next_state_joint, next_state_ag,
                            reward, success):
        # record data for episode
        self.episode_data['Epoch'].append(epoch)
        self.episode_data['Episode'].append(episode)
        self.episode_data['Step'].append(step)
        self.episode_data['State (joint)'].append(state_joint)
        self.episode_data['State (achieved goal)'].append(state_ag)
        self.episode_data['Action'].append(action)
        self.episode_data['Next state (joint)'].append(next_state_joint)
        self.episode_data['Next state (achieved goal)'].append(next_state_ag)
        self.episode_data['Reward'].append(reward)
        self.episode_data['Success'].append(success)

    def update_eval_epoch_data(self, epoch, success_rate, mean_reward_per_ep):
        self.eval_epoch_data['Epoch'].append(epoch)
        self.eval_epoch_data['Success rate'].append(success_rate)
        self.eval_epoch_data['Mean reward per episode'].append(mean_reward_per_ep)

    def print_data(self, data_name):
        if data_name == 'epoch':
            titles, values = self.convert_data_dict_list(self.epoch_data)
        elif data_name == 'episode':
            titles, values = self.convert_data_dict_list(self.episode_data)
        elif data_name == 'eval':
            titles, values = self.convert_data_dict_list(self.eval_epoch_data)
        else:
            print('Incorrect save_type parameter in print data function.')
            titles, values = 0, 0

        self.print_in_a_frame(titles, values)

    @staticmethod
    def convert_data_dict_list(data):
        titles = list(data.keys())
        values = list(data.values())
        return titles, values

    @staticmethod
    def print_in_a_frame(titles, values):
        padding = 4
        size = len(max(titles, key=len))
        for i in range(len(titles)):
            print('* {a:<{b}} {c:>{b}} {d}'.format(a=titles[i], b=size, c=str(values[i][-1]), d='*'))
        print('')
