import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os
import argparse
from utils import smooth, strtobool
from environments.maze_env import Maze


parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='reward_plot')
parser.add_argument('--eps', type=str, default='01')
parser.add_argument('--format', type=str, default='png')
parser.add_argument('--showplot', type=strtobool, default=True)
parser.add_argument('--extra_mazes', type=strtobool, default=True)
args = parser.parse_args()

if args.format == 'pdf':
    matplotlib.use('pdf')
    plt.rc('font', family='serif', serif='Times')
    plt.rc('text', usetex=True)

smoothing = True
weight = 0.5
seeds = 5
mazes = 6

if args.eps =='1':
    eps = '_eps1'
elif args.eps == '01':
    eps = '_eps01'

fig = plt.figure()
fig.set_size_inches(w=15, h=10)

end_to_end = False
dqn_only = True
ablation_inverse = True
pretrained = True
pretrained_planning = False

std_error = True                 # False for std_dev
#
iterations0 = np.load(os.getcwd() + '/paper_numpy_arrays/' + 'pretrain_planning_saved_model_parallel_5seeds_iterations0'+ '.npy')
reward0 = np.load(os.getcwd() + '/paper_numpy_arrays/' + 'pretrain_planning_saved_model_parallel_5seeds_reward0'+ '.npy')
reward1 = np.load(os.getcwd() + '/paper_numpy_arrays/' + 'pretrain_planning_saved_model_parallel_5seeds_reward1'+ '.npy')
reward2 = np.load(os.getcwd() + '/paper_numpy_arrays/' + 'pretrain_planning_saved_model_parallel_5seeds_reward2'+ '.npy')
reward3 = np.load(os.getcwd() + '/paper_numpy_arrays/' + 'pretrain_planning_saved_model_parallel_5seeds_reward3'+ '.npy')
reward4 = np.load(os.getcwd() + '/paper_numpy_arrays/' + 'pretrain_planning_saved_model_parallel_5seeds_reward4'+ '.npy')
mean = (reward0 + reward1 + reward2 + reward3 + reward4)/5
distance0 = np.absolute(reward0 - mean)
distance1 = np.absolute(reward1 - mean)
distance2 = np.absolute(reward2 - mean)
distance3 = np.absolute(reward3 - mean)
distance4 = np.absolute(reward4 - mean)
std_dev = np.sqrt((distance0 + distance1 + distance2 + distance3 + distance4)/5)
std_error_plan = std_dev/np.sqrt(5)
iterations0 = np.insert(iterations0, 0, 0)
mean = np.insert(mean, 0, -5)
std_error_plan = np.insert(std_error_plan, 0, 0)

if smoothing:
    mean = smooth(mean, weight=weight)
    std_error_plan = smooth(std_error_plan, weight=weight)
plt.plot(iterations0, mean, label='Interpretable + Planning')
plt.fill_between(iterations0, mean - std_error_plan,
                 mean + std_error_plan, alpha=0.25)

# directory = 'running_agent_eps1/'
# directory = 'random_buffer_eps01/'
# directory = 'running_agent_eps01/'
directory = ''

if args.format == 'pdf':
    matplotlib.use('pdf')
    plt.rc('font', family='serif', serif='Times')
    plt.rc('text', usetex=True)

if end_to_end:
    # End-to-End
    iterations_end_to_end = np.load(os.getcwd() +'/paper_numpy_arrays/' + directory + 'end-to-end_5seeds_iterations' +eps +'.npy')
    iterations_end_to_end = np.insert(iterations_end_to_end, 0, 0)
    mean_end_to_end = np.load(os.getcwd() +'/paper_numpy_arrays/' + directory + 'end-to-end_5seeds_mean' +eps +'.npy')
    mean_end_to_end = np.insert(mean_end_to_end, 0, -5)
    std_dev_end_to_end = np.load(os.getcwd() +'/paper_numpy_arrays/' + directory+ 'end-to-end_5seeds_std_dev' +eps +'.npy')
    std_dev_end_to_end = np.insert(std_dev_end_to_end, 0, 0)

    if std_error:
        std_dev_end_to_end = std_dev_end_to_end / np.sqrt(seeds)
    if smoothing:
        mean_end_to_end = smooth(mean_end_to_end, weight=weight)
        std_dev_end_to_end = smooth(std_dev_end_to_end, weight=weight)
    plt.plot(iterations_end_to_end, mean_end_to_end, label='end-to-end')
    plt.fill_between(iterations_end_to_end, mean_end_to_end-std_dev_end_to_end, mean_end_to_end+std_dev_end_to_end, alpha=0.25)

if dqn_only:
    # Dqn_only
    iterations_dqn_only = np.load(os.getcwd() +'/paper_numpy_arrays/' + directory + 'dqn_only_5seeds_iterations' +eps +'.npy')
    iterations_dqn_only = np.insert(iterations_dqn_only, 0, 0)
    mean_dqn_only = np.load(os.getcwd() +'/paper_numpy_arrays/' + directory + 'dqn_only_5seeds_mean' +eps +'.npy')
    mean_dqn_only = np.insert(mean_dqn_only, 0, -5)
    std_dev_dqn_only = np.load(os.getcwd() +'/paper_numpy_arrays/' + directory + 'dqn_only_5seeds_std_dev' +eps +'.npy')
    std_dev_dqn_only = np.insert(std_dev_dqn_only, 0, 0)
    if std_error:
        std_dev_dqn_only = std_dev_dqn_only / np.sqrt(seeds)
    if smoothing:
        mean_dqn_only = smooth(mean_dqn_only, weight=weight)
        std_dev_dqn_only = smooth(std_dev_dqn_only, weight=weight)
    plt.plot(iterations_dqn_only, mean_dqn_only, label='DDQN')
    plt.fill_between(iterations_dqn_only, mean_dqn_only-std_dev_dqn_only, mean_dqn_only+std_dev_dqn_only, alpha=0.25)

if pretrained:
    # Pretrain_saved_model
    iterations_pretrain = np.load(os.getcwd() +'/paper_numpy_arrays/' + directory + 'pretrain_saved_model_5seeds_iterations' +eps +'.npy')
    iterations_pretrain = np.insert(iterations_pretrain, 0 , 0)
    mean_pretrain = np.load(os.getcwd() +'/paper_numpy_arrays/' + directory + 'pretrain_saved_model_5seeds_mean' +eps +'.npy')
    mean_pretrain = np.insert(mean_pretrain, 0 , -5)
    std_dev_pretrain = np.load(os.getcwd() +'/paper_numpy_arrays/' + directory + 'pretrain_saved_model_5seeds_std_dev' +eps +'.npy')
    std_dev_pretrain = np.insert(std_dev_pretrain, 0 , 0)
    if std_error:
        std_dev_pretrain = std_dev_pretrain / np.sqrt(seeds)
    if smoothing:
        mean_pretrain = smooth(mean_pretrain, weight=weight)
        std_dev_pretrain = smooth(std_dev_pretrain, weight=weight)
    plt.plot(iterations_pretrain, mean_pretrain, label='Interpretable')
    plt.fill_between(iterations_pretrain, mean_pretrain-std_dev_pretrain, mean_pretrain+std_dev_pretrain, alpha=0.25)

if pretrained_planning:
    # Pretrain_saved_model_planning
    iterations_pretrain_planning = np.load(os.getcwd() +'/paper_numpy_arrays/' + directory + 'pretrain_planning_saved_model_5seeds_iterations' +eps +'.npy')
    iterations_pretrain_planning = np.insert(iterations_pretrain_planning, 0, 0)
    mean_pretrain_planning = np.load(os.getcwd() +'/paper_numpy_arrays/' + directory + 'pretrain_planning_saved_model_5seeds_mean' +eps +'.npy')
    mean_pretrain_planning = np.insert(mean_pretrain_planning, 0, -5)
    std_dev_pretrain_planning = np.load(os.getcwd() +'/paper_numpy_arrays/' + directory + 'pretrain_planning_saved_model_5seeds_std_dev' +eps +'.npy')
    std_dev_pretrain_planning = np.insert(std_dev_pretrain_planning, 0, 0)
    if std_error:
        std_dev_pretrain_planning = std_dev_pretrain_planning / np.sqrt(seeds)
    if smoothing:
        mean_pretrain_planning = smooth(mean_pretrain_planning, weight=weight)
        std_dev_pretrain_planning = smooth(std_dev_pretrain_planning, weight=weight)
    plt.plot(iterations_pretrain_planning, mean_pretrain_planning, label='Interpretable + Planning')
    plt.fill_between(iterations_pretrain_planning, mean_pretrain_planning-std_dev_pretrain_planning, mean_pretrain_planning+std_dev_pretrain_planning, alpha=0.25)

if ablation_inverse:
    # ablation_inverse
    iterations_ablation_inverse = np.load(os.getcwd() +'/paper_numpy_arrays/' + directory + 'ablation_inverse_5seeds_iterations' +eps +'.npy')
    iterations_ablation_inverse = np.insert(iterations_ablation_inverse, 0, 0)
    mean_ablation_inverse = np.load(os.getcwd() +'/paper_numpy_arrays/' + directory + 'ablation_inverse_5seeds_mean' +eps +'.npy')
    mean_ablation_inverse = np.insert(mean_ablation_inverse, 0, -5)
    std_dev_ablation_inverse = np.load(os.getcwd() +'/paper_numpy_arrays/' + directory + 'ablation_inverse_5seeds_std_dev' +eps +'.npy')
    std_dev_ablation_inverse = np.insert(std_dev_ablation_inverse, 0, 0)
    if std_error:
        std_dev_ablation_inverse = std_dev_ablation_inverse / np.sqrt(seeds)
    if smoothing:
        mean_ablation_inverse = smooth(mean_ablation_inverse, weight=weight)
        std_dev_ablation_inverse = smooth(std_dev_ablation_inverse, weight=weight)
    plt.plot(iterations_ablation_inverse, mean_ablation_inverse, label='Inverse Prediction')
    plt.fill_between(iterations_ablation_inverse, mean_ablation_inverse-std_dev_ablation_inverse, mean_ablation_inverse+std_dev_ablation_inverse, alpha=0.25)
plt.legend(fontsize=7)
plt.xlabel('Iterations')
plt.ylabel('Average Reward')

plt.gcf().set_size_inches(4, 3)
plt.savefig(os.getcwd() + '/paper_numpy_arrays' + args.name+'.png', bbox_inches='tight')
plt.savefig(os.getcwd() +'/paper_numpy_arrays' + args.name+'.pdf', bbox_inches='tight')

if args.showplot:
    plt.show()
plt.savefig(os.getcwd() + '/paper_numpy_arrays' + args.name+'.png', bbox_inches='tight')
plt.savefig(os.getcwd() +'/paper_numpy_arrays' + args.name+'.pdf', bbox_inches='tight')

if args.extra_mazes:
    if mazes == 9:
        rng = np.random.RandomState(12346)
        env = Maze(rng, higher_dim_obs=True, map_type='path_finding', maze_size=8, random_start=False)
        env.create_map()
        picture1 = env.observe()[0]
        env.reset(mode=10)
        picture2 = env.observe()[0]
        env.reset(mode=10)
        picture3 = env.observe()[0]
        env.reset(mode=10)
        picture4 = env.observe()[0]
        env.reset(mode=10)
        picture5 = env.observe()[0]
        env.reset(mode=10)
        picture6 = env.observe()[0]
        env.reset(mode=10)
        picture7 = env.observe()[0]
        env.reset(mode=10)
        picture8 = env.observe()[0]
        env.reset(mode=10)
        picture9 = env.observe()[0]
        fig = plt.figure()
        plt.subplot(3, 3, 1)
        plt.imshow(picture1, cmap='gray')
        ax = plt.gca()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.subplot(3, 3, 2)
        plt.imshow(picture2, cmap='gray')
        ax = plt.gca()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.subplot(3, 3, 3)
        plt.imshow(picture3, cmap='gray')
        ax = plt.gca()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.subplot(3, 3, 4)
        plt.imshow(picture4, cmap='gray')
        ax = plt.gca()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.subplot(3, 3, 5)
        plt.imshow(picture5, cmap='gray')
        ax = plt.gca()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.subplot(3, 3, 6)
        plt.imshow(picture6, cmap='gray')
        ax = plt.gca()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.subplot(3, 3, 7)
        plt.imshow(picture7, cmap='gray')
        ax = plt.gca()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.subplot(3, 3, 8)
        plt.imshow(picture8, cmap='gray')
        ax = plt.gca()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.subplot(3, 3, 9)
        plt.imshow(picture9, cmap='gray')
        ax = plt.gca()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        fig.tight_layout()
        plt.tight_layout()
        # plt.show()
        plt.savefig(os.getcwd() + '/paper_numpy_arrays' + '/9_mazes' + '.pdf', bbox_inches='tight')
    elif mazes == 6:
        rng = np.random.RandomState(12346)
        env = Maze(rng, higher_dim_obs=True, map_type='path_finding', maze_size=8, random_start=False)
        env.create_map()
        picture1 = env.observe()[0]
        env.reset(mode=10)
        picture2 = env.observe()[0]
        env.reset(mode=10)
        picture3 = env.observe()[0]
        env.reset(mode=10)
        picture4 = env.observe()[0]
        env.reset(mode=10)
        picture5 = env.observe()[0]
        env.reset(mode=10)
        picture6 = env.observe()[0]
        fig = plt.figure()
        plt.subplot(3, 2, 1)
        plt.imshow(picture1, cmap='gray')
        ax = plt.gca()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.subplot(3, 2, 2)
        plt.imshow(picture2, cmap='gray')
        ax = plt.gca()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.subplot(3, 2, 3)
        plt.imshow(picture3, cmap='gray')
        ax = plt.gca()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.subplot(3, 2, 4)
        plt.imshow(picture4, cmap='gray')
        ax = plt.gca()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.subplot(3, 2, 5)
        plt.imshow(picture5, cmap='gray')
        ax = plt.gca()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.subplot(3, 2, 6)
        plt.imshow(picture6, cmap='gray')
        ax = plt.gca()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        fig.tight_layout()
        plt.tight_layout()
        # plt.show()
        plt.savefig(os.getcwd() + '/paper_numpy_arrays' + '/6_mazes' + '.pdf', bbox_inches='tight')

