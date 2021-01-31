import sys
import time
import signal
import argparse
import pandas as pd
import numpy as np
import torch
import visdom
import data
from models import *
from comm import CommNetMLP
from utils import *
from action_utils import parse_action_args
from trainer import Trainer
from multi_processing import MultiProcessTrainer
import time
import datetime
import os
torch.utils.backcompat.broadcast_warning.enabled = True
torch.utils.backcompat.keepdim_warning.enabled = True

torch.set_default_tensor_type('torch.DoubleTensor')


csv_data = []



parser = argparse.ArgumentParser(description='PyTorch RL trainer')
# training
# note: number of steps per epoch = epoch_size X batch_size x nprocesses
parser.add_argument('--num_epochs', default=30000, type=int,
                    help='number of training epochs')
parser.add_argument('--epoch_size', type=int, default=1,
                    help='number of update iterations in an epoch')
parser.add_argument('--batch_size', type=int, default=60,
                    help='number of steps before each update (per thread)')
parser.add_argument('--nprocesses', type=int, default=1,
                    help='How many processes to run')
# model
parser.add_argument('--hid_size', default=64, type=int,
                    help='hidden layer size')
parser.add_argument('--recurrent', action='store_true', default=False,
                    help='make the model recurrent in time')
# optimization
parser.add_argument('--gamma', type=float, default=0.99,
                    help='discount factor')
parser.add_argument('--tau', type=float, default=1,
                    help='gae (remove?)')
parser.add_argument('--seed', type=int, default=12,
                    help='random seed. Pass -1 for random seed') # TODO: works in thread?
parser.add_argument('--normalize_rewards', action='store_true', default=False,
                    help='normalize rewards in each batch')
parser.add_argument('--lrate', type=float, default=0.007,
                    help='learning rate')
parser.add_argument('--entr', type=float, default=0.99,
                    help='entropy regularization coeff')
parser.add_argument('--value_coeff', type=float, default=0.01,
                    help='coeff for value loss term')
# environment
parser.add_argument('--env_name', default="Cartpole",
                    help='name of the environment to run')
parser.add_argument('--max_steps', default=600, type=int,
                    help='force to end the game after this many steps')
parser.add_argument('--nactions', default='1', type=str,
                    help='the number of agent actions (0 for continuous). Use N:M:K for multiple actions')
parser.add_argument('--action_scale', default=1.0, type=float,
                    help='scale action output from model')
# other
parser.add_argument('--plot', action='store_true', default=False,
                    help='plot training progress')
parser.add_argument('--plot_env', default='main', type=str,
                    help='plot env name')
parser.add_argument('--save', default='', type=str,
                    help='save the model after training')
parser.add_argument('--save_every', default=0, type=int,
                    help='save the model after every n_th epoch')
parser.add_argument('--load', default='', type=str,
                    help='load the model')
parser.add_argument('--display', action="store_true", default=False,
                    help='Display environment state')


parser.add_argument('--random', action='store_true', default=False,
                    help="enable random model")

# CommNet specific args
parser.add_argument('--commnet', action='store_true', default=False,
                    help="enable commnet model")
parser.add_argument('--ic3net', action='store_true', default=False,
                    help="enable commnet model")
parser.add_argument('--nagents', type=int, default=4,
                    help="Number of agents (used in multiagent)")
parser.add_argument('--comm_mode', type=str, default='avg',
                    help="Type of mode for communication tensor calculation [avg|sum]")
parser.add_argument('--comm_passes', type=int, default=1,
                    help="Number of comm passes per step over the model")
parser.add_argument('--comm_mask_zero', action='store_true', default=False,
                    help="Whether communication should be there")
parser.add_argument('--mean_ratio', default=1.0, type=float,
                    help='how much coooperative to do? 1.0 means fully cooperative')
parser.add_argument('--rnn_type', default='MLP', type=str,
                    help='type of rnn to use. [LSTM|MLP]')
parser.add_argument('--detach_gap', default=500, type=int,
                    help='detach hidden state and cell state for rnns at this interval.'
                    + ' Default 10000 (very high)')
parser.add_argument('--comm_init', default='uniform', type=str,
                    help='how to initialise comm weights [uniform|zeros]')
parser.add_argument('--hard_attn', default=False, action='store_true',
                    help='Whether to use hard attention: action - talk|silent')
parser.add_argument('--comm_action_one', default=False, action='store_true',
                    help='Whether to always talk, sanity check for hard attention.')
parser.add_argument('--advantages_per_action', default=False, action='store_true',
                    help='Whether to multipy log porb for each chosen action with advantages')
parser.add_argument('--share_weights', default=False, action='store_true',
                    help='Share weights for hops')


init_args_for_env(parser)
args = parser.parse_args()

if args.ic3net:
    args.commnet = 1
    args.hard_attn = 1
    args.mean_ratio = 0

    # For TJ set comm action to 1 as specified in paper to showcase
    # importance of individual rewards even in cooperative games
    if args.env_name == "traffic_junction":
        args.comm_action_one = True
# Enemy comm
args.nfriendly = args.nagents
if hasattr(args, 'enemy_comm') and args.enemy_comm:
    if hasattr(args, 'nenemies'):
        args.nagents += args.nenemies
    else:
        raise RuntimeError("Env. needs to pass argument 'nenemy'.")

env = data.init(args.env_name, args, False)

num_inputs = max(env.n_s_ls)

args.num_actions = env.n_a_ls[0]



# Multi-action
if not isinstance(args.num_actions, (list, tuple)): # single action case
    args.num_actions = [args.num_actions]


args.dim_actions = 1
args.num_inputs = num_inputs

# Hard attention
if args.hard_attn and args.commnet:
    # add comm_action as last dim in actions
    args.num_actions = [*args.num_actions, 2]
    args.dim_actions = 1 + 1

# Recurrence
if args.commnet and (args.recurrent or args.rnn_type == 'LSTM'):
    args.recurrent = True
    args.rnn_type = 'LSTM'


parse_action_args(args)

if args.seed == -1:
    args.seed = np.random.randint(0,10000)
torch.manual_seed(args.seed)

print(args)


if args.commnet:
    policy_net = CommNetMLP(args, num_inputs)
elif args.random:
    policy_net = Random(args, num_inputs)
elif args.recurrent:
    policy_net = RNN(args, num_inputs)
else:
    policy_net = MLP(args, num_inputs)

if not args.display:
    display_models([policy_net])

# share parameters among threads, but not gradients
for p in policy_net.parameters():
    p.data.share_memory_()



if args.nprocesses > 1:
    trainer = MultiProcessTrainer(args, lambda: Trainer(args, policy_net, data.init(args.env_name, args)))
else:
    trainer = Trainer(args, policy_net, data.init(args.env_name, args))

disp_trainer = Trainer(args, policy_net, data.init(args.env_name, args, False))
disp_trainer.display = True
def disp():
    x = disp_trainer.get_episode()

log = dict()
log['epoch'] = LogField(list(), False, None, None)
log['reward'] = LogField(list(), True, 'epoch', 'num_episodes')
log['enemy_reward'] = LogField(list(), True, 'epoch', 'num_episodes')
log['success'] = LogField(list(), True, 'epoch', 'num_episodes')
log['steps_taken'] = LogField(list(), True, 'epoch', 'num_episodes')
log['add_rate'] = LogField(list(), True, 'epoch', 'num_episodes')
log['comm_action'] = LogField(list(), True, 'epoch', 'num_steps')
log['enemy_comm'] = LogField(list(), True, 'epoch', 'num_steps')
log['value_loss'] = LogField(list(), True, 'epoch', 'num_steps')
log['action_loss'] = LogField(list(), True, 'epoch', 'num_steps')
log['entropy'] = LogField(list(), True, 'epoch', 'num_steps')

if args.plot:
    vis = visdom.Visdom(env=args.plot_env)




def _log_episode(global_step, mean_reward, std_reward):
    log = {'agent': 'ic3net',
           'step': global_step,
           'test_id': -1,
           'avg_reward': mean_reward,
           'std_reward': std_reward}
    csv_data.append(log)


def run(num_epochs):

    global_step =0
    for ep in range(num_epochs):
        epoch_begin_time = time.time()
        stat = dict()
        for n in range(args.epoch_size):
            if n == args.epoch_size - 1 and args.display:
                trainer.display = False
            s, step_taken, avg_reward, std_reward = trainer.train_batch(ep)
            merge_stat(s, stat)
            trainer.display = False
        global_step += step_taken
        if global_step > int(1001000):
            break
        epoch_time = time.time() - epoch_begin_time
        epoch = len(log['epoch'].data) + 1
        _log_episode(global_step, avg_reward, std_reward)
        '''
        for k, v in log.items():
            if k == 'epoch':
                v.data.append(epoch)
            else:
                if k in stat and v.divide_by is not None and stat[v.divide_by] > 0:
                    stat[k] = stat[k] / stat[v.divide_by]
                v.data.append(stat.get(k, 0))
        '''
        np.set_printoptions(precision=2)

        #print('Epoch {}\tReward {}\tTime {:.2f}s'.format(
       #         epoch, stat['reward'], epoch_time
       #))
        print(ep,'---',avg_reward)
        if 'enemy_reward' in stat.keys():
            pass
            #print('Enemy-Reward: {}'.format(stat['enemy_reward']))
        if 'add_rate' in stat.keys():
            pass
            #print('Add-Rate: {:.2f}'.format(stat['add_rate']))
        if 'success' in stat.keys():
            pass
            #print('Success: {:.2f}'.format(stat['success']))
        if 'steps_taken' in stat.keys():
            pass
            #print('Steps-taken: {:.2f}'.format(stat['steps_taken']))
        if 'comm_action' in stat.keys():
            pass
            #print('Comm-Action: {}'.format(stat['comm_action']))
        if 'enemy_comm' in stat.keys():
            pass
            #print('Enemy-Comm: {}'.format(stat['enemy_comm']))

        if args.plot:
            for k, v in log.items():
                if v.plot and len(v.data) > 0:
                    vis.line(np.asarray(v.data), np.asarray(log[v.x_axis].data[-len(v.data):]),
                    win=k, opts=dict(xlabel=v.x_axis, ylabel=k))

        if args.save_every and ep and args.save != '' and ep % args.save_every == 0:
            # fname, ext = args.save.split('.')
            # save(fname + '_' + str(ep) + '.' + ext)
            save(args.save + '_' + str(ep))

        if args.save != '':
            save(args.save)
    # save_dir = './results/{}/'.format(datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
    # model_dir = './models/{}/'.format(datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
    save_dir = './results/{}_{}/{}/'.format(args.env_name, args.nagents, datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
    model_dir = './models/{}_{}/{}/'.format(args.env_name, args.nagents, datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    df = pd.DataFrame(csv_data)
    #print(df)
    df.to_csv(save_dir+'train_reward.csv')

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    save(model_dir + 'model.pt')
    time.sleep(2)

def save(path):
    d = dict()
    d['policy_net'] = policy_net.state_dict()
    d['log'] = log
    d['trainer'] = trainer.state_dict()
    torch.save(d, path)


def load(path):
    d = torch.load(path)
    # log.clear()
    policy_net.load_state_dict(d['policy_net'])
    log.update(d['log'])
    trainer.load_state_dict(d['trainer'])

#load('/home/liubo/Desktop/ICLR2021/IC3net_Traffic/IC3Net/models/2021_01_24_14_54_43/model.pt')
def signal_handler(signal, frame):
        print('You pressed Ctrl+C! Exiting gracefully.')
        if args.display:
            env.end_display()
        sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

if args.load != '':
    load(args.load)

run(args.num_epochs)
# run(10)
if args.display:
    env.end_display()

if args.save != '':
    save(args.save)

if sys.flags.interactive == 0 and args.nprocesses > 1:
    trainer.quit()
    import os
    os._exit(0)
