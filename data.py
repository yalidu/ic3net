import sys
import gym

from env_wrappers import *
import pandas as pd
import configparser
import logging
import numpy as np
from cacc_env import CACCEnv


def init(env_name, args, final_init=True):
    if env_name == 'levers':
        env = gym.make('Levers-v0')
        env.multi_agent_init(args.total_agents, args.nagents)
        env = GymWrapper(env)
    elif env_name == 'number_pairs':
        env = gym.make('NumberPairs-v0')
        m = args.max_message
        env.multi_agent_init(args.nagents, m)
        env = GymWrapper(env)
    elif env_name == 'predator_prey':
        env = gym.make('PredatorPrey-v0')
        if args.display:
            env.init_curses()
        env.multi_agent_init(args)
        env = GymWrapper(env)
    elif env_name == 'traffic_junction':
        env = gym.make('TrafficJunction-v0')
        if args.display:
            env.init_curses()
        env.multi_agent_init(args)
        env = GymWrapper(env)
    elif env_name == 'starcraft':
        env = gym.make('StarCraftWrapper-v0')
        env.multi_agent_init(args, final_init)
        env = GymWrapper(env.env)


    elif env_name == 'cacc_catchup':
        output_path = '/home/liubo/Desktop/ICLR2021/IC3net_Traffic/IC3Net/test_temp_output/'
        config_path = '/home/liubo/Desktop/ICLR2021/IC3net_Traffic/IC3Net/config/config_ia2c_catchup.ini'
        config = configparser.ConfigParser()
        config.read(config_path)
        env = CACCEnv(config['ENV_CONFIG'])
        env.init_data(True, False, output_path)

    elif env_name == 'cacc_slowdown':
        output_path = './temp_output/'
        config_path = './config/config_ia2c_slowdown.ini'
        config = configparser.ConfigParser()
        config.read(config_path)
        env = CACCEnv(config['ENV_CONFIG'])
        env.init_data(True, False, output_path)


    else:
        raise RuntimeError("wrong env name")

    return env
