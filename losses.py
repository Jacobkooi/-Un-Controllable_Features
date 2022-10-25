import torch
import numpy as np
import random


def compute_entropy_loss(agent):

    random_states, _, _, _, _ = agent.buffer.sample(agent.batch_size)
    random_states = agent.encoder(random_states)
    random_states_rolled = torch.roll(random_states, 1, dims=0)
    difference_states = random_states-random_states_rolled

    # normal random states loss
    loss = torch.exp(-agent.entropy_scaler * torch.norm(difference_states, dim=1, p=2)).mean()

    return loss


def compute_entropy_loss_featuremaps_randompixels(agent, STATE):

    random_states_agent, random_features_wall = agent.encoder(STATE)

    id = torch.tensor(random.sample(range(len(random_features_wall[0].flatten(0))), agent.feature_entropy_int))
    average_wall_features_array = random_features_wall.flatten(1)[:, id]

    random_states_agent_rolled = torch.roll(random_states_agent, 1, dims=0)
    average_wall_features_array_rolled = torch.roll(average_wall_features_array, 1, dims=0)

    concatenated_states = torch.cat((random_states_agent, average_wall_features_array), dim=1)
    concatenated_states_rolled = torch.cat((random_states_agent_rolled, average_wall_features_array_rolled), dim=1)
    difference_states = concatenated_states - concatenated_states_rolled
    loss = torch.exp(-agent.entropy_scaler * torch.norm(difference_states, dim=1, p=2)).mean()

    return loss


def compute_entropy_loss_trajectory(agent):

    trajectory = agent.buffer.sample_trajectory()[0]
    encoded_trajectory = agent.encoder(trajectory)[0]

    # Roll the states uniformly random so we eventually get all combinations
    encoded_trajectory_rolled = torch.roll(encoded_trajectory, np.random.randint(low=1, high=len(encoded_trajectory-1)), dims=0)

    loss = torch.exp(-agent.entropy_scaler * torch.norm(encoded_trajectory_rolled - encoded_trajectory, dim=1, p=2)).mean()

    return loss


def compute_entropy_loss_subsequent_states(agent, STATE, NEXT_STATE):

    random_states_agent, _ = agent.encoder(STATE)
    next_random_states_agent, _ = agent.encoder(NEXT_STATE)

    difference_states = next_random_states_agent - random_states_agent
    loss = torch.exp(-agent.entropy_scaler * torch.norm(difference_states, dim=1, p=2)).mean()

    return loss


