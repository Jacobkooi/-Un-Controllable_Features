import torch.nn as nn
import torch


class EncoderCRAR(nn.Module):
    """Convolutional encoder_C for image-based observations."""
    def __init__(self, obs_channels, latent_dim, tanh=False, scale=1):
        super().__init__()

        self.latent_dim = latent_dim
        self.obs_channels = obs_channels
        self.outputs = dict()
        self.tanh = tanh
        self.scale = scale

        self.convs = nn.Sequential(
            nn.Conv2d(self.obs_channels, out_channels=8, kernel_size=(2, 2), stride=(1, 1)),
            nn.Tanh(),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(2, 2), stride=(1, 1)),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=(1, 1)),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=(3, 3)),
            nn.Conv2d(in_channels=32, out_channels=2, kernel_size=(1, 1))
        )

        self.dummy_input = torch.ones((1, 1, 48, 48))
        self.dummy_output = self.convs(self.dummy_input)
        self.fc_size = int(self.dummy_output.flatten(1).shape[1]/2)
        self.wall_output = self.dummy_output[:, 1]
        self.wall_size = self.wall_output.shape[2] ** 2

        self.mlp = nn.Sequential(
            nn.Linear(in_features=self.fc_size, out_features=int(self.latent_dim)),
            nn.Tanh(),
        )

    def forward(self, obs, detach=False):

        if len(obs.shape) == 3:
            obs = obs.unsqueeze(1)
        features = self.convs(obs)

        agent_features = features[:, 0]
        wall_features = features[:, 1].unsqueeze(1)

        agent_features = agent_features.flatten(1)

        agent_latent = self.mlp(agent_features)

        return agent_latent, wall_features

class Averagepool8x8featuremap(nn.Module):
    """Convolutional encoder_C for image-based observations."""
    def __init__(self, kernel_size=2, stride=1):
        super().__init__()

        self.maxpool = nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=0)
        # self.maxpool = nn.FractionalMaxPool2d(kernel_size=kernel_size, output_size=3)
        # self.maxpool = nn.FractionalMaxPool2d(kernel_size=kernel_size, output_size=4)

    def forward(self, obs):

        pooled_values_flattened = self.maxpool(obs).flatten(1)

        return pooled_values_flattened


class Maxpool8x8featuremap(nn.Module):
    """Convolutional encoder_C for image-based observations."""
    def __init__(self, kernel_size=2, stride=1):
        super().__init__()

        self.maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=0)
        # self.maxpool = nn.FractionalMaxPool2d(kernel_size=kernel_size, output_size=3)
        # self.maxpool = nn.FractionalMaxPool2d(kernel_size=kernel_size, output_size=4)

    def forward(self, obs):

        pooled_values_flattened = self.maxpool(obs).flatten(1)

        return pooled_values_flattened



class EncoderDRQ(nn.Module):
    """Convolutional encoder_C for image-based observations."""
    def __init__(self, obs_channels, latent_dim, tanh=False, scale=1):
        super().__init__()

        self.latent_dim = latent_dim
        self.obs_channels = obs_channels
        self.outputs = dict()
        self.tanh = tanh
        self.scale = scale

        self.convs = nn.Sequential(
            nn.Conv2d(self.obs_channels, out_channels=32, kernel_size=(3, 3), stride=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(),
        )

        self.mlp = nn.Sequential(
            nn.Linear(in_features=9248, out_features=self.latent_dim),
            nn.Tanh()
            )
        if self.tanh:
            self.mlp.add_module("tanh_activation", nn.Tanh())

    def forward(self, obs, detach=False):

        if len(obs.shape) == 3:
            obs = obs.unsqueeze(1)
        features = self.convs(obs)

        if detach:
            features = features.detach()

        features = features.flatten(1)

        latent = self.mlp(features)

        self.outputs['output_MlP'] = latent

        return latent


class EncoderDMC(nn.Module):
    """Convolutional encoder_C for image-based observations."""
    def __init__(self, latent_dim, scale=1):
        super().__init__()

        self.latent_dim = latent_dim
        self.obs_channels = 1
        self.scale = scale

        self.convs = nn.Sequential(
            nn.Conv2d(self.obs_channels, out_channels=32, kernel_size=(3, 3), stride=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(),
        )

        self.mlp = nn.Sequential(
            nn.Linear(in_features=14112, out_features=self.latent_dim),
            nn.Tanh())

    def forward(self, obs, detach=False):

        if len(obs.shape) == 3:
            obs = obs.unsqueeze(1)
        features = self.convs(obs)

        features = features.flatten(1)

        latent = self.mlp(features)

        if detach:
            latent = latent.detach()

        return latent


class EncoderDMC_Catcher(nn.Module):
    """Convolutional encoder_C for image-based observations."""
    def __init__(self, obs_channels, latent_dim, tanh=False, scale=1):
        super().__init__()

        self.latent_dim = latent_dim
        self.obs_channels = obs_channels
        self.outputs = dict()
        self.tanh = tanh
        self.scale = scale

        self.convs = nn.Sequential(
            nn.Conv2d(self.obs_channels, out_channels=32, kernel_size=(3, 3), stride=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(),
        )

        self.mlp = nn.Sequential(
            nn.Linear(in_features=7200, out_features=self.latent_dim),
            nn.Tanh())
        if self.tanh:
            self.mlp.add_module("tanh_activation", nn.Tanh())

    def forward(self, obs, detach=False):

        if len(obs.shape) == 3:
            obs = obs.unsqueeze(1)
        features = self.convs(obs)

        if detach:
            features = features.detach()

        features = features.flatten(1)

        latent = self.mlp(features)

        self.outputs['output_MlP'] = latent

        return latent


class EncoderDMC_Catcher_8x8(nn.Module):
    """Convolutional encoder_C for image-based observations."""
    def __init__(self, obs_channels, scale=1):
        super().__init__()

        self.obs_channels = obs_channels
        self.outputs = dict()

        self.convs = nn.Sequential(
            nn.Conv2d(self.obs_channels, out_channels=int(32/scale), kernel_size=(3, 3), stride=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(in_channels=int(32/scale)
                      , out_channels=int(32/scale), kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(),
        )

        self.conv_paddle = nn.Sequential(
            nn.Conv2d(in_channels=int(32/scale), out_channels=1, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(),
        )

        self.conv_ball = nn.Sequential(
            nn.Conv2d(in_channels=int(32/scale), out_channels=1, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(),
        )

    def forward(self, obs, detach=False):

        if len(obs.shape) == 3:
            obs = obs.unsqueeze(1)
        features = self.convs(obs)

        self.outputs['features'] = features

        if detach:
            features = features.detach()

        paddle_features = self.conv_paddle(features)
        ball_features = self.conv_ball(features)

        return torch.cat((paddle_features, ball_features), dim=1)


class EncoderDMC_8x8wall(nn.Module):
    """Convolutional encoder_C for image-based observations."""
    def __init__(self, obs_channels, latent_dim, tanh=False, scale=16, neuron_dim=100):
        super().__init__()

        self.latent_dim = latent_dim
        self.obs_channels = obs_channels
        self.outputs = dict()
        self.tanh = tanh
        self.scale = scale

        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=self.obs_channels, out_channels=int(32), kernel_size=(3, 3), stride=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(in_channels=int(32), out_channels=int(32), kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(),
        )

        self.conv_walls = nn.Sequential(
            nn.Conv2d(in_channels=int(32), out_channels=int(1), kernel_size=(4, 4), stride=(1, 1)),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(output_size=6),  # Todo

        )

        self.dummy_input = torch.ones((1, 1, 48, 48))
        self.dummy_output = self.convs(self.dummy_input)
        self.fc_size = self.dummy_output.flatten(1).shape[1]
        self.wall_output = self.conv_walls(self.dummy_output)
        self.wall_size = self.wall_output.shape[2]**2

        self.mlp = nn.Sequential(
            nn.Linear(in_features=self.fc_size, out_features=neuron_dim),
            nn.Tanh(),
            nn.Linear(in_features=neuron_dim, out_features=self.latent_dim),
            nn.Tanh())

    def forward(self, obs, detach=''):

        if len(obs.shape) == 3:
            obs = obs.unsqueeze(1)
        features = self.convs(obs)

        if detach:
            if detach =='base':
                features = features.detach()


        wall_features = self.conv_walls(features)
        self.outputs['wall_features'] = wall_features

        features = features.flatten(1)

        latent_agent = self.mlp(features)

        self.outputs['output_MlP'] = latent_agent

        if detach:
            if detach =='wall':
                wall_features = wall_features.detach()
            elif detach =='agent':
                latent_agent = latent_agent.detach()
            elif detach =='both':
                wall_features = wall_features.detach()
                latent_agent = latent_agent.detach()

        return latent_agent, wall_features


class EncoderDMC_Atari(nn.Module):
    """Convolutional encoder_C for image-based observations."""
    def __init__(self, obs_channels, latent_dim, tanh=False, scale=16, neuron_dim=200, wall_dim=14, pixel_dim=84):
        super().__init__()

        self.latent_dim = latent_dim
        self.obs_channels = obs_channels
        self.outputs = dict()
        self.tanh = tanh
        self.scale = scale

        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=self.obs_channels, out_channels=int(32), kernel_size=(3, 3), stride=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(in_channels=int(32), out_channels=int(32), kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(),
        )

        self.conv_walls = nn.Sequential(
            nn.Conv2d(in_channels=int(32), out_channels=int(32), kernel_size=(4, 4), stride=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=int(32), out_channels=int(1), kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(output_size=wall_dim),  # Todo

        )

        self.dummy_input = torch.ones((1, 1, pixel_dim, pixel_dim))
        self.dummy_output = self.convs(self.dummy_input)
        self.fc_size = self.dummy_output.flatten(1).shape[1]
        self.wall_output = self.conv_walls(self.dummy_output)
        self.wall_size = self.wall_output.shape[2]**2

        self.mlp = nn.Sequential(
            nn.Linear(in_features=self.fc_size, out_features=neuron_dim),
            nn.Tanh(),
            nn.Linear(in_features=neuron_dim, out_features=self.latent_dim),
            nn.Tanh())

    def forward(self, obs, detach=''):

        if len(obs.shape) == 3:
            obs = obs.unsqueeze(1)
        features = self.convs(obs)

        if detach:
            if detach =='base':
                features = features.detach()


        wall_features = self.conv_walls(features)
        self.outputs['wall_features'] = wall_features

        features = features.flatten(1)

        latent_agent = self.mlp(features)

        self.outputs['output_MlP'] = latent_agent

        if detach:
            if detach =='wall':
                wall_features = wall_features.detach()
            elif detach =='agent':
                latent_agent = latent_agent.detach()
            elif detach =='both':
                wall_features = wall_features.detach()
                latent_agent = latent_agent.detach()

        return latent_agent, wall_features


class EncoderDMC_8x8wall_old(nn.Module):
    """Convolutional encoder_C for image-based observations."""
    def __init__(self, obs_channels, latent_dim, tanh=False, scale=1):
        super().__init__()

        self.latent_dim = latent_dim
        self.obs_channels = obs_channels
        self.outputs = dict()
        self.tanh = tanh
        self.scale = scale

        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=self.obs_channels, out_channels=int(32/scale), kernel_size=(3, 3), stride=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(in_channels=int(32), out_channels=int(32), kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(),
        )

        self.conv_walls = nn.Sequential(
            # nn.Conv2d(in_channels=int(32), out_channels=int(32), kernel_size=(2, 2), stride=(1, 1)),
            # nn.ReLU(),
            nn.Conv2d(in_channels=int(32), out_channels=int(1), kernel_size=(4, 4), stride=(1, 1)),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(output_size=6),  # Todo

        )

        self.dummy_input = torch.ones((1, 1, 48, 48))
        self.dummy_output = self.convs(self.dummy_input)
        self.fc_size = self.dummy_output.flatten(1).shape[1]
        self.wall_output = self.conv_walls(self.dummy_output)
        self.wall_size = self.wall_output.shape[2]**2
        self.mlp = nn.Sequential(
            nn.Linear(in_features=self.fc_size, out_features=self.latent_dim),
            nn.Tanh())

    def forward(self, obs, detach=''):

        if len(obs.shape) == 3:
            obs = obs.unsqueeze(1)
        features = self.convs(obs)

        if detach:
            if detach =='base':
                features = features.detach()


        wall_features = self.conv_walls(features)
        self.outputs['wall_features'] = wall_features

        features = features.flatten(1)

        latent_agent = self.mlp(features)

        self.outputs['output_MlP'] = latent_agent

        if detach:
            if detach =='wall':
                wall_features = wall_features.detach()
            elif detach =='agent':
                latent_agent = latent_agent.detach()
            elif detach =='both':
                wall_features = wall_features.detach()
                latent_agent = latent_agent.detach()

        return latent_agent, wall_features


class EncoderDMC_8x8wall_reward(nn.Module):
    """Convolutional encoder_C for image-based observations."""
    def __init__(self, obs_channels, latent_dim, tanh=False, scale=1):
        super().__init__()

        self.latent_dim = latent_dim
        self.obs_channels = obs_channels
        self.outputs = dict()
        self.tanh = tanh
        self.scale = scale

        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=self.obs_channels, out_channels=int(32/scale), kernel_size=(3, 3), stride=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(in_channels=int(32/scale), out_channels=int(32/scale), kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(),
        )

        self.conv_walls = nn.Sequential(
            nn.Conv2d(in_channels=int(32/scale), out_channels=int(8/scale), kernel_size=(3, 3), stride=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(in_channels=int(8/scale), out_channels=1, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(),
        )

        self.conv_reward = nn.Sequential(
            nn.Conv2d(in_channels=int(32/scale), out_channels=int(8/scale), kernel_size=(3, 3), stride=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(in_channels=int(8/scale), out_channels=1, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(),
        )

        self.dummy_input = torch.ones((1, 1, 48, 48))
        self.dummy_output = self.convs(self.dummy_input)
        self.fc_size = self.dummy_output.flatten(1).shape[1]
        self.wall_output = self.conv_walls(self.dummy_output)
        self.wall_size = self.wall_output.shape[2]**2
        self.reward_size = self.wall_size
        self.mlp = nn.Sequential(
            nn.Linear(in_features=self.fc_size, out_features=self.latent_dim),
            nn.Tanh())

    def forward(self, obs, detach=''):

        if len(obs.shape) == 3:
            obs = obs.unsqueeze(1)
        features = self.convs(obs)

        if detach:
            if detach =='base':
                features = features.detach()


        wall_features = self.conv_walls(features)
        self.outputs['wall_features'] = wall_features

        reward_features = self.conv_reward(features)

        features = features.flatten(1)

        latent_agent = self.mlp(features)

        self.outputs['output_MlP'] = latent_agent

        if detach:
            if detach =='wall':
                wall_features = wall_features.detach()
            elif detach =='agent':
                latent_agent = latent_agent.detach()
            elif detach =='reward':
                reward_features = reward_features.detach()
            elif detach =='both':
                wall_features = wall_features.detach()
                latent_agent = latent_agent.detach()

        return latent_agent, wall_features, reward_features


class EncoderDMC_8x8wall_2channel(nn.Module):
    """Convolutional encoder_C for image-based observations."""
    def __init__(self, obs_channels, latent_dim, tanh=False, scale=1):
        super().__init__()

        self.latent_dim = latent_dim
        self.obs_channels = obs_channels
        self.outputs = dict()
        self.tanh = tanh
        self.scale = scale

        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=self.obs_channels, out_channels=int(32/scale), kernel_size=(3, 3), stride=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(in_channels=int(32/scale), out_channels=int(32), kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(),
        )

        self.conv_walls = nn.Sequential(
            # nn.Conv2d(in_channels=int(32 / scale), out_channels=int(2), kernel_size=(5, 5), stride=(1, 1)),
            # nn.ReLU(),
            nn.AdaptiveAvgPool2d(output_size=8)

            # nn.Conv2d(in_channels=int(8/scale), out_channels=2, kernel_size=(3, 3), stride=(1, 1)),
            # nn.ReLU(),
        )

        self.dummy_input = torch.ones((1, 1, 48, 48))
        self.dummy_output = self.convs(self.dummy_input)
        self.fc_size = int(self.dummy_output.flatten(1).shape[1]*(30/32))
        self.wall_output = self.conv_walls(self.dummy_output)
        self.wall_size = self.wall_output.shape[2]**2
        self.mlp = nn.Sequential(
            nn.Linear(in_features=self.fc_size, out_features=self.latent_dim),
            nn.Tanh())

    def forward(self, obs, detach=''):

        if len(obs.shape) == 3:
            obs = obs.unsqueeze(1)
        features = self.convs(obs)

        if detach:
            if detach =='base':
                features = features.detach()

        features = torch.split(features, 30, dim=1)
        agent_features = features[0]
        wall_features = features[1][:, [0]]
        reward_features = features[1][:, [1]]

        wall_features = self.conv_walls(wall_features)
        reward_features = self.conv_walls(reward_features)

        agent_features = agent_features.flatten(1)

        latent_agent = self.mlp(agent_features)

        if detach:
            if detach =='wall':
                wall_features = wall_features.detach()
            elif detach =='agent':
                latent_agent = latent_agent.detach()
            elif detach =='both':
                wall_features = wall_features.detach()
                latent_agent = latent_agent.detach()

        return latent_agent, wall_features, reward_features


class EncoderDMC_8x8wall_2channeltest2(nn.Module):
    """Convolutional encoder_C for image-based observations."""
    def __init__(self, obs_channels, latent_dim, tanh=False, scale=1, final_tanh=False):
        super().__init__()

        self.latent_dim = latent_dim
        self.obs_channels = obs_channels
        self.outputs = dict()
        self.tanh = tanh
        self.scale = scale

        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=self.obs_channels, out_channels=int(32/scale), kernel_size=(3, 3), stride=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(in_channels=int(32/scale), out_channels=int(32), kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(),
        )

        self.conv_walls = nn.Sequential(
            # nn.Conv2d(in_channels=int(32 / scale), out_channels=int(2), kernel_size=(5, 5), stride=(1, 1)),
            # nn.ReLU(),
            nn.AdaptiveAvgPool2d(output_size=8)

            # nn.Conv2d(in_channels=int(8/scale), out_channels=2, kernel_size=(3, 3), stride=(1, 1)),
            # nn.ReLU(),
        )

        self.conv_agent = nn.Sequential(
            nn.Conv2d(in_channels=30, out_channels=int(32/scale), kernel_size=(3, 3), stride=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(in_channels=int(32/scale), out_channels=int(16), kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=int(16), out_channels=int(8), kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(),
        )

        self.dummy_input = torch.ones((1, 1, 48, 48))
        self.dummy_output = self.convs(self.dummy_input)
        self.dummy_agent_input = torch.split(self.dummy_output, 30, 1)[0]
        self.dummy_agent_output = self.conv_agent(self.dummy_agent_input)

        self.fc_size = int(self.dummy_agent_output.flatten(1).shape[1])
        self.wall_output = self.conv_walls(self.dummy_output)
        self.wall_size = self.wall_output.shape[2]**2
        self.mlp = nn.Sequential(
            nn.Linear(in_features=self.fc_size, out_features=self.latent_dim),
        )

        if final_tanh:
            self.mlp.add_module("tanh_activation", nn.Tanh())

    def forward(self, obs, detach=''):

        if len(obs.shape) == 3:
            obs = obs.unsqueeze(1)
        features = self.convs(obs)

        if detach:
            if detach =='base':
                features = features.detach()

        features = torch.split(features, 30, dim=1)
        agent_features = features[0]
        wall_features = features[1][:, [0]]
        reward_features = features[1][:, [1]]

        wall_features = self.conv_walls(wall_features)
        reward_features = self.conv_walls(reward_features)

        agent_features = self.conv_agent(agent_features)

        agent_features = agent_features.flatten(1)

        latent_agent = self.mlp(agent_features)

        if detach:
            if detach =='wall':
                wall_features = wall_features.detach()
            elif detach =='agent':
                latent_agent = latent_agent.detach()
            elif detach =='both':
                wall_features = wall_features.detach()
                latent_agent = latent_agent.detach()

        return latent_agent, wall_features, reward_features


class Encoder_reward_in_controllable(nn.Module):
    """Convolutional encoder_C for image-based observations."""
    def __init__(self, obs_channels, latent_dim, tanh=False, scale=1, wall_dim=6):
        super().__init__()

        self.latent_dim = latent_dim
        self.obs_channels = obs_channels
        self.outputs = dict()
        self.tanh = tanh
        self.scale = scale

        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=self.obs_channels, out_channels=int(32), kernel_size=(3, 3), stride=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(in_channels=int(32), out_channels=int(32), kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=int(32), out_channels=int(32), kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=int(32), out_channels=int(3), kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(), # Channel 1: Agent. Channel 2: Reward. Channel 3: Walls.
        )

        self.averagepool = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=wall_dim)
        )

        self.dummy_input = torch.ones((1, 1, 48, 48))
        self.dummy_output = torch.split(self.convs(self.dummy_input), 1, dim=1)[0]
        self.fc_size = int(self.dummy_output.flatten(1).shape[1])
        self.wall_output = self.averagepool(self.dummy_output)
        self.wall_size = self.wall_output.shape[2]**2

    def forward(self, obs, detach=''):

        if len(obs.shape) == 3:
            obs = obs.unsqueeze(1)
        features = self.convs(obs)

        if detach:
            if detach =='base':
                features = features.detach()

        features = torch.split(features, 1, dim=1)
        agent_features = features[0]
        reward_features = features[1]
        wall_features = features[2]

        agent_features = self.averagepool(agent_features)
        reward_features = self.averagepool(reward_features)
        wall_features = self.averagepool(wall_features)

        return agent_features, wall_features, reward_features


class Encoder_pathfinding(nn.Module):
    """Convolutional encoder_C for image-based observations."""
    def __init__(self, obs_channels, latent_dim, tanh=False, scale=1, final_tanh=False):
        super().__init__()

        self.latent_dim = latent_dim
        self.obs_channels = obs_channels
        self.outputs = dict()
        self.tanh = tanh
        self.scale = scale

        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=self.obs_channels, out_channels=int(32), kernel_size=(3, 3), stride=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(in_channels=int(32), out_channels=int(32), kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=int(32), out_channels=int(32), kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=int(32), out_channels=int(2), kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(), # Channel 1: Agent. Channel 2: Walls
        )

        self.averagepool = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=8)
        )

        self.dummy_input = torch.ones((1, 1, 48, 48))
        self.dummy_output = torch.split(self.convs(self.dummy_input), 1, dim=1)[0]
        self.fc_size = int(self.dummy_output.flatten(1).shape[1])
        self.wall_output = self.averagepool(self.dummy_output)
        self.wall_size = self.wall_output.shape[2]**2
        self.mlp = nn.Sequential(
            nn.Linear(in_features=self.fc_size, out_features=self.latent_dim),
        )

        if final_tanh:
            self.mlp.add_module("tanh_activation", nn.Tanh())

    def forward(self, obs, detach=''):

        if len(obs.shape) == 3:
            obs = obs.unsqueeze(1)
        features = self.convs(obs)

        features = torch.split(features, 1, dim=1)
        agent_features = features[0]
        wall_features = features[2]

        wall_features = self.averagepool(wall_features)

        agent_features = agent_features.flatten(1)

        latent_agent = self.mlp(agent_features)

        return latent_agent, wall_features


class EncoderDMC_Controllable_reward(nn.Module):
    """Convolutional encoder_C for image-based observations."""
    def __init__(self, obs_channels, latent_dim, tanh=False, scale=1, output_size=8):
        super().__init__()

        self.latent_dim = latent_dim
        self.obs_channels = obs_channels
        self.outputs = dict()
        self.tanh = tanh
        self.scale = scale
        self.output_size = output_size

        self.base_convs = nn.Sequential(
            nn.Conv2d(in_channels=self.obs_channels, out_channels=int(32), kernel_size=(3, 3), stride=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(in_channels=int(32), out_channels=int(32), kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(),
        )

        self.conv_walls = nn.Sequential(
            nn.Conv2d(in_channels=self.obs_channels, out_channels=int(32), kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=int(32), out_channels=int(1), kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(output_size=self.output_size)

            # nn.Conv2d(in_channels=int(8/scale), out_channels=2, kernel_size=(3, 3), stride=(1, 1)),
            # nn.ReLU(),
        )

        self.conv_agent = nn.Sequential(
            nn.Conv2d(in_channels=self.obs_channels, out_channels=int(32), kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=int(32), out_channels=int(1), kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(output_size=self.output_size)
        )

        self.wall_size = self.output_size**2

    def forward(self, obs, detach=''):

        if len(obs.shape) == 3:
            obs = obs.unsqueeze(1)
        base_features = self.base_convs(obs)

        if detach:
            if detach =='base':
                base_features = base_features.detach()

        agent_features = self.conv_agent(base_features)
        wall_features = self.conv_walls(base_features)

        return agent_features, wall_features


class EncoderDMC_8x8wall_2channel_test(nn.Module):
    """Convolutional encoder_C for image-based observations."""
    def __init__(self, obs_channels, latent_dim, tanh=False, scale=1):
        super().__init__()

        self.latent_dim = latent_dim
        self.obs_channels = obs_channels
        self.outputs = dict()
        self.tanh = tanh
        self.scale = scale

        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=self.obs_channels, out_channels=int(32/scale), kernel_size=(3, 3), stride=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=int(32/scale), out_channels=int(3), kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(),
        )

        self.wallpool = nn.AdaptiveAvgPool2d(output_size=8)

        self.dummy_input = torch.ones((1, 1, 48, 48))
        self.dummy_output = self.convs(self.dummy_input)
        self.fc_size = int((self.dummy_output.flatten(1).shape[1])/3)
        self.wall_size = 8**2
        self.mlp = nn.Sequential(
            nn.Linear(in_features=self.fc_size, out_features=self.latent_dim),
            nn.Tanh())

    def forward(self, obs, detach=''):

        if len(obs.shape) == 3:
            obs = obs.unsqueeze(1)
        features = self.convs(obs)
        if detach:
            if detach =='base':
                features = features.detach()

        features = torch.split(features, 1, dim=1)
        agent_features = features[0]
        wall_features = features[1]
        reward_features = features[2]

        # TODO Try maxpool on agent features?

        wall_features = self.wallpool(wall_features)
        reward_features = self.wallpool(reward_features)
        latent_agent = self.mlp(agent_features.flatten(1))

        if detach:
            if detach =='wall':
                wall_features = wall_features.detach()
            elif detach =='agent':
                latent_agent = latent_agent.detach()
            elif detach =='both':
                wall_features = wall_features.detach()
                latent_agent = latent_agent.detach()

        return latent_agent, wall_features, reward_features



class EncoderDMC_8x8wall_reward_DQN(nn.Module):
    """Convolutional encoder_C for image-based observations."""
    def __init__(self, obs_channels, latent_dim, tanh=False, scale=1):
        super().__init__()

        self.latent_dim = latent_dim
        self.obs_channels = obs_channels
        self.outputs = dict()
        self.tanh = tanh
        self.scale = scale

        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=self.obs_channels, out_channels=int(32/scale), kernel_size=(3, 3), stride=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(in_channels=int(32/scale), out_channels=int(32/scale), kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(),
        )

        self.conv_walls = nn.Sequential(
            nn.Conv2d(in_channels=int(32/scale), out_channels=int(8/scale), kernel_size=(3, 3), stride=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(in_channels=int(8/scale), out_channels=2, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(),
        )

        self.dummy_input = torch.ones((1, 1, 48, 48))
        self.dummy_output = self.convs(self.dummy_input)
        self.fc_size = self.dummy_output.flatten(1).shape[1]
        self.wall_output = self.conv_walls(self.dummy_output)
        self.wall_size = self.wall_output.shape[2]**2
        self.mlp = nn.Sequential(
            nn.Linear(in_features=self.fc_size, out_features=self.latent_dim),
            nn.Tanh())

    def forward(self, obs, detach=''):

        if len(obs.shape) == 3:
            obs = obs.unsqueeze(1)
        features = self.convs(obs)

        if detach:
            if detach =='base':
                features = features.detach()


        wall_features = self.conv_walls(features)
        self.outputs['wall_features'] = wall_features

        features = features.flatten(1)

        latent_agent = self.mlp(features)

        self.outputs['output_MlP'] = latent_agent

        if detach:
            if detach =='wall':
                wall_features = wall_features.detach()
            elif detach =='agent':
                latent_agent = latent_agent.detach()
            elif detach =='both':
                wall_features = wall_features.detach()
                latent_agent = latent_agent.detach()

        return latent_agent, wall_features


class EncoderDMC_8x8wall_maxpooled(nn.Module):
    """Convolutional encoder_C for image-based observations."""
    def __init__(self, obs_channels, latent_dim, tanh=False, scale=1):
        super().__init__()

        self.latent_dim = latent_dim
        self.obs_channels = obs_channels
        self.outputs = dict()
        self.tanh = tanh
        self.scale = scale

        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=self.obs_channels, out_channels=int(32/scale), kernel_size=(3, 3), stride=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(in_channels=int(32/scale), out_channels=int(32/scale), kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(),
        )

        self.conv_walls = nn.Sequential(
            nn.Conv2d(in_channels=int(32/scale), out_channels=8, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=int(8), out_channels=1, kernel_size=(3, 3), stride=(2, 2)),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d(output_size=6)
        )

        self.dummy_input = torch.ones((1, 1, 48, 48))
        self.dummy_output = self.convs(self.dummy_input)
        self.fc_size = self.dummy_output.flatten(1).shape[1]
        self.wall_output = self.conv_walls(self.dummy_output)
        self.wall_size = self.wall_output.shape[2]**2
        self.mlp = nn.Sequential(
            nn.Linear(in_features=self.fc_size, out_features=self.latent_dim),
            nn.Tanh())

    def forward(self, obs, detach=''):

        if len(obs.shape) == 3:
            obs = obs.unsqueeze(1)
        features = self.convs(obs)

        wall_features = self.conv_walls(features)
        self.outputs['wall_features'] = wall_features

        features = features.flatten(1)

        latent_agent = self.mlp(features)

        self.outputs['output_MlP'] = latent_agent

        if detach:
            if detach =='wall':
                wall_features = wall_features.detach()
            elif detach =='agent':
                latent_agent = latent_agent.detach()
            elif detach =='both':
                wall_features = wall_features.detach()
                latent_agent = latent_agent.detach()

        return latent_agent, wall_features


class EncoderDMC_8x8wall_reward2(nn.Module):
    """Convolutional encoder_C for image-based observations."""
    def __init__(self, obs_channels, agent_dim, tanh=False, scale=1):
        super().__init__()

        self.agent_dim = agent_dim
        self.obs_channels = obs_channels
        self.outputs = dict()
        self.tanh = tanh
        self.scale = scale

        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=self.obs_channels, out_channels=int(32/scale), kernel_size=(3, 3), stride=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(in_channels=int(32/scale), out_channels=int(32/scale), kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(),
        )

        self.conv_walls = nn.Sequential(
            nn.Conv2d(in_channels=int(32/scale), out_channels=int(8/scale), kernel_size=(3, 3), stride=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(in_channels=int(8/scale), out_channels=1, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(),
        )

        self.conv_reward = nn.Sequential(
            nn.Conv2d(in_channels=int(32/scale), out_channels=int(8/scale), kernel_size=(3, 3), stride=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(in_channels=int(8/scale), out_channels=1, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(),
        )

        self.dummy_input = torch.ones((1, 1, 48, 48))
        self.dummy_output = self.convs(self.dummy_input)
        self.fc_size = self.dummy_output.flatten(1).shape[1]
        self.wall_output = self.conv_walls(self.dummy_output)
        self.wall_size = self.wall_output.shape[2]**2
        self.reward_size = self.wall_size
        self.mlp = nn.Sequential(
            nn.Linear(in_features=self.fc_size, out_features=self.agent_dim),
            nn.Tanh())
        if self.tanh:
            self.mlp.add_module("tanh_activation", nn.Tanh())

    def forward(self, obs, detach=''):

        if len(obs.shape) == 3:
            obs = obs.unsqueeze(1)
        features = self.convs(obs)

        wall_features = self.conv_walls(features)
        reward_features = self.conv_reward(features)
        self.outputs['wall_features'] = wall_features
        self.outputs['reward_features'] = reward_features

        features = features.flatten(1)

        latent_agent = self.mlp(features)

        self.outputs['output_MlP'] = latent_agent

        if detach:
            if detach =='wall':
                wall_features = wall_features.detach()
            elif detach =='agent':
                latent_agent = latent_agent.detach()
            elif detach =='reward':
                reward_features = reward_features.detach()
            elif detach =='both':
                wall_features = wall_features.detach()
                latent_agent = latent_agent.detach()

        return latent_agent, wall_features, reward_features


class EncoderDMC_8x8wall_small(nn.Module):
    """Convolutional encoder_C for image-based observations."""
    def __init__(self, obs_channels, latent_dim, tanh=False, scale=1):
        super().__init__()

        self.latent_dim = latent_dim
        self.obs_channels = obs_channels
        self.outputs = dict()
        self.tanh = tanh
        self.scale = scale

        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=self.obs_channels, out_channels=int(32/scale), kernel_size=(3, 3), stride=(2, 2)),
            nn.ReLU()
        )

        self.conv_walls = nn.Sequential(
            nn.Conv2d(in_channels=int(32/scale), out_channels=int(8/scale), kernel_size=(3, 3), stride=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(in_channels=int(8/scale), out_channels=1, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(),
        )

        self.dummy_input = torch.ones((1, 1, 48, 48))
        self.dummy_output = self.convs(self.dummy_input)
        self.fc_size = self.dummy_output.flatten(1).shape[1]
        self.wall_output = self.conv_walls(self.dummy_output)
        self.wall_size = self.wall_output.shape[2]**2
        self.mlp = nn.Sequential(
            nn.Linear(in_features=self.fc_size, out_features=self.latent_dim),
            nn.Tanh())
        if self.tanh:
            self.mlp.add_module("tanh_activation", nn.Tanh())

    def forward(self, obs, detach=False):

        if len(obs.shape) == 3:
            obs = obs.unsqueeze(1)
        features = self.convs(obs)

        if detach:
            features = features.detach()

        wall_features = self.conv_walls(features)
        self.outputs['wall_features'] = wall_features

        features = features.flatten(1)

        latent_agent = self.mlp(features)

        self.outputs['output_MlP'] = latent_agent

        return latent_agent, wall_features


class EncoderDMC_8x8wall_agent(nn.Module):
    """Convolutional encoder_C for image-based observations."""
    def __init__(self, obs_channels, scale=1):
        super().__init__()

        self.obs_channels = obs_channels
        self.scale = scale

        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=self.obs_channels, out_channels=int(32/scale), kernel_size=(3, 3), stride=(2, 2)),
            nn.Tanh(),
            nn.Conv2d(in_channels=int(32/scale), out_channels=int(32/scale), kernel_size=(3, 3), stride=(1, 1)),
            nn.Tanh(),
        )

        self.conv_walls = nn.Sequential(
            nn.Conv2d(in_channels=int(32/scale), out_channels=int(8/scale), kernel_size=(3, 3), stride=(2, 2)),
            nn.Tanh(),
            nn.Conv2d(in_channels=int(8/scale), out_channels=1, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(),
        )

        self.conv_agent = nn.Sequential(
            nn.Conv2d(in_channels=int(32/scale), out_channels=int(8/scale), kernel_size=(3, 3), stride=(2, 2)),
            nn.Tanh(),
            nn.Conv2d(in_channels=int(8/scale), out_channels=1, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(),
        )  # TODO try first convolutions and then pooling into an 8x8 featruremap.

    def forward(self, obs, detach=''):

        if len(obs.shape) == 3:
            obs = obs.unsqueeze(1)
        features = self.convs(obs)

        wall_features = self.conv_walls(features)
        agent_features = self.conv_agent(features)

        if detach:
            if detach =='wall':
                wall_features = wall_features.detach()
            elif detach =='agent':
                agent_features = agent_features.detach()
            elif detach =='both':
                wall_features = wall_features.detach()
                agent_features = agent_features.detach()

        return agent_features, wall_features


class EncoderDMC_8x8wall_agent_reward(nn.Module):
    """Convolutional encoder_C for image-based observations."""
    def __init__(self, obs_channels, scale=1):
        super().__init__()

        self.obs_channels = obs_channels
        self.scale = scale

        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=self.obs_channels, out_channels=int(32/scale), kernel_size=(3, 3), stride=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(in_channels=int(32/scale), out_channels=int(32/scale), kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=int(32 / scale), out_channels=int(8 / scale), kernel_size=(3, 3), stride=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(in_channels=int(8 / scale), out_channels=3, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(),  #TODO 1 encoder or split up between different heads?
        )

    def forward(self, obs, detach=False):

        if len(obs.shape) == 3:
            obs = obs.unsqueeze(1)
        features = self.convs(obs)
        if detach:
            features = features.detach()

        agent_features = features[0]
        wall_features = features[1]
        reward_features = features[2]

        return agent_features, wall_features, reward_features


# class EncoderDMC_half_features_catcher(nn.Module):
#     """Convolutional encoder_C for image-based observations."""
#     def __init__(self, obs_channels, latent_dim, scale=1):
#         super().__init__()
#
#         self.latent_dim = latent_dim
#         self.obs_channels = obs_channels
#         self.outputs = dict()
#         self.scale = scale
#
#         self.convs = nn.Sequential(
#             nn.Conv2d(in_channels=self.obs_channels, out_channels=32, kernel_size=(3, 3), stride=(2, 2)),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(1, 1)),
#             nn.ReLU(),
#         )
#
#         self.conv_ball = nn.Sequential(
#             nn.Conv2d(in_channels=32, out_channels=8, kernel_size=(3, 3), stride=(2, 2)),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=8, out_channels=1, kernel_size=(3, 3), stride=(1, 1)),
#             nn.ReLU(),
#             # nn.Conv2d(in_channels=32, out_channels=1, kernel_size=(4, 4), stride=(1, 1)),
#             # nn.ReLU(),
#             # nn.AdaptiveAvgPool2d(output_size=8)
#         )
#
#         self.dummy_input = torch.ones((1, 1, 48, 48))
#         self.dummy_output = self.convs(self.dummy_input)
#         self.fc_size = self.dummy_output.flatten(1).shape[1]
#         self.wall_output = self.conv_ball(self.dummy_output)
#         self.wall_size = self.wall_output.shape[2]**2
#
#         self.mlp = nn.Sequential(
#             nn.Linear(in_features=16928, out_features=self.latent_dim),
#             nn.Tanh())
#
#     def forward(self, obs, detach=str):
#
#         if len(obs.shape) == 3:
#             obs = obs.unsqueeze(1)
#         features = self.convs(obs)
#         self.outputs['features'] = features
#
#         if detach:
#             if detach =='base':
#                 features = features.detach()
#
#         ball_features = self.conv_ball(features)
#
#         features = features.flatten(1)
#
#         latent_agent = self.mlp(features)
#
#         if detach:                  # TODO check if algorithm works with the new detachment.
#             if detach =='ball':
#                 ball_features = ball_features.detach()
#             elif detach =='agent':
#                 latent_agent = latent_agent.detach()
#             elif detach =='both':
#                 ball_features = ball_features.detach()
#                 latent_agent = latent_agent.detach()
#
#         return latent_agent, ball_features

class EncoderDMC_half_features_catcher(nn.Module):
    """Convolutional encoder_C for image-based observations."""
    def __init__(self, obs_channels, latent_dim, scale=1, neuron_dim=100):
        super().__init__()

        self.latent_dim = latent_dim
        self.obs_channels = obs_channels
        self.outputs = dict()
        self.scale = scale

        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=self.obs_channels, out_channels=32, kernel_size=(3, 3), stride=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(),
        )

        self.conv_ball = nn.Sequential(
            # nn.Conv2d(in_channels=int(32), out_channels=int(32), kernel_size=(1, 1), stride=(1, 1)),
            # nn.ReLU(),
            nn.Conv2d(in_channels=int(32), out_channels=int(1), kernel_size=(4, 4), stride=(1, 1)),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(output_size=6),  # Todo
        )

        self.mlp = nn.Sequential(
            nn.Linear(in_features=16928, out_features=neuron_dim),
            nn.Tanh(),
            nn.Linear(in_features=neuron_dim, out_features=self.latent_dim),
            nn.Tanh())

    def forward(self, obs, detach=str):

        if len(obs.shape) == 3:
            obs = obs.unsqueeze(1)
        features = self.convs(obs)
        self.outputs['features'] = features

        if detach:
            if detach =='base':
                features = features.detach()

        ball_features = self.conv_ball(features)

        features = features.flatten(1)

        latent_agent = self.mlp(features)

        if detach:                  # TODO check if algorithm works with the new detachment.
            if detach =='ball':
                ball_features = ball_features.detach()
            elif detach =='agent':
                latent_agent = latent_agent.detach()
            elif detach =='both':
                ball_features = ball_features.detach()
                latent_agent = latent_agent.detach()

        return latent_agent, ball_features


class EncoderDMC_small(nn.Module):
    """Convolutional encoder_C for image-based observations."""
    def __init__(self, obs_channels, latent_dim, tanh=False, scale=1):
        super().__init__()

        self.latent_dim = latent_dim
        self.obs_channels = obs_channels
        self.outputs = dict()
        self.tanh = tanh
        self.scale = scale

        self.convs = nn.Sequential(
            nn.Conv2d(self.obs_channels, out_channels=32, kernel_size=(3, 3), stride=(2, 2)),
            nn.ReLU()
        )

        self.mlp = nn.Sequential(
            nn.Linear(in_features=16928, out_features=self.latent_dim),
            nn.Tanh())

    def forward(self, obs, detach=False):

        if len(obs.shape) == 3:
            obs = obs.unsqueeze(1)
        features = self.convs(obs)

        if detach:
            features = features.detach()

        features = features.flatten(1)

        latent = self.mlp(features)

        self.outputs['output_MlP'] = latent

        return latent


class EncoderMLP(nn.Module):
    def __init__(self, obs, latent_dim, scale=4, tanh=False):
        super().__init__()

        self.latent_dim = latent_dim
        self.obs_dim = len(obs.flatten())
        self.outputs = dict()
        self.counter = 0
        self.scale = scale
        self.tanh = tanh

        self.mlp = nn.Sequential(
            nn.Linear(in_features=self.obs_dim, out_features=int(256/self.scale)),
            nn.Tanh(),
            nn.Linear(in_features=int(256/self.scale), out_features=int(128/self.scale)),
            nn.Tanh(),
            nn.Linear(in_features=int(128/self.scale), out_features=int(64/self.scale)),
            nn.Tanh(),
            nn.Linear(in_features=int(64/self.scale), out_features=int(16/self.scale)),
            nn.Tanh(),
            nn.Linear(in_features=int(16/self.scale), out_features=self.latent_dim))

        if self.tanh:
            self.mlp.add_module("tanh_activation", nn.Tanh())

    def forward(self, obs, detach=False):

        if len(obs.shape) == 2:
            obs = obs.unsqueeze(0)
        obs = obs.flatten(1)
        mlp = self.mlp(obs)

        self.outputs['out'] = mlp

        if detach:
            mlp = mlp.detach()
        self.counter += 1

        return mlp


class TransitionModel(nn.Module):
    """ Transition function MLP head for both w/o and w/ action"""
    def __init__(self, latent_dim, action_dim, scale=1, tanh=False, prediction_dim=2):
        super().__init__()

        self.input_dim = latent_dim + action_dim
        self.outputs = dict()

        self.prediction_dim = prediction_dim
        self.counter = 0
        self.tanh = tanh

        self.linear_layers = nn.Sequential(
            nn.Linear(in_features=self.input_dim, out_features=int(8*scale)),
            nn.Tanh(),
            nn.Linear(in_features=int(8*scale), out_features=int(32*scale)),
            nn.Tanh(),
            nn.Linear(in_features=int(32*scale), out_features=int(32*scale)),
            nn.Tanh(),
            # nn.Linear(in_features=int(32 / scale), out_features=int(32 / scale)),
            # nn.Tanh(),
            nn.Linear(in_features=int(32*scale), out_features=int(8*scale)),
            nn.Tanh(),
            nn.Linear(in_features=int(8*scale), out_features=self.prediction_dim),
        )   # TODO test larger sizes of transitionmodels for perfect transitions in multimazes??


    def forward(self, z, detach=False):

        prediction = self.linear_layers(z)

        if detach:
            prediction = prediction.detach()

        return prediction


class DQNmodel(nn.Module):
    """ Transition function MLP head for both w/o and w/ action"""
    def __init__(self, input_dim, scale=16, prediction_dim=4):
        super().__init__()

        self.counter = 0
        self.scale = int(scale)

        self.linear_layers = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=int(8*scale)),
            nn.Tanh(),
            nn.Linear(in_features=int(8*scale), out_features=int(32*scale)),
            nn.Tanh(),
            nn.Linear(in_features=int(32*scale), out_features=int(32*scale)),
            nn.Tanh(),
            nn.Linear(in_features=int(32*scale), out_features=int(8*scale)),
            nn.Tanh(),
            nn.Linear(in_features=int(8*scale), out_features=prediction_dim),
        )

    def forward(self, controllable_latent, uncontrollable_features, detach=False):

        q_values = self.linear_layers(torch.cat((controllable_latent, uncontrollable_features.flatten(1)), dim=1))

        if detach:
            q_values = q_values.detach()

        return q_values


class ConvTransitionModel(nn.Module):
    """ Transition function MLP head for both w/o and w/ action"""
    def __init__(self, input_channels=3, output_channels=1):
        super().__init__()

        self.input_channels = input_channels
        self.outputs = dict()

        self.convs = nn.Sequential(
            nn.Conv2d(self.input_channels, out_channels=16, kernel_size=(1, 1), stride=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(2, 2), stride=(1, 1), padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(2, 2), stride=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(1, 1), stride=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=output_channels, kernel_size=(1, 1), stride=(1, 1)),
        )

    def forward(self, z, detach=False):

        prediction = self.convs(z)

        self.outputs['prediction'] = prediction

        if detach:
            prediction = prediction.detach()

        return prediction


class HalfConv_TransitionModel(nn.Module):
    """ Transition function MLP head for both w/o and w/ action"""
    def __init__(self, agent_dim=2, action_dim=1):
        super().__init__()

        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(2, 2), stride=(1, 1)),
            nn.ReLU(),
            # nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(2, 2), stride=(1, 1)),
            # nn.ReLU(),
            # nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(1, 1), stride=(1, 1)),
            # nn.ReLU(),
        )

        self.dummy_input = torch.ones((1, 1, 6, 6))
        self.dummy_output = self.convs(self.dummy_input)
        self.conv_size = self.dummy_output.flatten(1).shape[1]

        self.mlp = nn.Sequential(
            nn.Linear(in_features=self.conv_size + agent_dim + action_dim, out_features=128),
            nn.Tanh(),
            nn.Linear(in_features=128, out_features=256),
            nn.Tanh(),
            nn.Linear(in_features=256, out_features=128),
            nn.Tanh(),
            nn.Linear(in_features=128, out_features=2)
            )

    def forward(self, agent, wallfeatures_non_flattened, action, detach=False):

        if len(agent.shape) == 1:
            agent = agent.unsqueeze(0)
        if len(action.shape) == 1:
            action = action.unsqueeze(0)

        wall_features = self.convs(wallfeatures_non_flattened).flatten(1)

        prediction = self.mlp(torch.cat((agent, wall_features, action), dim=1))

        if detach:
            prediction = prediction.detach()

        return prediction



class InversePredictionModel(nn.Module):
    """ MLP for the inverse prediction"""

    def __init__(self, input_dim, num_actions=4, scale=1, activation='tanh', final_activation=True):
        super().__init__()

        if activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()

        self.final_activation = final_activation
        self.scale = int(scale)

        self.linear_layers = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=int(8 * scale)),
            self.activation,
            nn.Linear(in_features=int(8 * scale), out_features=int(32 * scale)),
            self.activation,
            nn.Linear(in_features=int(32 * scale), out_features=int(32 * scale)),
            self.activation,
            nn.Linear(in_features=int(32 * scale), out_features=int(8 * scale)),
            self.activation,
            nn.Linear(in_features=int(8 * scale), out_features=num_actions),
        )

    def forward(self, z, detach=False):

        prediction = self.linear_layers(z)

        if detach:
            prediction = prediction.detach()

        return prediction


class DQNconvmodel(nn.Module):
    """ Transition function MLP head for both w/o and w/ action"""
    def __init__(self, input_channels=1, num_actions=4, agent_size=2, intermediate_size=50, intermediate_agent_size=30):
        super().__init__()

        self.input_channels = input_channels
        self.outputs = dict()

        self.convs = nn.Sequential(
            nn.Conv2d(self.input_channels, out_channels=32, kernel_size=(1, 1), stride=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(2, 2), stride=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(1, 1), stride=(1, 1)),
        )

        self.dummy_input = torch.ones((1, self.input_channels, 6, 6))
        self.dummy_output = self.convs(self.dummy_input)
        self.fc_size = self.dummy_output.flatten(1).shape[1]

        self.mlp_features = nn.Sequential(
            nn.Linear(in_features=self.fc_size, out_features=intermediate_size),
            nn.Tanh(),
        )

        self.mlp_final = nn.Sequential(
            nn.Linear(in_features=intermediate_size+agent_size, out_features=num_actions)
        )

    def forward(self, mlp_states, features, detach=False):

        intermediate_feature_input1 = self.convs(features)
        intermediate_feature_input2 = self.mlp_features(intermediate_feature_input1.flatten(1))
        prediction = self.mlp_final(torch.cat((mlp_states, intermediate_feature_input2), dim=1))
        self.outputs['prediction'] = prediction

        if detach:
            prediction = prediction.detach()

        return prediction



class Encoder_agent(nn.Module):
    """Convolutional encoder_C for image-based observations."""
    def __init__(self, obs_channels, latent_dim, tanh=False, scale=1):
        super().__init__()

        self.latent_dim = latent_dim
        self.obs_channels = obs_channels
        self.outputs = dict()
        self.tanh = tanh
        self.scale = scale

        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=self.obs_channels, out_channels=int(32/scale), kernel_size=(3, 3), stride=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(in_channels=int(32/scale), out_channels=int(32/scale), kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(),
        )

        self.dummy_input = torch.ones((1, 1, 48, 48))
        self.dummy_output = self.convs(self.dummy_input)
        self.fc_size = self.dummy_output.flatten(1).shape[1]
        self.mlp = nn.Sequential(
            nn.Linear(in_features=self.fc_size, out_features=self.latent_dim),
            nn.Tanh())

    def forward(self, obs, detach=''):

        if len(obs.shape) == 3:
            obs = obs.unsqueeze(1)
        features = self.convs(obs)

        if detach:
            if detach =='base':
                features = features.detach()

        features = features.flatten(1)

        latent_agent = self.mlp(features)

        self.outputs['output_MlP'] = latent_agent

        if detach =='agent':
            latent_agent = latent_agent.detach()

        return latent_agent


class Encoder_wall(nn.Module):
    """Convolutional encoder_C for image-based observations."""
    def __init__(self, obs_channels, latent_dim, tanh=False, scale=1):
        super().__init__()

        self.latent_dim = latent_dim
        self.obs_channels = obs_channels
        self.outputs = dict()
        self.tanh = tanh
        self.scale = scale
        self.output_size=8

        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=self.obs_channels, out_channels=int(32/scale), kernel_size=(3, 3), stride=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(in_channels=int(32/scale), out_channels=int(1), kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(output_size=self.output_size)
        )

        self.dummy_input = torch.ones((1, 1, 48, 48))
        self.dummy_output = self.convs(self.dummy_input)
        self.wall_size = self.output_size**2

    def forward(self, obs, detach=''):

        if len(obs.shape) == 3:
            obs = obs.unsqueeze(1)
        wall_features = self.convs(obs)

        self.outputs['wall_features'] = wall_features

        if detach =='wall':
                wall_features = wall_features.detach()

        return wall_features


class TransitionModel_old(nn.Module):
    """ Transition function MLP head for both w/o and w/ action"""
    def __init__(self, latent_dim, action_dim, scale=1, tanh=False, prediction_dim=2):
        super().__init__()

        self.input_dim = latent_dim + action_dim
        self.outputs = dict()

        self.prediction_dim = prediction_dim
        self.counter = 0
        self.scale = int(scale)
        self.tanh = tanh

        self.linear_layers = nn.Sequential(
            nn.Linear(in_features=self.input_dim, out_features=int(8/scale)),
            nn.Tanh(),
            nn.Linear(in_features=int(8/scale), out_features=int(32/scale)),
            nn.Tanh(),
            nn.Linear(in_features=int(32/scale), out_features=int(32/scale)),
            nn.Tanh(),
            nn.Linear(in_features=int(32/scale), out_features=int(8/scale)),
            nn.Tanh(),
            nn.Linear(in_features=int(8/scale), out_features=self.prediction_dim),
        )   # TODO test larger sizes of transitionmodels for perfect transitions in multimazes??

        if self.tanh:
            self.linear_layers.add_module("tanh_activation", nn.Tanh())

    def forward(self, z, detach=False):

        prediction = self.linear_layers(z)

        if detach:
            prediction = prediction.detach()

        return prediction