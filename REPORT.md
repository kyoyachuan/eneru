# Lab6: Deep Q-Network and Deep Deterministic Policy Gradient

### A tensorboard plot shows episode rewards of at least 800 training episodes in LunarLander-v2
![](artifacts/lunarlander-v2.png)

### A tesorboard plot shows episode rewards of at least 800 training episodes in LunarLanderContinuous-v2
![](artifacts/lunarlandercont-v2.png)


### Describe your major implementation of both algorithms in detail.

#### DQN
Most of the architecture and algorithms implementations were based on the given spec. For the neural network, I implemented 2 fully connected layers, ReLU activation function used in hidden layer, and used Adam optimizer for learning.

```python
class Net(nn.Module):
    def __init__(self, state_dim=8, action_dim=4, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(inplace=True),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),

            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x):
        x = torch.tensor(x, device="cuda")
        return self.net(x)

class DQN:
    def __init__(self, args):
        self._behavior_net = Net().to(args.device)
        self._target_net = Net().to(args.device)
        # initialize target network
        self._target_net.load_state_dict(self._behavior_net.state_dict())
        self._optimizer = optim.Adam(self._behavior_net.parameters(), lr=args.lr)
	
	...
```

Based on the algorithm in spec, I sample a random minibatch data from the replay memory. After that, get the Q-value from behavior net and calculate its Q-target value by the target net given next state, and calculate its reward. Used mean squared error (MSE) as loss function to calculate the loss between Q-value and reward and perform back propagation on the behavior net. Given a certain period, perform target net update by directly copy parameters from behavior net to target net.

```python
	...

    def _update_behavior_network(self, gamma):
        # sample a minibatch of transitions
        state, action, reward, next_state, done = self._memory.sample(
            self.batch_size, self.device)

        q_value = self._behavior_net(state).gather(1, action.long())
        with torch.no_grad():
            q_next = self._target_net(next_state)
            q_next, _ = torch.max(q_next, 1)
            q_next = q_next.unsqueeze(1)
            q_target = reward + gamma * q_next * (1 - done)
        criterion = nn.MSELoss()
        loss = criterion(q_value, q_target)
        # optimize
        self._optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self._behavior_net.parameters(), 5)
        self._optimizer.step()

    def _update_target_network(self):
        '''update target network by copying from behavior network'''
        self._target_net.load_state_dict(self._behavior_net.state_dict())
	
	...
```

When performing data collection, I applied epsilon greedy algorithm for exploration. Given probability epsilon, I sampled a random action from action space if random number smaller than epsilon, else choose the best action that has the max Q-value given next state. Finally, I implemented testing function by using this `select_action` method to get the action and apply on the game environment, to get reward, next state and done.

```python
	...

    def select_action(self, state, epsilon, action_space):
        '''epsilon-greedy based on behavior network'''
        if random.random() < epsilon:
            return action_space.sample()
        else:
            v = self._behavior_net(state)
            _, action = torch.max(v, 0)
            return action.item()
	
	...

def test(args, env, agent, writer):
    print('Start Testing')
    action_space = env.action_space
    epsilon = args.test_epsilon
    seeds = (args.seed + i for i in range(10))
    rewards = []
    for n_episode, seed in enumerate(seeds):
        total_reward = 0
        env.seed(seed)
        state = env.reset()
        with torch.no_grad():
            done = 0
            while not done:
                action = agent.select_action(state, epsilon, action_space)
                next_state, reward, done, _ = env.step(action)
                state = next_state
                total_reward += reward

            rewards.append(total_reward)

        print(total_reward)
    print('Average Reward', np.mean(rewards))
    env.close()
```

#### DDPG
Similarly, I implemented architecture and algorithm based on the given spec. For the actor net, I implemented 2 fully connected layers, ReLU activation function used in hidden layer, Tanh activation function used in output layer, and used Adam optimizer for actor, critic learning.

```python
class ActorNet(nn.Module):
    def __init__(self, state_dim=8, action_dim=2, hidden_dim=(400, 300)):
        super().__init__()
        h1, h2 = hidden_dim
        self.net = nn.Sequential(
            nn.Linear(state_dim, h1),
            nn.ReLU(inplace=True),

            nn.Linear(h1, h2),
            nn.ReLU(inplace=True),

            nn.Linear(h2, action_dim),
            nn.Tanh()
        )

    def forward(self, x):
        x = torch.tensor(x, device="cuda")
        return self.net(x)

	...

class DDPG:
    def __init__(self, args):
        # behavior network
        self._actor_net = ActorNet().to(args.device)
        self._critic_net = CriticNet().to(args.device)
        # target network
        self._target_actor_net = ActorNet().to(args.device)
        self._target_critic_net = CriticNet().to(args.device)
        # initialize target network
        self._target_actor_net.load_state_dict(self._actor_net.state_dict())
        self._target_critic_net.load_state_dict(self._critic_net.state_dict())
        self._actor_opt = optim.Adam(self._actor_net.parameters(), lr=args.lra)
        self._critic_opt = optim.Adam(self._critic_net.parameters(), lr=args.lrc)

	...
```

For the learning algorithm, I sampled a random minibatch data from the replay memory. After that, get the Q-value from the critic net. Use actor net to predict the probability of actions given next state and use corresponding action to predict its Q-value. Use this to calculate Q-target value and calculate loss by using MSE, and perform back propagation on the critic net.

After updated critic net, I used actor net to predict the probability of actions given current state. Feed the state and predicted actions into critic net and use its mean as loss function for performing back propagation.

Moreover, I used soft update to update the target network by specify tau value.

```python
	...

    def _update_behavior_network(self, gamma):
        actor_net, critic_net, target_actor_net, target_critic_net = self._actor_net, self._critic_net, self._target_actor_net, self._target_critic_net
        actor_opt, critic_opt = self._actor_opt, self._critic_opt

        # sample a minibatch of transitions
        state, action, reward, next_state, done = self._memory.sample(
            self.batch_size, self.device)

        ## update critic ##
        # critic loss
        q_value = critic_net(state, action)
        with torch.no_grad():
            a_next = target_actor_net(next_state)
            q_next = target_critic_net(next_state, a_next)
            q_target = reward + gamma * q_next * (1 - done)
        criterion = nn.MSELoss()
        critic_loss = criterion(q_value, q_target)
        # optimize critic
        actor_net.zero_grad()
        critic_net.zero_grad()
        critic_loss.backward()
        critic_opt.step()

        ## update actor ##
        # actor loss
        action = actor_net(state)
        actor_loss = -critic_net(state, action).mean()
        # optimize actor
        actor_net.zero_grad()
        critic_net.zero_grad()
        actor_loss.backward()
        actor_opt.step()

    @staticmethod
    def _update_target_network(target_net, net, tau):
        '''update target network by _soft_ copying from behavior network'''
        for target, behavior in zip(target_net.parameters(), net.parameters()):
            target.data.copy_(tau * behavior.data + (1.0 - tau) * target.data)
	
	...
```

I add noise value option for exploration when getting the action from actor net. In the testing phase, I disable the noise option and get the best action for the game.

```python
	...
    
    def select_action(self, state, noise=True):
        '''based on the behavior (actor) network and exploration noise'''
        action = self._actor_net(state).detach().cpu().numpy()
        if noise:
            action = action + self._action_noise.sample()
        return action

	...
	
def test(args, env, agent, writer):
    print('Start Testing')
    seeds = (args.seed + i for i in range(10))
    rewards = []
    for n_episode, seed in enumerate(seeds):
        total_reward = 0
        env.seed(seed)
        state = env.reset()
        with torch.no_grad():
            done = 0
            while not done:
                action = agent.select_action(state, noise=False)
                next_state, reward, done, _ = env.step(action)

                state = next_state
                total_reward += reward

            rewards.append(total_reward)

        print(total_reward)
    print('Average Reward', np.mean(rewards))
    env.close()
```

### Describe differences between your implementation and algorithms.
Most of the implementation was same as the algorithms as I mainly based on the spec.

### Describe your implementation and the gradient of actor updating.
This is discussed in the [implementation DDPG section](#ddpg)

### Describe your implementation and the gradient of critic updating
This is discussed in the [implementation DDPG section](#ddpg)

### Explain effects of the discount factor.
The discount factor stated as gamma in codebase. The usage of discount factor was to control the attention to the future reward. If the discount factor is close to 0, it means it will pay more attention on short term rewards. Since this is a sequential decision learning, it would be better to pay more attention on longest future reward instead of short rewards.

### Explain benefits of epsilon-greedy in comparison to greedy action selection.
Epsilon-greedy is useful for for exploring and collecting unexplored states/actions. Using greedy action selection would affect agents choose certain states/actions, led to local optimal.

### Explain the necessity of the target network.
Since the model is similar as supervised regression problem, the goal is to fit the network output Q-value to target (reward at t + gamma * Q-value given next state). Without using the target network, the training target would change every iteration, it would led the model hard to converge.

### Explain the effect of replay buffer size in case of too large or too small.
Using larger experience replay would less likely sampled correlated elements, hence the training would be stable, but it might slow training. If using smaller experience replay, meaning using the very most recent data, it might overfit to that and things break.

### Implement and experiment on Double-DQN
In the implementation of Double DQN, I modified `update_behavior_network` method. I calculate the largest Q-value of next state's action by behavior network. After that, get the predicted Q-value as Q-target value by target network given this action and next state. The experiment result is shown in [first section](#a-tensorboard-plot-shows-episode-rewards-of-at-least-800-training-episodes-in-lunarlander-v2).

```python
	...

    def _update_behavior_network(self, gamma):
        # sample a minibatch of transitions
        state, action, reward, next_state, done = self._memory.sample(
            self.batch_size, self.device)

        q_value = self._behavior_net(state).gather(1, action.long())
        with torch.no_grad():
            q_next_value = self._behavior_net(next_state)
            next_action = torch.max(q_next_value, 1)[1].unsqueeze(1)
            q_next = self._target_net(next_state).gather(1, next_action)
            q_target = reward + gamma * q_next * (1 - done)
        criterion = nn.MSELoss()
        loss = criterion(q_value, q_target)
	
	...
```

### Implement and experiment on TD3 (Twin-Delayed DDPG)
There were some code changes based on `ddpg.py`. First, I implemented twin critic net (two same net in `Critic`). When performing behavior network updating, I used the smaller next Q-value from the twin critic net as Q-target value. After that, calculate loss between Q-target value and the corresponding two Q-value independently and perform back propagation. In addition, I added noise in predicted action given next state. Finally, I specify the policy update frequency for the actor net and corresponding target net for delayed update purpose. The experiment result is shown in [second section](#a-tesorboard-plot-shows-episode-rewards-of-at-least-800-training-episodes-in-lunarlandercontinuous-v2).

```python
class CriticNet(nn.Module):
    def __init__(self, state_dim=8, action_dim=2, hidden_dim=(400, 300)):
        super().__init__()
        h1, h2 = hidden_dim
        self.critic_head_1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, h1),
            nn.ReLU(),
        )
        self.critic_1 = nn.Sequential(
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, 1),
        )

        self.critic_head_2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, h1),
            nn.ReLU(),
        )
        self.critic_2 = nn.Sequential(
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, 1),
        )

    def forward(self, x, action):
        q1 = self.q1(x, action)

        x2 = self.critic_head_2(torch.cat([x, action], dim=1))
        q2 = self.critic_2(x2)

        return q1, q2

    def q1(self, x, action):
        x1 = self.critic_head_1(torch.cat([x, action], dim=1))
        return self.critic_1(x1)

	...

    def update(self):
        self.total_it += 1
        # update the behavior networks
        self._update_behavior_network(self.gamma)
        # update the target networks
        if self.total_it % self.policy_freq == 0:
            self._update_target_network(self._target_actor_net, self._actor_net,
                                        self.tau)
            self._update_target_network(self._target_critic_net, self._critic_net,
                                        self.tau)

    def _update_behavior_network(self, gamma):
        actor_net, critic_net, target_actor_net, target_critic_net = self._actor_net, self._critic_net, self._target_actor_net, self._target_critic_net
        actor_opt, critic_opt = self._actor_opt, self._critic_opt

        # sample a minibatch of transitions
        state, action, reward, next_state, done = self._memory.sample(
            self.batch_size, self.device)

        ## update critic ##
        # critic loss
        q_value_1, q_value_2 = critic_net(state, action)
        with torch.no_grad():
            a_next = target_actor_net(next_state)
            a_next = a_next + torch.from_numpy(self._action_noise.sample()).float().to(self.device)
            q_next_1, q_next_2 = target_critic_net(next_state, a_next)
            q_next = torch.min(q_next_1, q_next_2)
            q_target = reward + gamma * q_next * (1 - done)
        criterion = nn.MSELoss()
        critic_loss = criterion(q_value_1, q_target) + criterion(q_value_2, q_target)
        # optimize critic
        actor_net.zero_grad()
        critic_net.zero_grad()
        critic_loss.backward()
        critic_opt.step()

        ## update actor ##
        # actor loss
        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:
            action = actor_net(state)
            actor_loss = -critic_net.q1(state, action).mean()
            # optimize actor
            actor_net.zero_grad()
            critic_net.zero_grad()
            actor_loss.backward()
            actor_opt.step()

	...
```

### Performance
- **LunarLander-v2**: average reward **226.84**.
![](artifacts/lunarlander-test.png)

- **LunarLanderContinuous-v2**: average reward **288.50**.
![](artifacts/lunarlandercont-test.png)
