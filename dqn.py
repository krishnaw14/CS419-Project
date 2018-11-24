#!/usr/bin/env python
# coding: utf-8

# In[2]:


import gym


# In[3]:


import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image



# In[4]:


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T


# In[5]:
env = gym.make('CartPole-v0').unwrapped

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

x =128
gamma = 0.9
Transition = namedtuple('Transition',('state', 'action', 'next_state', 'reward'))
class experience(object):
#this class will define the memory of the q-learning

    def __init__(self,capacity):#constructor function
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def pushing(self, *args):
        #Saving a transition

        if len(self.memory)<x:
            return
        self.memory[self.position] = Transition(*args)
        self.position = (self.position+1)%self.capacity
    def batch(self,x):
        return random.sample(self.memory, x)
    def __len__(self):
        return len(self.memory)

class qlearn(nn.Module):
    def __init__(self):
        super(qlearn,self).__init__()
        self.linear=torch.nn.Linear(400,2)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
    def ahead(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))


# In[6]:


start = .99
x=128
end = .05
decay = 200
update = 10
mem = experience(10000)


# In[7]:


#cpu/gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[8]:


policy_net = qlearn().to(device)
target_net = qlearn().to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())


# In[9]:


steps = 0 


# In[10]:


def action(state):
    global steps
    sample = random.random()
    threshold  = end + (start-end) * math.exp(-1.0 * steps/decay )
    if sample > threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(2)]], device=device, dtype=torch.long)


# In[11]:


duration = []


# In[12]:


#plot


# In[13]:


def plotting():
    plt.figure(2)
    plt.clf()
    t = torch.tensor(duration, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(t.numpy())
    if len(t) >= 100:
        means = tt.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.01)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())


# In[16]:


def optimization():
    if len(mem)>x:
        return
    trans = mem.batch(x)
    batch = Transition(*zip(*trans))
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,batch.next_state)), device=device, dtype=torch.uint8)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


# In[17]:


num = 50

resize = T.Compose([T.ToPILImage(),
                    T.Resize(40, interpolation=Image.CUBIC),
                    T.ToTensor()])
screen_width = 600

def location():
    width = env.x_threshold * 2
    scale = screen_width / width
    return int(env.state[0] * scale + screen_width / 2.0)
#Change the screen by stripping	top and bottom
def screen():
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))
    screen = screen[:, 160:320]
    view_width = 320
    cart_location = location()
    
    if cart_location < view_width // 2:
        slicing = slice(view_width)
    elif cart_location > (screen_width - view_width // 2):
        slicing = slice(-view_width, None)
    else:
        slicing = slice(cart_location - view_width // 2,
                            cart_location + view_width // 2)
    screen = screen[:, :, slicing]
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    return resize(screen).unsqueeze(0).to(device)


env.reset()
plt.figure()
plt.imshow(screen().cpu().squeeze(0).permute(1, 2, 0).numpy(),
           interpolation='none')
plt.title('Example extracted screen')
plt.show()
for i_episode in range(num):
    
    env.reset()
    last = screen()
    current = screen()
    state = current - last
    for t in count():
        action = action(state)
        _, reward, done, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        last = current
        current = screen()
        if not done:
            next_state = current - last
        else:
            next_state = None
        mem.pushing(state, action, next_state, reward)
        state = next_state
        optimization()
        if done:
            episode_durations.append(t + 1)
            plotting()
            break
    if i % update == 0:
        target_net.load_state_dict(policy_net.state_dict())

print('Complete')
env.render()
env.close()
plt.ioff()
plt.show()