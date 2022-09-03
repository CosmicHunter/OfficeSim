# This is the action format used for reinforcement learning
# This is the same action format used in CrowdNav implementation on https://github.com/vita-epfl/CrowdNav

from collections import namedtuple
ActionXY = namedtuple('ActionXY', ['vx', 'vy'])