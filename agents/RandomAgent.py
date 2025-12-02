from .Agent import SJAgent, StageModule
from typing import List
import sys
import random
from env.CardSet import CardSet, MoveType

sys.path.append('.')
from env.Actions import Action, LeadAction
from env.utils import ORDERING_INDEX, Stage, softmax
from env.Observation import Observation
from networks.Models import *

class RandomActorModule(StageModule):
    def act(self, obs: Observation, epsilon=None, training=True):
        if obs.stage != Stage.main_stage:
            return random.choice(obs.actions)
        
        non_combo: List[Action] = []
        combo: List[Action] = []
        others: List[Action] = []
        for a in obs.actions:
            if isinstance(a, LeadAction):
                if isinstance(a.move, MoveType.Combo):
                    combo.append(a)
                else:
                    non_combo.append(a)
            else:
                others.append(a)
                
        if others:
            return random.choice(others), None, None
        elif not non_combo:
            return random.choice(combo), None, None
        elif not combo:
            return random.choice(non_combo), None, None
        else:
            return (random.choice(combo) if random.random() > 0.9 else random.choice(non_combo)), None, None

class RandomAgent(SJAgent):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        
        self.declare_module = RandomActorModule()
        self.kitty_module = RandomActorModule()
        self.chaodi_module = RandomActorModule()
        self.name_module = RandomActorModule()
        self.main_module = RandomActorModule()