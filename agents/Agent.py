
import logging
import random
import sys
from typing import List, Tuple
import numpy as np
import pickle
import torch
from torch import nn

sys.path.append('.')
from env.Observation import Observation
from env.Actions import Action, ChaodiAction, DeclareAction, DontChaodiAction, DontDeclareAction, FollowAction, LeadAction, PlaceKittyAction
from env.utils import Stage
from env.CardSet import CardSet, MoveType

from env.utils import AbsolutePosition

class StageModule:
    def act(self, obs: Observation, epsilon=None, training=True):
        return NotImplementedError()

    def load_model(self, model: nn.Module):
        raise NotImplementedError()

    def learn_from_samples(self, samples: List[Tuple[Observation, Action, float]]):
        raise NotImplementedError()

class SJAgent:
    def __init__(self, name: str) -> None:
        self.name = name

        self.declare_module: StageModule = None
        self.kitty_module: StageModule = None
        self.chaodi_module: StageModule = None
        self.main_module: StageModule = None
        self.name_module: StageModule = None
    
    def act(self, obs: Observation, epsilon=None, training=True):
        assert self.declare_module is not None and self.kitty_module is not None and self.name_module is not None and self.main_module is not None, "At least one required model is not loaded"
        if obs.stage == Stage.declare_stage:
            return self.declare_module.act(obs, epsilon, training)
        elif obs.stage == Stage.kitty_stage:
            return self.kitty_module.act(obs, epsilon, training)
        elif obs.stage == Stage.chaodi_stage:
            assert self.chaodi_module is not None, "chaodi module must be configured when chaodi mode is turned on"
            return self.chaodi_module.act(obs, epsilon, training)
        elif obs.stage == Stage.name_stage:
            assert self.name_module is not None, "name module must be configured when name stage is used"
            return self.name_module.act(obs, epsilon, training)
        elif obs.stage == Stage.main_stage:
            return self.main_module.act(obs, epsilon, training)
        else:
            raise NotImplementedError()
    
    def learn_from_samples(self, samples: List[Tuple[Observation, Action, float]], stage: Stage):
        if stage == Stage.declare_stage:
            self.declare_module.learn_from_samples(samples)
        elif stage == Stage.kitty_stage:
            self.kitty_module.learn_from_samples(samples)
        elif stage == Stage.chaodi_stage:
            self.chaodi_module.learn_from_samples(samples)
        elif stage == Stage.name_stage:
            assert self.name_module is not None, "name module must be configured when name stage is used"
            self.name_module.learn_from_samples(samples)
        elif stage == Stage.main_stage:
            self.main_module.learn_from_samples(samples)
        else:
            raise NotImplementedError()

    def optimizer_states(self):
        return {}

    def load_optimizer_states(self, state):
        pass
    
    # Try to load models from disk. Return whether the models were loaded successfully.
    def load_models_from_disk(self) -> bool:
        raise NotImplementedError()

    def save_models_to_disk(self):
        raise NotImplementedError()

