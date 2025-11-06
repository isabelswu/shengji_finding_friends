# Helper classes, functions and enums
from typing import List
import random
from enum import Enum
import torch
import numpy as np

ORDERING = ['A♦', 'K♦', 'Q♦', 'J♦', '10♦', '9♦', '8♦', '7♦', '6♦', '5♦', '4♦', '3♦', '2♦', 'A♣', 'K♣', 'Q♣', 'J♣', '10♣', '9♣', '8♣', '7♣', '6♣', '5♣', '4♣', '3♣', '2♣', 'A♥', 'K♥', 'Q♥', 'J♥', '10♥', '9♥', '8♥', '7♥', '6♥', '5♥', '4♥', '3♥', '2♥', 'A♠', 'K♠', 'Q♠', 'J♠', '10♠', '9♠', '8♠', '7♠', '6♠', '5♠', '4♠', '3♠', '2♠', 'XJ', 'DJ']
ORDERING_INDEX = {k:i for i, k in enumerate(ORDERING)}

class TrumpSuit(str, Enum):
    "All the possible suits for a trump declaration."
    CLUB = "♣"
    SPADE = "♠"
    HEART = "♥"
    DIAMOND = "♦"
    XJ = "XJ" # NT type 1
    DJ = "DJ" # NT type 2

    @property
    def is_NT(self):
        return self == 'XJ' or self == 'DJ'
    
    @property
    def tensor(self):
        rep = torch.zeros(6)
        idx = [self.DIAMOND, self.CLUB, self.HEART, self.SPADE, self.XJ, self.DJ].index(self)
        rep[idx] = 1
        return rep
    
    @classmethod
    def from_tensor(self, tensor: torch.Tensor) -> None:
        assert tensor.shape[0] == 6 and torch.sum(tensor == 1) == 1 and tensor.sum() == 1, "tensor must be one hot encoded"
        return [self.DIAMOND, self.CLUB, self.HEART, self.SPADE, self.XJ, self.DJ][tensor.argmax()]

class CardSuit(str, Enum):
    "All suits that a card can belong to in a game."
    CLUB = "♣"
    SPADE = "♠"
    HEART = "♥"
    DIAMOND = "♦"
    TRUMP = 'T'

class RelativePosition(str, Enum):
    LEFT = 'L'
    RIGHT = 'R'
    OPPOL = 'OL'
    OPPOR = 'OR'
    SELF = 'S'

    @classmethod
    def in_order(self):
        "Returns 5-tuple of RelativePosition in playing order: [S, R, OR, OL, L]"
        return [RelativePosition.SELF, RelativePosition.RIGHT, RelativePosition.OPPOR, RelativePosition.OPPOL, RelativePosition.LEFT]

class AbsolutePosition(str, Enum):
    # NORTH = 'N'
    # SOUTH = 'S'
    # EAST = 'E'
    # WEST = 'W'
    # Number order counterclockwise
    ONE = '1'
    TWO = '2'
    THREE = '3'
    FOUR = '4'
    FIVE = '5'

    def relative_to(self, position: 'AbsolutePosition') -> RelativePosition:
        "Helper function that converts an absolute seat position to a relative seat position."
        return ['S', 'L', 'OL', 'R', 'OR'][(['1', '2', '3', '4', '5'].index(position) - ['1', '2', '3', '4', '5'].index(self)) % 5]

    @property
    def next_position(self):
        return {'2': AbsolutePosition.THREE, '3': AbsolutePosition.FOUR, '4': AbsolutePosition.FIVE, '5': AbsolutePosition.ONE, '1': AbsolutePosition.TWO}[self]
    
    @property
    def last_position(self):
        return {'2': AbsolutePosition.ONE, '3': AbsolutePosition.TWO, '4': AbsolutePosition.THREE, '5': AbsolutePosition.FOUR, '1': AbsolutePosition.FIVE}[self]

    @classmethod
    def in_order(self):
        return [AbsolutePosition.ONE, AbsolutePosition.TWO, AbsolutePosition.THREE, AbsolutePosition.FOUR, AbsolutePosition.FIVE]
    
    @classmethod
    def random(self):
        return random.choice(AbsolutePosition.in_order())

class Stage(str, Enum):
    declare_stage = 'DECLARE'
    kitty_stage = 'KITTY'
    chaodi_stage = 'CHAODI'
    name_stage = 'NAME'
    main_stage = 'PLAY'

class Ordinality(str, Enum):
    FIRST = '1st'
    SECOND = '2nd'

class FriendCard:
    "Contains information about the named friend card."
    def __init__(self, suit: CardSuit, rank: int, ord: Ordinality):
        self.suit = suit
        self.rank = rank
        self.ord = ord
        self.times_played = 0

    def __repr__(self) -> str:
        return f"FriendCard({self.ord} {self.suit} {self.rank})"

    @property
    def card(self):
        return LETTER_RANK[self.rank] + self.suit.value

    @property
    def tensor(self):
        "Return a fixed size binary tensor of shape (57,) representing the friend card (54-jokers), ord (binary feature), times played (3 values for 0, 1, 2 instances), and whether friend is public."
        rep = torch.zeros(58) 
        rep[ORDERING_INDEX[self.card]] = 1
        rep[52] = int(self.ord == Ordinality.SECOND)
        rep[53 + self.times_played] = 1
        rep[56] = int(self.public)
        return rep

    @property
    def public(self):
        "Whether friend card has been played and teams publicly determined."
        return (self.ord == Ordinality.FIRST and self.times_played >= 1 or 
                self.ord == Ordinality.SECOND and self.times_played == 2)

def abs_positions_excluding(excluded: List[AbsolutePosition]):
    all_positions = AbsolutePosition.in_order()
    return [p for p in all_positions if p not in excluded]

def determine_teams(friend_position: AbsolutePosition, dealer_position: AbsolutePosition):
    "Finds the positions of the defending and opposing teams after friend card played."
    if friend_position == dealer_position:
        defender_team = [dealer_position] # Dealer played friend card
    else:
        defender_team = [dealer_position, friend_position]
    return defender_team, abs_positions_excluding(defender_team)

class Declaration:
    "Contains information about the trump suit being declared."
    def __init__(self, suit: TrumpSuit, level: int, position: AbsolutePosition, relative_position: RelativePosition = None) -> None:
        self.suit = suit
        self.level = level
        self.absolute_position: AbsolutePosition = position
        self.relative_position = relative_position # Depends on the position of the player that observes this declaration

    def __repr__(self) -> str:
        return f"Declaration(player={self.absolute_position}, cards={self.suit.value} x{1 + int(self.level > 1)})"
    
    def relative_to(self, position: AbsolutePosition):
        return Declaration(self.suit, self.level, self.absolute_position, self.absolute_position.relative_to(position))
    
    @property
    def tensor(self):
        "A tensor of shape (7,) representing the suit and multiplicity of the declaration."
        return torch.cat([self.suit.tensor, torch.tensor([int(self.level > 1)])])
    
    def get_card(self, dominant_rank: int):
        rank_symbol = LETTER_RANK[dominant_rank]
        count = 1 if self.level == 0 else 2
        if self.suit == TrumpSuit.XJ or self.suit == TrumpSuit.DJ:
            return self.suit.value, count
        else:
            return rank_symbol + self.suit.value, count


    @classmethod
    def chaodi_level(self, suit: TrumpSuit, level: int):
        if level >= 1:
            if suit == TrumpSuit.DIAMOND:
                return 1
            elif suit == TrumpSuit.CLUB:
                return 2
            elif suit == TrumpSuit.HEART:
                return 3
            elif suit == TrumpSuit.SPADE:
                return 4
            elif suit == TrumpSuit.XJ:
                return 5
            else: # DJ
                return 6
        else:
            return 0

    
LETTER_RANK = {
    2: '2',
    3: '3',
    4: '4',
    5: '5',
    6: '6',
    7: '7',
    8: '8',
    9: '9',
    10: '10',
    11: 'J',
    12: 'Q',
    13: 'K',
    14: 'A',
}

NUMERIC_RANK = {v:k for k,v in LETTER_RANK.items()}

def get_suit(card: str, dominant_suit: TrumpSuit, dominant_rank: int):
    "Determines if the card is a trump card, and if not, determines which suit it is in."
    if card == 'XJ' or card == 'DJ':
        return CardSuit.TRUMP
    
    rank = NUMERIC_RANK[card[:-1]]
    suit = CardSuit(card[-1])
    
    if rank == dominant_rank or suit == dominant_suit:
        return CardSuit.TRUMP
    else:
        return suit

def get_rank(card: str, dominant_suit: TrumpSuit, dominant_rank: int):
    "Get the rank of a card within its suit."

    if card == 'DJ':
        return 18
    elif card == 'XJ':
        return 17
    else:
        suit = CardSuit(card[-1])
        rank = NUMERIC_RANK[card[:-1]]
        if rank == dominant_rank and (suit == dominant_suit or dominant_suit.is_NT):
            return 16
        elif rank == dominant_rank:
            return 15
        elif rank < dominant_rank:
            return rank + 1 # shift the rank of cards smaller than dominant rank by 1 to support tractors across the dominant rank
        else:
            return rank

def softmax(arr):
    return np.exp(arr) / np.sum(np.exp(arr))