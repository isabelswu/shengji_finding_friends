# Encapsulates all the information that's available to a player during a game.

from typing import List, Tuple, Dict
from env.Actions import Action
from env.CardSet import CardSet
from env.utils import LETTER_RANK, AbsolutePosition, Declaration, RelativePosition, Stage, TrumpSuit, FriendCard
import torch

class Observation:
    def __init__(self, hand: CardSet, position: AbsolutePosition, actions: List[Action], stage: Stage, dominant_rank: int, declaration: Declaration, next_declaration_turn: RelativePosition, dealer_position: RelativePosition, unknown_points: int, defender_points: int, opponent_points: int, individual_points: Dict[RelativePosition, int], friend_card: FriendCard, opponent_team: List[RelativePosition], defender_team: List[RelativePosition], round_history: List[Tuple[RelativePosition, List[CardSet]]], unplayed_cards: CardSet, leads_current_trick: bool, chaodi_times: List[int], kitty: CardSet = None, is_chaodi_turn = False, perceived_left = CardSet(), perceived_right = CardSet(), perceived_opleft = CardSet(), perceived_opright = CardSet(), actual_left = CardSet(), actual_right = CardSet(), actual_opleft = CardSet(), actual_opright = CardSet(), oracle_value=0.0) -> None:
        self.hand = hand
        self.position = position
        self.actions = actions
        self.stage = stage
        self.dominant_rank = dominant_rank
        self.declaration = declaration
        self.next_declaration_turn = next_declaration_turn
        self.dealer = dealer_position
        self.unknown_points = unknown_points
        self.defender_points = defender_points
        self.opponent_points = opponent_points
        self.individual_points = individual_points
        self.friend_card = friend_card
        self.opponent_team = opponent_team
        self.defender_team = defender_team
        # self.unknown_team = unknown_team
        self.round_history = round_history
        self.unplayed_cards = unplayed_cards
        self.leads_current_round = leads_current_trick # If the player is going to lead the next trick
        self.chaodi_times = chaodi_times
        self.kitty = kitty # Only observable to the last person who placed the kitty. In chaodi mode, this might not be the dealer.
        self.perceived_left = perceived_left
        self.perceived_right = perceived_right
        self.perceived_opleft = perceived_opleft
        self.perceived_opright = perceived_opright
        self.actual_left = actual_left
        self.actual_right = actual_right
        self.actual_opleft = actual_opleft
        self.actual_opright = actual_opright
        self.historical_rounds = 14 # TODO
        self.oracle_value = oracle_value # Bernoulli variable parameter
        self.is_chaodi_turn = is_chaodi_turn

    def __repr__(self) -> str:
        return f"Observation({self.position.value}, hand={self.hand})"
    
    @property
    def dominant_suit(self):
        return self.declaration.suit if self.declaration else TrumpSuit.XJ

    # Mod for ff
    @property
    def points_tensor(self):
        """Returns a (120,) tensor representing the current team-wide point situation. First 40 values is for the defender team, second 40 for the opponent team, third 40 for the unknown players.
        Unknown points zeroed out after friend made public."""
        defender_points_tensor = torch.zeros(40)
        opponent_points_tensor = torch.zeros(40)
        unknown_points_tensor = torch.zeros(40)
        defender_points_tensor[:(self.defender_points // 5)] = 1
        # if self.opponent_points is not None:
        if self.friend_card.public and self.opponent_points is not None:
            opponent_points_tensor[:(self.opponent_points // 5)] = 1
        elif self.unknown_points is not None:
            unknown_points_tensor[:(self.unknown_points // 5)] = 1
        assert (self.opponent_points is None and self.unknown_points is not None
                or self.opponent_points is not None and self.unknown_points is None) 
        return torch.cat([defender_points_tensor, opponent_points_tensor, unknown_points_tensor])
    
    # Mod for ff
    @property
    def individual_points_tensor(self):
        "Returns a (200,) tensor representing individual points for each of the 5 players relative to the current player. Zeroed out after friend made public."
        # Order: SELF, RIGHT, OPPOR, OPPOL, LEFT
        if self.friend_card.public:
            return torch.zeros(200)
        else: 
            individual_tensor = torch.empty(0)
            for pos in RelativePosition.in_order():
                current_tensor = torch.zeros(40)
                current_tensor[:(self.individual_points[pos] // 5)] = 1
                individual_tensor = torch.cat([individual_tensor, current_tensor])
            return individual_tensor
    
    # Mod for ff
    @property
    def teams_tensor(self):
        """Returns a (15,) tensor representing team membership. 3 sets of 5 values where the 1st, 2nd, and 3rd
        sets are for the defender, opponent, and unknown teams respectively. Each set is a multi-hot encoding
        for player membership based on their position relative to the current player. [S, R, OR, OL, L, ..., ...]"""
        defender_tensor = torch.zeros(5)
        opponent_tensor = torch.zeros(5)
        unknown_tensor = torch.zeros(5)

        for i, pos in enumerate(RelativePosition.in_order()):
            if self.friend_card.public:
                if pos in self.defender_team:
                    defender_tensor[i] = 1
                else:
                    opponent_tensor[i] = 1
            else: 
                if pos == self.dealer:
                    defender_tensor[i] = 1
                else:
                    unknown_tensor[i] = 1
            return torch.cat([defender_tensor, opponent_tensor, unknown_tensor])
    
    # Mod for ff
    @property
    def friend_card_tensor(self):
        """Returns information about the freind card. Shape: (57,)"""
        return self.friend_card.tensor
    
    # Mod for ff
    @property
    def dealer_position_tensor(self):
        "Returns a (5,) one-hot tensor representing the dealer's position relative to the player."
        pos = torch.zeros(5)
        if not self.dealer:
            return pos
        index = RelativePosition.in_order().index(self.dealer)
        pos[index] = 1
        return pos
    
    # Mod for ff
    @property
    def declarer_position_tensor(self):
        "Returns a (5,) one-hot tensor representing the location of the declarer relative to self."
        pos = torch.zeros(5)
        if not self.declaration:
            return pos
        index = index = RelativePosition.in_order().index(self.declaration.relative_position)
        pos[index] = 1
        return pos
    
    @property
    def trump_tensor(self):
        "Returns a (20,) tensor representing the current trump suit and trump rank."
        rank_tensor = torch.zeros(13)
        rank_tensor[self.dominant_rank - 2] = 1

        return torch.cat([self.declaration.tensor if self.declaration else torch.zeros(7), rank_tensor])

    @property
    def kitty_tensor(self):
        "If the player buried the kitty, return information about the kitty. Otherwise, return an empty matrix. Shape: (108,)"
        return self.kitty.tensor if self.kitty is not None else torch.zeros(108)
    
    @property
    def kitty_dynamic_tensor(self):
        "If the player buried the kitty, return information about the kitty. Otherwise, return an empty matrix. Shape: (108,)"
        return self.kitty.get_dynamic_tensor(self.dominant_suit, self.dominant_rank) if self.kitty is not None else torch.zeros(108)
    
    @property
    def dynamic_hand_tensor(self):
        return self.hand.get_dynamic_tensor(self.declaration.suit if self.declaration else TrumpSuit.XJ, self.dominant_rank)
    
    @property
    def unplayed_cards_tensor(self):
        "Returns a (108,) tensor representing all cards not played (and not owned by the current player)."
        unplayed = self.unplayed_cards.copy()
        if self.kitty:
            unplayed.remove_cardset(self.kitty)
        unplayed.remove_cardset(self.hand)
        return unplayed.tensor
    
    @property
    def unplayed_cards_dynamic_tensor(self):
        "Returns a (108,) tensor representing all cards not played (and not owned by the current player)."
        unplayed = self.unplayed_cards.copy()
        if self.kitty:
            unplayed.remove_cardset(self.kitty)
        unplayed.remove_cardset(self.hand)
        return unplayed.get_dynamic_tensor(self.dominant_suit, self.dominant_rank)
    
    # Only used in oracle_cardsets
    @property
    def perceived_cardsets(self):
        "Returns the cards for each player from the current player's perspective, starting from themselves going anti-clickwise. Shape: (540,)"
        # For 5 players: RIGHT, OPPOR, OPPOL, LEFT (excluding SELF)
        return torch.cat([
            self.perceived_right.get_dynamic_tensor(self.dominant_suit, self.dominant_rank),
            self.perceived_opright.get_dynamic_tensor(self.dominant_suit, self.dominant_rank),
            self.perceived_opleft.get_dynamic_tensor(self.dominant_suit, self.dominant_rank),
            self.perceived_left.get_dynamic_tensor(self.dominant_suit, self.dominant_rank)
        ])
    
    # Only used if agent.use_oracle true
    @property
    def oracle_cardsets(self):
        perfect_info = torch.cat([
            self.actual_right.get_dynamic_tensor(self.dominant_suit, self.dominant_rank),
            self.actual_opright.get_dynamic_tensor(self.dominant_suit, self.dominant_rank),
            self.actual_opleft.get_dynamic_tensor(self.dominant_suit, self.dominant_rank),
            self.actual_left.get_dynamic_tensor(self.dominant_suit, self.dominant_rank)
        ])
        # Mask using Bernoulli random variables
        mask = torch.bernoulli(torch.ones_like(perfect_info) * self.oracle_value)
        return torch.maximum(perfect_info * mask, self.perceived_cardsets)

    @property
    def perceived_trump_cardsets(self):
        "Return a (48,) tensor describing the dominant rank trump cards each of the 4 other players are known to have."
        diamond_card = LETTER_RANK[self.dominant_rank] + TrumpSuit.DIAMOND
        club_card = LETTER_RANK[self.dominant_rank] + TrumpSuit.CLUB
        heart_card = LETTER_RANK[self.dominant_rank] + TrumpSuit.HEART
        spade_card = LETTER_RANK[self.dominant_rank] + TrumpSuit.SPADE

        trump_card_counts = []

        for cardset in [self.perceived_right, self.perceived_opright, self.perceived_opleft, self.perceived_left]:
            card_vector = torch.zeros(12)
            for i, trump_card in enumerate([diamond_card, club_card, heart_card, spade_card, 'XJ', 'DJ']):
                card_vector[i * 2 : i * 2 + cardset._cards[trump_card]] = 1
            trump_card_counts.append(card_vector)
        
        return torch.cat(trump_card_counts)
    
    @property
    def historical_moves_tensor(self):
        "Returns two tensors of shape (15, 545), (437,) (update for 5 players) representing the historical rounds of the current game."
        # For 5 players: 5 position indicators + 5 * 108 card tensors = 545 per round
        history_tensor = torch.zeros((min(15, len(self.round_history)), 5 + 5 * 108))
        for i, (pos, round) in enumerate(self.round_history[-self.historical_rounds - 1:]):
            current_player_index = RelativePosition.in_order().index(pos)
            history_tensor[i, current_player_index] = 1
            for cardset in round:
                history_tensor[i, 5 + 108 * current_player_index : 5 + 108 * (current_player_index + 1)] = cardset.tensor
                current_player_index = (current_player_index + 1) % 5
        
        padded_history = torch.vstack([
            torch.zeros((15 - history_tensor.shape[0], 545)),
            history_tensor
        ])
        return padded_history, torch.cat([history_tensor[-1][:5], history_tensor[-1][113:]])
    
    @property
    def historical_moves_dynamic_tensor(self):
        "Returns two tensor representing the historical rounds of the current game."
        # For 5 players: 5 position indicators + 5 * 108 card tensors = 545 per round
        history_tensor = torch.zeros((min(15, len(self.round_history)), 5 + 5 * 108))
        position_order = [RelativePosition.SELF, RelativePosition.RIGHT, RelativePosition.OPPOR, RelativePosition.OPPOL, RelativePosition.LEFT]
        for i, (pos, round) in enumerate(self.round_history[-self.historical_rounds - 1:]):
            current_player_index = position_order.index(pos)
            history_tensor[i, current_player_index] = 1
            for cardset in round:
                history_tensor[i, 5 + 108 * current_player_index : 5 + 108 * (current_player_index + 1)] = cardset.get_dynamic_tensor(self.dominant_suit, self.dominant_rank)
                current_player_index = (current_player_index + 1) % 5
        
        padded_history = torch.vstack([
            torch.zeros((15 - history_tensor.shape[0], 5 + 5 * 108)),
            history_tensor
        ])
        return padded_history, torch.cat([history_tensor[-1][:5], history_tensor[-1][113:]])

    @property
    def chaodi_times_tensor(self):
        chaodi_times = torch.zeros(5)
        positions = ['1', '2', '3', '4', '5']
        current_position = self.position
        for i in range(5):
            idx = positions.index(current_position.value)
            chaodi_times[i] = self.chaodi_times[idx]
            current_position = current_position.next_position
        
        return chaodi_times
    
    # Nothing uses this
    # @property
    # def current_dominating_player_index(self):
    #     encoding = torch.zeros(3) # first 3 represent which players have played. last 3 represent who's the biggest
    #     if self.round_history[-1][1]:
    #         winning_index = CardSet.round_winner(self.round_history[-1][1], self.declaration.suit if self.declaration else TrumpSuit.XJ, self.dominant_rank)
    #         encoding[3 - len(self.round_history[-1][1]) + winning_index] = 1
    #     return encoding

    def dominates_all_tensor(self, cardset: CardSet):
        if CardSet.round_winner(self.round_history[-1][1] + [cardset], self.declaration.suit if self.declaration else TrumpSuit.XJ, self.dominant_rank) == len(self.round_history[-1][1]):
            return torch.tensor([1])
        else:
            return torch.tensor([0])
