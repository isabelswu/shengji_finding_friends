# Encapsulates all the information that's available to a player during a game.

from typing import List, Tuple, Dict
from env.Actions import Action
from env.CardSet import CardSet
from env.utils import LETTER_RANK, AbsolutePosition, Declaration, RelativePosition, Stage, TrumpSuit
import torch

class Observation:
    def __init__(self, hand: CardSet, position: AbsolutePosition, actions: List[Action], stage: Stage, dominant_rank: int, declaration: Declaration, next_declaration_turn: RelativePosition, dealer_position: RelativePosition, defender_points: int, opponent_points: int, individual_points: Dict[RelativePosition, int], friend_found: bool, opponent_team: List[RelativePosition], defender_team: List[RelativePosition], round_history: List[Tuple[RelativePosition, List[CardSet]]], unplayed_cards: CardSet, leads_current_trick: bool, chaodi_times: List[int], kitty: CardSet = None, is_chaodi_turn = False, perceived_left = CardSet(), perceived_right = CardSet(), perceived_opleft = CardSet(), perceived_opright = CardSet(), actual_left = CardSet(), actual_right = CardSet(), actual_opleft = CardSet(), actual_opright = CardSet(), oracle_value=0.0) -> None:
        self.hand = hand
        self.position = position
        self.actions = actions
        self.stage = stage
        self.dominant_rank = dominant_rank
        self.declaration = declaration
        self.next_declaration_turn = next_declaration_turn
        self.dealer = dealer_position
        self.defender_points = defender_points
        self.opponent_points = opponent_points
        self.individual_points = individual_points
        self.friend_found = friend_found
        self.opponent_team = opponent_team
        self.defender_team = defender_team
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

    @property
    def points_tensor(self):
        "Returns a (80,) tensor representing the current point situation as observed by the player. First 40 values is for the player's team, last 40 for the other team."
        defenders_points_tensor = torch.zeros(40)
        opponents_points_tensor = torch.zeros(40)
        defenders_points_tensor[:(self.defender_points // 5)] = 1
        opponents_points_tensor[:(self.opponent_points // 5)] = 1

        if RelativePosition.SELF in self.defender_team:
            return torch.cat([defenders_points_tensor, opponents_points_tensor])
        else
           return torch.cat([opponents_points_tensor, defenders_points_tensor])
    
    @property
    def individual_points_tensor(self):
        "Returns a (20,) tensor representing individual points for each of the 5 players relative to the current player."
        # Order: SELF, RIGHT, OPPOR, OPPOL, LEFT
        individual_tensor = torch.zeros(20)  # 5 players * 4 buckets (0-20 points each)
        position_order = [RelativePosition.SELF, RelativePosition.RIGHT, RelativePosition.OPPOR, RelativePosition.OPPOL, RelativePosition.LEFT]
        
        for i, pos in enumerate(position_order):
            if pos in self.individual_points:
                points = self.individual_points[pos]
                # Each player gets 4 buckets: 0-5, 5-10, 10-15, 15-20
                bucket = min(points // 5, 3)
                individual_tensor[i * 4 + bucket] = 1
        
        return individual_tensor
    
    @property
    def team_membership_tensor(self):
        "Returns a (5,) tensor representing team membership: 1 for defender team, 0 for attacker team, -1 if teams not determined."
        team_tensor = torch.zeros(5)
        position_order = [RelativePosition.SELF, RelativePosition.RIGHT, RelativePosition.OPPOR, RelativePosition.OPPOL, RelativePosition.LEFT]
        
        if self.friend_found:
            for i, pos in enumerate(position_order):
                if pos in self.defender_team:
                    team_tensor[i] = 1
                elif pos in self.opponent_team:
                    team_tensor[i] = 0
                # If pos not in either team (shouldn't happen), stays 0
        else:
            # Teams not determined - use -1 to indicate unknown
            team_tensor.fill_(-1)
        
        return team_tensor
    
    @property
    def dealer_position_tensor(self):
        "Returns a (5,) one-hot tensor representing the dealer's position relative to the player."
        pos = torch.zeros(5)

        if not self.dealer:
            return pos
        
        index = [RelativePosition.SELF, RelativePosition.RIGHT, RelativePosition.OPPOR, RelativePosition.OPPOL, RelativePosition.LEFT].index(self.dealer)
        pos[index] = 1
        return pos
    
    @property
    def declarer_position_tensor(self):
        "Returns a (5,) one-hot tensor representing the location of the declarer relative to self."
        pos = torch.zeros(5)
        if not self.declaration:
            return pos
        index = [RelativePosition.SELF, RelativePosition.RIGHT, RelativePosition.OPPOR, RelativePosition.OPPOL, RelativePosition.LEFT].index(self.declaration.relative_position)
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
        "Returns two tensors of shape ??  (20, 436), (328,) (update for 5 players) representing the historical rounds of the current game."
        # For 5 players: 5 position indicators + 5 * 108 card tensors = 545 per round
        history_tensor = torch.zeros((min(15, len(self.round_history)), 5 + 5 * 108))
        position_order = [RelativePosition.SELF, RelativePosition.RIGHT, RelativePosition.OPPOR, RelativePosition.OPPOL, RelativePosition.LEFT]
        for i, (pos, round) in enumerate(self.round_history[-self.historical_rounds - 1:]):
            current_player_index = position_order.index(pos)
            history_tensor[i, current_player_index] = 1
            for cardset in round:
                history_tensor[i, 5 + 108 * current_player_index : 5 + 108 * (current_player_index + 1)] = cardset.tensor
                current_player_index = (current_player_index + 1) % 5
        
        padded_history = torch.vstack([
            torch.zeros((15 - history_tensor.shape[0], 5 + 5 * 108)),
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
    
    @property
    def current_dominating_player_index(self):
        encoding = torch.zeros(4) # first 4 represent which players have played. last 4 represent who's the 
        biggest
        if self.round_history and len(self.round_history[-1][1]) > 0:
            winning_index = CardSet.round_winner(self.round_history[-1][1], self.declaration.suit if self.declaration else TrumpSuit.XJ, self.dominant_rank)
            encoding[4 - len(self.round_history[-1][1]) + winning_index] = 1
        return encoding

    def dominates_all_tensor(self, cardset: CardSet):
        if CardSet.round_winner(self.round_history[-1][1] + [cardset], self.declaration.suit if self.declaration else TrumpSuit.XJ, self.dominant_rank) == len(self.round_history[-1][1]):
            return torch.tensor([1])
        else:
            return torch.tensor([0])

