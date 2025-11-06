# Describes the current state of a Shengji game. Obeys Markov's property.

from math import ceil
import random
from typing import List, Tuple, Union
from env.Actions import Action, AppendLeadAction, ChaodiAction, DeclareAction, DontChaodiAction, DontDeclareAction, EndLeadAction, FollowAction, LeadAction, NameFriendCard, PlaceAllKittyAction, PlaceKittyAction

from env.Observation import Observation
from env.utils import *
from .CardSet import CardSet, MoveType
import logging

class Game:
    def __init__(self, dominant_rank=2, dealer_position: AbsolutePosition = None, enable_chaodi = False, enable_combos = False, deck: List[str] = None, is_warmup_game=False, oracle_value=0.0, combo_penalty=0.1, combo_alternation=False) -> None:
        # Player information
        self.hands = {
            AbsolutePosition.ONE: CardSet(),
            AbsolutePosition.TWO: CardSet(),
            AbsolutePosition.THREE: CardSet(),
            AbsolutePosition.FOUR: CardSet(),
            AbsolutePosition.FIVE: CardSet()
        }
        self.public_cards = {
            AbsolutePosition.ONE: CardSet(),
            AbsolutePosition.TWO: CardSet(),
            AbsolutePosition.THREE: CardSet(),
            AbsolutePosition.FOUR: CardSet(),
            AbsolutePosition.FIVE: CardSet()
        }
        self.unplayed_cards, self.card_list = CardSet.new_deck()
        if is_warmup_game:
            self.card_list = CardSet.get_tutorial_deck()
        else:
            self.card_list = deck or self.card_list
        self.deck = iter(self.card_list) # First 100 are drawn, last 8 given to dealer.
        self.kitty = CardSet() # dealer puts the kitty in 8 rounds, one card per round

        # Public game information
        self.round_history: List[Tuple[AbsolutePosition, List[CardSet]]] = []
        "A list of past rounds, each having the structure (P, [CardSet...]), where P is one of '1', '2', '3', '4', or '5', and CardSets are in the order they were played (so for instance, element 0 is played by P)."
        
        self.stage = Stage.declare_stage
        self.kitty_stage_completed = False # True if the kitty is fixed.
        self.kitty_owner = dealer_position
        self.dominant_rank = dominant_rank
        self.declarations: List[Declaration] = [] # Information about who declared the trump suit, what it is, and its level
        self.current_declaration_turn: AbsolutePosition = None # Once a player declares a trump suit, every other player takes turn to decide if they want to override.
        self.initial_declaration_position: AbsolutePosition = None # Record the position of the first player that started the declaration round while drawing cards
        self.is_initial_game = dealer_position is None # Whether overriding the declaration lets the overrider becomes the dealer.
        self.dealer_position = dealer_position # Position of the dealer. At the start of game 1, the dealer is not determined yet.
        self.unknown_points = 0  # Multiples of 5, None after friend public
        self.opponent_points = None # None until friend public
        self.defender_points = 0 # Dealer's points until friend public
        self.game_ended = False
        self.kitty_multiplier = None

        # Finding friends fields
        self.friend_card: FriendCard = None 
        self.unknown_team: List[AbsolutePosition] = [] # All players but dealer before friend becomes public
        self.defender_team: List[AbsolutePosition] = [] # Dealer and friend
        self.opponent_team: List[AbsolutePosition] = [] # Remaining players
        self.individual_points = {  # Individual points collected before teams determined
            AbsolutePosition.ONE: 0,
            AbsolutePosition.TWO: 0,
            AbsolutePosition.THREE: 0,
            AbsolutePosition.FOUR: 0,
            AbsolutePosition.FIVE: 0
        }
        self.pooled_defender_points = None
        self.pooled_opponent_points = None

        self.points_per_round: List[int] = [] # Positive means defenders escaped points; negative means opponents scored points
        self.final_opponent_reward: float = None
        self.final_defender_reward: float = None
        
        # Chaodi mode, TODO: update for 5 players
        self.enable_chaodi = enable_chaodi
        self.current_chaodi_turn: AbsolutePosition = None
        self.initial_chaodi_position: AbsolutePosition = None
        self.chaodi_times = [0, 0, 0, 0] # In the order N, W, S, E

        # Combo mode
        self.enable_combos = enable_combos

        self.draw_order = []
        self.is_warmup_game = is_warmup_game # In a warm up game, we don't allow declarations for fairness
        self.oracle_value = oracle_value # the coefficient for the oracle
        self.consecutive_moves = 0
        self.combo_penalty = combo_penalty
        self.combo_alternation = combo_alternation
    @property
    def dominant_suit(self):
        return self.declarations[-1].suit if self.declarations else TrumpSuit.XJ

    @property
    def game_started(self):
        return sum([c.size for c in self.hands.values()]) > 0 or self.game_ended
        
    def start_game(self):
        # Finally, we kick off the game by letting one of the players be the first to draw a card from the deck.
        first_player = self.dealer_position or AbsolutePosition.random()
        next_card = next(self.deck)
        self.hands[first_player].add_card(next_card)
        self.draw_order.append((first_player, next_card))
        logging.debug(f"Starting new game with dominant rank {self.dominant_rank}, first player is {first_player}, dealer is {self.dealer_position}")
        return first_player

    def get_observation(self, position: AbsolutePosition) -> Observation:
        "Derive the current observation for a given player."

        # Compute actions
        actions: List[Action] = []

        if self.stage == Stage.declare_stage: # Stage 1: drawing cards phase
            if not self.is_warmup_game:
                for suit, level in self.hands[position].trump_declaration_options(self.dominant_rank).items():
                    if not self.declarations or self.declarations[-1].level < level and (self.declarations[-1].suit == suit or self.declarations[-1].absolute_position != position):
                        actions.append(DeclareAction(Declaration(suit, level, position)))
            actions.append(DontDeclareAction())
        elif self.hands[position].size > 25: # Stage 2: choosing the kitty
            for card, count in self.hands[position]._cards.items():
                if count > 0: actions.append(PlaceKittyAction(card, count))
        elif self.current_chaodi_turn == position:
            # Chaodi
            if self.enable_chaodi and self.declarations:
                for suit, level in self.hands[position].trump_declaration_options(self.dominant_rank).items():
                    if self.declarations and Declaration.chaodi_level(suit, level) > Declaration.chaodi_level(self.declarations[-1].suit, self.declarations[-1].level):
                        actions.append(ChaodiAction(Declaration(suit, level, position)))
                        logging.debug(f"{position.value} can chaodi using {suit.value}")
            actions.append(DontChaodiAction())
        elif self.stage == Stage.name_stage: # Stage 3: name friend card phase
            assert position == self.dealer_position, "Only dealer acts in naming stage"
            non_dominant_suits = [CardSuit.CLUB, CardSuit.SPADE, CardSuit.HEART, CardSuit.DIAMOND]
            if self.dominant_suit == TrumpSuit.CLUB:
                non_dominant_suits.remove(CardSuit.CLUB)
            elif self.dominant_suit == TrumpSuit.SPADE:
                non_dominant_suits.remove(CardSuit.SPADE)
            elif self.dominant_suit == TrumpSuit.HEART:
                non_dominant_suits.remove(CardSuit.HEART)
            elif self.dominant_suit == TrumpSuit.DIAMOND:
                non_dominant_suits.remove(CardSuit.DIAMOND)
            for suit in non_dominant_suits:
                for rank in range(2, 15):
                    if rank != self.dominant_rank:
                        for ord in [Ordinality.FIRST, Ordinality.SECOND]:
                            actions.append(NameFriendCard(FriendCard(suit, rank, ord)))
        elif self.stage == Stage.main_stage and self.round_history[-1][0] == position:
            # In combo alternation mode, players at positions East or West cannot play combo moves
            if not self.enable_combos or not self.round_history[-1][1]:
                ban_combos = self.combo_alternation and position in (AbsolutePosition.EAST, AbsolutePosition.WEST)
                for move in self.hands[position].get_leading_moves(self.dominant_suit, self.dominant_rank):
                    actions.append(LeadAction(move) if (self.enable_combos and not ban_combos) else EndLeadAction(move))
            else:
                current_action = self.round_history[-1][1][0]
                current_action_suit = get_suit(current_action.card_list()[0], self.dominant_suit, self.dominant_rank)
                remaining_cards = self.hands[position].filter_by_suit(current_action_suit, self.dominant_suit, self.dominant_rank)
                remaining_cards.remove_cardset(current_action)
                # complement = self.unplayed_cards.copy()
                # complement.remove_cardset(current_action)
                if self.consecutive_moves < 3:
                    for move in remaining_cards.get_leading_moves(self.dominant_suit, self.dominant_rank):
                        actions.append(AppendLeadAction(current_action, move))
                actions.append(EndLeadAction(MoveType.Combo(current_action)))
        elif self.stage == Stage.main_stage:
            # Combo is a catch-all type if we don't know the composition of the cardset
            for cardset in self.hands[position].get_matching_moves(MoveType.Combo(self.round_history[-1][1][0]), self.dominant_suit, self.dominant_rank):
                actions.append(FollowAction(cardset))
        assert actions, f"Agent {position} has no action to choose from!"

        relative_points: dict[RelativePosition, int] = {}
        for abspos, pts in self.individual_points.keys():
            relative_points[abspos.relative_to(position)] = pts

        observation = Observation(
            hand = self.hands[position].copy(),
            position = position,
            actions = actions,
            stage = self.stage,
            dominant_rank = self.dominant_rank,
            declaration = self.declarations[-1].relative_to(position) if self.declarations else None,
            next_declaration_turn = self.current_declaration_turn.relative_to(position) if self.current_declaration_turn else None,
            dealer_position = self.dealer_position.relative_to(position) if self.dealer_position else None,
            unknown_points = self.unknown_points,
            opponent_points = self.opponent_points,
            defender_points = self.defender_points,
            individual_points = relative_points,
            friend_card = self.friend_card,
            unknown_team = [p.relative_to(position) for p in self.unknown_team] if self.unknown_team else None,
            opponent_team = [p.relative_to(position) for p in self.opponent_team] if self.opponent_team else None,
            defender_team = [p.relative_to(position) for p in self.defender_team],
            round_history = [(p.relative_to(position), cards[:]) for p, cards in self.round_history],
            unplayed_cards = self.unplayed_cards.copy(),
            leads_current_trick = self.round_history[-1][0] == position if self.round_history else position == self.dealer_position,
            chaodi_times = self.chaodi_times,
            kitty = self.kitty.copy() if position == self.kitty_owner else None,
            is_chaodi_turn = self.current_chaodi_turn == position,
            perceived_left = self.public_cards[position.last_position].copy(),
            perceived_right = self.public_cards[position.next_position].copy(),
            perceived_opleft = self.public_cards[position.last_position.last_position].copy(),  
            perceived_opright = self.public_cards[position.next_position.next_position].copy(),
            actual_left=self.hands[position.last_position].copy(),
            actual_right=self.hands[position.next_position].copy(),
            actual_opleft=self.hands[position.last_position.last_position].copy(),
            actual_opright=self.hands[position.next_position.next_position].copy(),
            oracle_value=self.oracle_value)

        return observation

    def pool_team_points(self):
        "Pool individual points into team points"
        self.unknown_points = None
        self.pooled_defender_points = 0
        self.pooled_opponent_points = 0

        for pos in self.defender_team:
            if pos != self.dealer_position:
                self.defender_points += self.individual_points[pos]
                self.pooled_defender_points += self.individual_points[pos]

        assert self.opponent_points is None and self.opponent_team is not None
        self.opponent_points = 0
        for pos in self.opponent_team:
            self.opponent_points += self.individual_points[pos]
        self.pooled_opponent_points = self.opponent_points

        logging.debug(f"Pooled individual points. Defender points: {self.defender_points}, Opponent points: {self.opponent_points}") 

    def update_find_friends(self, action: Action, player_position: AbsolutePosition) -> None:
        """For EndAction or FollowAction, if the friend has not been made public, update ordinality
           of friend card if it was played. If friend public, then determine teams and pool individual points."""
        if not self.friend_card.public and action.move.cardset.has_card(self.friend_card.card):
            self.friend_card.times_played += 1 # Update
            if self.friend_card.public:
                self.unknown_team = None
                self.defender_team, self.opponent_team = determine_teams(player_position, self.dealer_position)
                if len(self.defender_team) == 1:
                    logging.info(f"Dealer played friend card in lead. Defender alone: {self.defender_team}")
                else:
                    logging.info(f"Friend card played by {player_position} in lead. Teams: Defenders={self.defender_team}, Attackers={self.attacker_team}")
                self.pool_team_points()

    def run_action(self, action: Action, player_position: AbsolutePosition) -> Tuple[AbsolutePosition, float]:
        "Run an action on the game, and return the position of the player that needs to act next and the reward for the current action."
        if isinstance(action, DontDeclareAction):
            if self.current_declaration_turn == player_position: # It's the current player's turn to declare if he wants to. But he chooses not, so the opportunity is passed on to the next player. But if the next player is the one who made the original declaration, then the declaration is over and the dealer proceeds with swapping the kitty.
                logging.debug(f"Player {player_position.value} chose not to declare")
                if player_position.next_position == self.initial_declaration_position:
                    self.current_declaration_turn = None
                    self.stage = Stage.kitty_stage
                    if not self.dealer_position: self.dealer_position = AbsolutePosition.random()
                    for remaining_card in self.deck:
                        self.hands[self.dealer_position].add_card(remaining_card)
                    return self.dealer_position, 0
                else:
                    self.current_declaration_turn = self.current_declaration_turn.next_position
                    return self.current_declaration_turn, 0
                    
            elif sum([h.size for h in self.hands.values()]) < 100: # Distribute the next card from the deck if less than 100 are distributed.
                assert self.hands[player_position.next_position].size < 25, "Current player already has 25 cards"
                next_card = next(self.deck)
                self.hands[player_position.next_position].add_card(next_card)
                self.draw_order.append((player_position.next_position, next_card))
                return player_position.next_position, 0
            else: # Otherwise, all 100 cards are drawn from the deck. Begin one final round of declarations.
                self.initial_declaration_position = player_position.next_position
                self.current_declaration_turn = player_position.next_position
                return self.current_declaration_turn, 0
        
        elif isinstance(action, DeclareAction):
            assert not self.declarations or self.declarations[-1].level < action.declaration.level, "New trump suit declaration must have higher level than the existing one."
            assert self.hands[player_position].get_count(action.declaration.suit, self.dominant_rank) >= 1, "Invalid declaration"
            
            if self.dealer_position is None or self.is_initial_game:
                self.dealer_position = player_position # Round 1, player becomes dealer (抢庄)
                self.kitty_owner = player_position
            self.declarations.append(action.declaration)
            self.public_cards[player_position].add_card(*action.declaration.get_card(self.dominant_rank))
            logging.info(f"Player {player_position} declared {action.declaration.suit} x {1 + int(action.declaration.level >= 1)}")
            logging.info(f"Player {player_position} has cards: {self.hands[player_position]}")
            
            if sum([h.size for h in self.hands.values()]) < 100:
                # If 100 cards are not yet distributed, draw next card
                next_card = next(self.deck)
                self.hands[player_position.next_position].add_card(next_card)
                self.draw_order.append((player_position.next_position, next_card))
                return player_position.next_position, 0
            else:
                # Otherwise, other players take turns to see if they would like to redeclare.
                self.initial_declaration_position = player_position
                self.current_declaration_turn = player_position.next_position
                return self.current_declaration_turn, 0
        
        elif isinstance(action, PlaceKittyAction):
            assert self.kitty.size < 8, "Kitty already has 8 cards"

            suit_count_before = 0
            for suit in [CardSuit.CLUB, CardSuit.DIAMOND, CardSuit.HEART, CardSuit.SPADE]:
                if self.hands[player_position].count_suit(suit, self.dominant_suit, self.dominant_rank) > 0:
                    suit_count_before += 1
            trump_count_before = self.hands[player_position].count_suit(CardSuit.TRUMP, self.dominant_suit, self.dominant_rank)

            self.kitty.add_card(action.card)
            self.hands[player_position].remove_card(action.card)
            logging.debug(f"Player {player_position} discarded {action.card} to kitty")

            suit_count_after = 0
            trump_count_after = 0
            for suit in [CardSuit.CLUB, CardSuit.DIAMOND, CardSuit.HEART, CardSuit.SPADE]:
                if self.hands[player_position].count_suit(suit, self.dominant_suit, self.dominant_rank) > 0:
                    suit_count_after += 1
            trump_count_after = self.hands[player_position].count_suit(CardSuit.TRUMP, self.dominant_suit, self.dominant_rank)
            
            reward = 0.2 * (suit_count_before - suit_count_after) # Encourage players to get rid of a suit completely
            if trump_count_after < trump_count_before:
                reward -= 0.5 # Discourage players from discarding trump cards
            if get_rank(action.card, self.dominant_suit, self.dominant_rank) >= 15:
                reward -= 1 # Highly discourage players from discarding dominant rank cards or jokers
            if self.kitty.size == 8:
                # if not self.round_history:
                #     self.round_history.append((player_position, []))
                logging.debug(f"Current declaration sequence: {self.declarations}")
                logging.debug(f"Hands of all players:")
                logging.debug(f"  One: {self.hands['1']}")
                logging.debug(f"  Two: {self.hands['2']}")
                logging.debug(f"  Three: {self.hands['3']}")
                logging.debug(f"  Four: {self.hands['4']}")
                logging.debug(f"  Five: {self.hands['5']}")
                logging.debug(f"  Kitty: {self.kitty}")
                logging.info(f"Player {player_position.value} discarded kitty: {self.kitty}")
            else:
                return player_position, 0 # Current player needs to first finish placing kitty
            
            if not self.enable_chaodi:
                self.stage = Stage.name_stage
                return self.dealer_position, reward
            else:
                self.stage = Stage.chaodi_stage
                self.initial_chaodi_position = player_position
                self.current_chaodi_turn = player_position.next_position
                return player_position.next_position, reward
        
        elif isinstance(action, PlaceAllKittyAction):
            self.kitty.add_cardset(action.cards)
            self.hands[player_position].remove_cardset(action.cards)
            logging.debug(f"Player {player_position.value} discarded kitty {action.cards}")
            # if not self.round_history:
            #     self.round_history.append((player_position, []))
            
            if not self.enable_chaodi or not self.declarations:
                self.stage = Stage.name_stage
                return self.dealer_position, 0
            else:
                self.stage = Stage.chaodi_stage
                self.initial_chaodi_position = player_position
                self.current_chaodi_turn = player_position.next_position
                return player_position.next_position, 0

        elif isinstance(action, DontChaodiAction):
            logging.debug(f"Player {player_position} chose not to chaodi")
            # Note: chaodi is only an option if no one declares.
            if self.current_chaodi_turn.next_position == self.initial_chaodi_position: # We went around the table and no one chaodied.
                self.current_chaodi_turn = None
                self.stage = Stage.name_stage
                return self.dealer_position, 0
            else:
                self.current_chaodi_turn = self.current_chaodi_turn.next_position
                return self.current_chaodi_turn, 0

        elif isinstance(action, ChaodiAction):
            self.declarations.append(action.declaration)
            self.hands[player_position].add_cardset(self.kitty) # Player picks up kitty
            self.kitty.remove_cardset(self.kitty)
            self.public_cards[player_position].add_card(*action.declaration.get_card(self.dominant_rank))
            self.kitty_owner = player_position
            logging.info(f"Player {player_position} chose to chaodi using {action.declaration.suit}")
            self.stage = Stage.kitty_stage
            self.chaodi_times[['N', 'W', 'S', 'E'].index(player_position)] += 1
            if action.declaration.level == 3:
                self.current_chaodi_turn = None
                return player_position, 0
            else:
                return player_position, 0

        elif isinstance(action, NameFriendCard):
            # Dealer names the friend card
            assert player_position == self.dealer_position, "Only dealer can name the friend card"
            assert action.friend_card.rank != self.dominant_rank, "Friend card cannot be the dominant rank"
            self.friend_card = action.friend_card
            logging.info(f"Dealer {player_position} named friend card: {self.friend_card}")
            # discourage players from naming cards in the kitty or in their hand?

            self.unknown_team = [AbsolutePosition.ONE, AbsolutePosition.TWO, AbsolutePosition.THREE, AbsolutePosition.FOUR, AbsolutePosition.FIVE].remove(self.dealer_position)
            self.defender_team = [self.dealer_position]
            self.stage = Stage.main_stage
            if not self.round_history:
                self.round_history.append((self.dealer_position, []))
            return self.dealer_position, 0
        
        elif isinstance(action, LeadAction):
            logging.debug(f"Round {len(self.round_history)}: {player_position.value} places {action.move.cardset}")
            self.round_history[-1][1].append(action.move.cardset)
            self.consecutive_moves = 1
            return player_position, 0 # current player continues to place cards until EndLeadAction
        elif isinstance(action, AppendLeadAction):
            logging.debug(f"Round {len(self.round_history)}: {player_position.value} updates lead action to {action.move.cardset}")
            self.round_history[-1][1][0] = action.move.cardset
            self.consecutive_moves += 1
            return player_position, 0
        elif isinstance(action, EndLeadAction):
            logging.debug(f"Round {len(self.round_history)}: {player_position.value} leads with {action.move}")
            other_player_hands = [
                self.hands[player_position.next_position],
                self.hands[player_position.next_position.next_position],
                self.hands[player_position.next_position.next_position.next_position],
                self.hands[player_position.next_position.next_position.next_position.next_position]
            ]
            is_legal, penalty_move = CardSet.is_legal_combo(action.move, other_player_hands, self.dominant_suit, self.dominant_rank)
            if is_legal:
                if not self.round_history[-1][1]:
                    self.round_history[-1][1].append(action.move.cardset)
                self.hands[player_position].remove_cardset(action.move.cardset)
                self.unplayed_cards.remove_cardset(action.move.cardset)
                for card in action.move.cardset.card_list():
                    if self.public_cards[player_position].has_card(card):
                        self.public_cards[player_position].remove_card(card)
                self.update_find_friends(action, player_position)
                return player_position.next_position, 0
            else:
                self.round_history[-1][1].pop()
                self.round_history[-1][1].append(penalty_move)
                self.hands[player_position].remove_cardset(penalty_move)
                self.unplayed_cards.remove_cardset(penalty_move)

                # all the cards which the player failed to play are revealed to other players as public information
                failed_cards = action.move.cardset
                failed_cards.remove_cardset(penalty_move)
                for (card, count) in failed_cards.count_iterator():
                    if self.public_cards[player_position]._cards[card] < count:
                        self.public_cards[player_position]._cards[card] = count
                self.update_find_friends(action, player_position)
                logging.debug(f"Combo move failed. Player {player_position.value} forced to play {penalty_move}")
                return player_position.next_position, -self.combo_penalty
        elif isinstance(action, FollowAction):
            logging.debug(f"Round {len(self.round_history)}: {player_position.value} follows with {action.cardset}")
            lead_position, moves = self.round_history[-1]
            self.hands[player_position].remove_cardset(action.cardset)
            moves.append(action.cardset)
            self.unplayed_cards.remove_cardset(action.cardset)
            for card in action.cardset.card_list():
                if self.public_cards[player_position].has_card(card):
                    self.public_cards[player_position].remove_card(card)
            self.update_find_friends(action, player_position)
            if len(moves) == 5: # round finished (5 players), find max player and update points
                winner_index = CardSet.round_winner(moves, self.dominant_suit, self.dominant_rank)
                total_points = sum([c.total_points() for c in moves])
                
                position_array = [lead_position, lead_position.next_position, lead_position.next_position.next_position, lead_position.last_position.last_position, lead_position.last_position]
                round_winner_position = position_array[winner_index]

                if self.friend_card.public:
                    defender_wins_round = round_winner_position in self.defender_team
                    pooled_this_round = self.pooled_defender_points is not None and self.unknown_points is None
                    point_change = total_points
                    if not defender_wins_round:
                        point_change *= -1

                    if pooled_this_round:
                        point_change += self.pooled_defender_points - self.pooled_opponent_points
                        self.pooled_defender_points = None
                        self.pooled_opponent_points = None
                    
                    if defender_wins_round:
                        self.defender_points += total_points
                        logging.debug(f"Defenders escaped {point_change} points")
                    else:
                        self.opponent_points += total_points
                        logging.debug(f"Opponents scored {point_change} points")
                    self.points_per_round.append(point_change)
                else:
                    # Update individual points/dealer's points in defender points
                    self.unknown_points += total_points
                    self.individual_points[round_winner_position] += total_points
                    if player_position == self.dealer_position:
                        self.defender_points += total_points
                    logging.debug(f"Teams not determined. Player {round_winner_position.value} earned {total_points} points")

                # Checks if game is finished
                if self.hands[player_position].size == 0:
                    self.game_ended = True
                    logging.info(f"Game ends! Opponent current points: {self.opponent_points}")
                    logging.debug(f"Points per round: {self.points_per_round}")

                    # Determine if defenders or opponents won this round
                    if self.friend_card.public:
                        opponent_wins_round = round_winner_position in self.opponent_team
                    else:
                        # Determine teams if game ends with no friend found
                        logging.info("Game ended with no friend found. Dealer is the only defender.")
                        self.unknown_team = None
                        self.opponent_team = abs_positions_excluding([self.dealer_position])
                        assert self.defender_team == [self.dealer_position]
                        self.pool_team_points()
                        self.points_per_round[-1] += self.pooled_defender_points - self.pooled_opponent_points
                        self.pooled_defender_points = None
                        self.pooled_opponent_points = None
                        opponent_wins_round = round_winner_position != self.dealer_position

                    if opponent_wins_round:
                        multiplier = MoveType.Combo(moves[winner_index]).get_multiplier(self.dominant_suit, self.dominant_rank)
                        self.opponent_points += self.kitty.total_points() * multiplier
                        self.points_per_round[-1] -= self.kitty.total_points() * multiplier # Opponents earned points from kitty
                        logging.info(f"Opponents received {self.kitty.total_points()} x {multiplier} from kitty. Final points: {self.opponent_points}")
                    else:
                        self.points_per_round[-1] += self.kitty.total_points() * 2 # The points which the defenders prevented from being stolen
                    
                    opponent_reward = 0
                    if self.opponent_points >= 80:
                        opponent_reward = 1 + (self.opponent_points - 80) // 40
                    elif self.opponent_points >= 40:
                        opponent_reward = -1
                    elif self.opponent_points > 0:
                        opponent_reward = -2
                    else:
                        opponent_reward = -3
                    
                    self.final_opponent_reward = opponent_reward
                    self.final_defender_reward = -opponent_reward
                    
                    return None, 0 # Don't return rewards yet. They will be calculated in the end
                else:
                    logging.debug(f"Player {round_winner_position.value} wins round {len(self.round_history)}")
                    self.round_history.append((round_winner_position, [])) # start new round
                return round_winner_position, 0
            else:
                return player_position.next_position, 0
        
        raise AssertionError(f"Unidentified action class {type(action)}: {action}")

    def print_status(self):
        print("Hands:")
        print(f" • One ({self.hands['1'].size}):", self.hands['1'])
        print(f" • Two ({self.hands['2'].size}):", self.hands['2'])
        print(f" • Three ({self.hands['3'].size}):", self.hands['3'])
        print(f" • Four ({self.hands['4'].size}):", self.hands['4'])
        print(f" • Five ({self.hands['5'].size}):", self.hands['5'])
        print("")

        print("Public cards:")
        print(f" • One ({self.public_cards['1'].size}):", self.public_cards['1'])
        print(f" • Two ({self.public_cards['2'].size}):", self.public_cards['2'])
        print(f" • Three ({self.public_cards['3'].size}):", self.public_cards['3'])
        print(f" • Four ({self.public_cards['4'].size}):", self.public_cards['4'])
        print(f" • Five ({self.public_cards['5'].size}):", self.public_cards['5'])
        print("")
        print("Game Status:")
        print("Dealer:", self.dealer_position.value)
        print("Declarers:", self.declarations)
        if self.round_history:
            print(f"Round history ({len(self.round_history)} total):")
            for leader, moves in self.round_history:
                print(f"    {leader} leads: {moves}")
        print("Kitty:", self.kitty)
        if self.kitty_multiplier:
            print(f"Opponents won the last round. Kitty points: {self.kitty.total_points()} x {self.kitty_multiplier}")
        print("Opponents' total points:", self.opponent_points)

    def return_status(self):
        # hands = {}
        # hands['north'] = [self.hands['N'].size, self.hands['N']]
        # hands['west'] = [self.hands['W'].size, self.hands['W']]
        # hands['south'] = [self.hands['S'].size, self.hands['S']]
        # hands['east'] = [self.hands['E'].size, self.hands['E']]

        info = {}
        info['dealer'] = self.dealer_position.value
        info['declarer'] = self.declarations
        info['kitty'] = self.kitty
        info['opponents_total_points'] = self.opponent_points

        rounds = []
        leaders = []
        moves = []
        if self.round_history:
            cnt = 1
            for leader, move in self.round_history:
                rounds.append(cnt)
                leaders.append(leader)
                moves.append(move)
                cnt += 1

        if self.kitty_multiplier:
            print(f"Opponents won the last round. Kitty points: {self.kitty.total_points()} x {self.kitty_multiplier}")

        return info, rounds, leaders, moves