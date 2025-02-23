#!/usr/bin/env python3
"""
game_equilibrium.py

This module computes an approximate Nash equilibrium for a four-player Chinese Poker game.
Given four 13-card hands, it generates all valid arrangements (splits into front (3 cards),
middle (5 cards), and back (5 cards)) for each player using precomputed evaluation dictionaries.
It then runs a best-response dynamics procedure to find a strategy profile where no player
can unilaterally improve their payoff by switching to another arrangement.

If a cycle is detected in the dynamics, it selects a "best" profile from the cycle based on
a weighted combination of average payoff, minimum payoff, and payoff variance.

The output includes each player's hand, their chosen arrangement, and their final payoff.

Usage:
    python game_equilibrium.py --random --max_iter 100
    (or provide four hands via input)

Dependencies:
    - poker.py (for Card and Deck classes)
    - scoring_fast.py (for arrangement evaluation and scoring functions)
    - Precomputed evaluation dictionaries (e.g., three_card.pkl and five_card.pkl)
"""

import argparse
import random
import time
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
from poker import Card, Deck
from scoring_fast import (
    Arrangement,
    load_evaluation_dicts,
    find_valid_arrangements,
    compare_arrangements_fast,
    calculate_overall_bonus_fast,
)


@dataclass
class PlayerState:
    """Represents a player's current state in the game."""

    hand: List[Card]  # Original 13-card hand
    arrangements: List[Arrangement]  # All valid arrangements
    current_strategy: int  # Index of current chosen arrangement
    payoffs: List[float]  # Historical payoffs

    def __str__(self) -> str:
        """String representation of the player's state."""
        arr = self.arrangements[self.current_strategy]
        return (
            f"Hand: {' '.join(str(c) for c in sorted(self.hand, key=lambda c: (c.value, c.suit), reverse=True))}\n"
            f"Current arrangement:\n{arr}\n"
            f"Current payoff: {self.payoffs[-1] if self.payoffs else 0}"
        )


class ChinesePokerGame:
    """Manages the Chinese Poker game state and Nash equilibrium search."""

    def __init__(
        self,
        hands: List[List[Card]],
        three_dict: Dict[Tuple[int, ...], float],
        five_dict: Dict[Tuple[int, ...], float],
    ) -> None:
        """
        Initialize the game with four players' hands and evaluation dictionaries.

        Args:
            hands: A list of four lists, each containing 13 Card objects.
            three_dict: Evaluation dictionary for 3-card hands.
            five_dict: Evaluation dictionary for 5-card hands.

        Raises:
            ValueError: If there are not exactly four players or if a player has no valid arrangements.
        """
        if len(hands) != 4:
            raise ValueError("Must have exactly 4 players")

        self.players: List[PlayerState] = []
        self.payoff_cache: Dict[Tuple[int, int, int, int], List[float]] = {}
        self.history: List[Tuple[Tuple[int, int, int, int], List[float]]] = []
        self.cycle_stats: Dict[str, Any] = {}

        for hand in hands:
            arrangements = find_valid_arrangements(hand, three_dict, five_dict)
            if not arrangements:
                raise ValueError(f"No valid arrangements found for hand: {' '.join(str(c) for c in hand)}")

            self.players.append(
                PlayerState(
                    hand=hand,
                    arrangements=arrangements,
                    current_strategy=random.randrange(len(arrangements)),
                    payoffs=[],
                )
            )

    def compute_payoffs(self) -> List[float]:
        """Compute payoffs for the current arrangement choices.

        Returns:
            A list of four floats representing each player's payoff.
        """
        strategy_key = tuple(p.current_strategy for p in self.players)
        if strategy_key in self.payoff_cache:
            return self.payoff_cache[strategy_key]

        arrangements = [p.arrangements[p.current_strategy] for p in self.players]
        scores = [0.0] * 4

        for i in range(4):
            for j in range(i + 1, 4):
                score, _ = compare_arrangements_fast(arrangements[i], arrangements[j])
                scores[i] += score
                scores[j] -= score

        bonus_scores = calculate_overall_bonus_fast(arrangements)
        final_scores = [s + b for s, b in zip(scores, bonus_scores)]

        self.payoff_cache[strategy_key] = final_scores
        return final_scores

    def find_best_response(self, player_idx: int) -> Tuple[int, float]:
        """Find the best response arrangement for a player given others' strategies.

        Args:
            player_idx: Index of the player (0 to 3).

        Returns:
            A tuple containing the best strategy index and its corresponding payoff.
        """
        best_score = float("-inf")
        best_strategy = self.players[player_idx].current_strategy

        for i in range(len(self.players[player_idx].arrangements)):
            old_strategy = self.players[player_idx].current_strategy
            self.players[player_idx].current_strategy = i
            payoffs = self.compute_payoffs()
            score = payoffs[player_idx]

            if score > best_score:
                best_score = score
                best_strategy = i

            self.players[player_idx].current_strategy = old_strategy

        return best_strategy, best_score

    def is_approximate_cycle(self, length: int, tolerance: float = 1e-6) -> bool:
        """Check if the last 'length' states form an approximate cycle.

        Args:
            length: The cycle length to check.
            tolerance: The tolerance for comparing payoffs.

        Returns:
            True if a cycle is detected, False otherwise.
        """
        if len(self.history) < length * 2:
            return False

        for i in range(length):
            if self.history[-(i + 1)][0] != self.history[-(i + 1 + length)][0]:
                return False

            curr_payoffs = self.history[-(i + 1)][1]
            prev_payoffs = self.history[-(i + 1 + length)][1]
            if any(abs(c - p) > tolerance for c, p in zip(curr_payoffs, prev_payoffs)):
                return False

        return True

    def is_oscillating(self, length: int, tolerance: float = 2.0) -> bool:
        """Check if payoffs are oscillating with a given period.

        Args:
            length: The period length to check.
            tolerance: The maximum allowed difference in payoff sequences.

        Returns:
            True if oscillation is detected, False otherwise.
        """
        if len(self.history) < length * 3:
            return False

        sequences = [[p[1][i] for p in self.history[-length * 3 :]] for i in range(4)]

        for seq in sequences:
            for offset in range(length):
                values = seq[offset::length]
                if max(values) - min(values) > tolerance:
                    return False

        return True

    def analyze_cycle(self, cycle_start: int, cycle_length: int) -> Dict[str, Any]:
        """Analyze properties of a detected cycle.

        Args:
            cycle_start: The starting index of the cycle in history.
            cycle_length: The length of the cycle.

        Returns:
            A dictionary containing cycle statistics.
        """
        cycle = self.history[cycle_start : cycle_start + cycle_length]
        stats = {
            "length": cycle_length,
            "profiles": len(set(p[0] for p in cycle)),
            "avg_payoffs": [np.mean([p[1][i] for p in cycle]) for i in range(4)],
            "min_payoffs": [min(p[1][i] for p in cycle) for i in range(4)],
            "max_payoffs": [max(p[1][i] for p in cycle) for i in range(4)],
            "var_payoffs": [np.var([p[1][i] for p in cycle]) for i in range(4)],
            "total_payoffs": [sum(p[1][i] for p in cycle) for i in range(4)],
        }
        return stats

    def get_best_profile_in_cycle(
        self, cycle_start: int, cycle_length: int
    ) -> Tuple[Tuple[int, int, int, int], List[float]]:
        """Select the best strategy profile in a cycle using weighted criteria.

        Args:
            cycle_start: The starting index of the cycle in history.
            cycle_length: The length of the cycle.

        Returns:
            A tuple containing the best strategy profile and its payoffs.
        """
        cycle = self.history[cycle_start : cycle_start + cycle_length]
        scores = []

        for profile, payoffs in cycle:
            avg_payoff = np.mean(payoffs)
            min_payoff = min(payoffs)
            var_payoff = np.var(payoffs)
            score = 0.4 * avg_payoff + 0.4 * min_payoff - 0.2 * var_payoff
            scores.append(score)

        best_idx = np.argmax(scores)
        return cycle[best_idx]

    def detect_cycle(self) -> Optional[Tuple[int, int]]:
        """Detect if the best-response dynamics have entered a cycle.

        Returns:
            A tuple (cycle_start, cycle_length) if a cycle is found, None otherwise.
        """
        n = len(self.history)
        if n < 2:
            return None

        for length in range(2, n // 2 + 1):
            if n < length * 2:
                break

            if self.is_approximate_cycle(length) or self.is_oscillating(length):
                return (n - length, length)

        return None

    def find_nash_equilibrium(self, max_iter: int = 100, verbose: bool = True) -> bool:
        """
        Find an approximate Nash equilibrium using best-response dynamics.

        Iteratively updates each player's strategy to their best response given others' strategies.
        Detects convergence or cycles in strategy profiles or payoff oscillations.

        Args:
            max_iter: Maximum number of iterations to perform.
            verbose: If True, print progress information.

        Returns:
            True if a Nash equilibrium was found, False if a cycle was detected or max_iter was reached.
        """
        start_time = time.perf_counter()
        iteration = 0
        converged = False

        while iteration < max_iter and not converged:
            current_strategies = tuple(p.current_strategy for p in self.players)
            current_payoffs = self.compute_payoffs()
            self.history.append((current_strategies, current_payoffs))

            cycle = self.detect_cycle()
            if cycle:
                cycle_start, cycle_length = cycle
                self.cycle_stats = self.analyze_cycle(cycle_start, cycle_length)
                best_profile, best_payoffs = self.get_best_profile_in_cycle(cycle_start, cycle_length)

                if verbose:
                    print(f"\nCycle detected of length {cycle_length}!")
                    print("\nCycle Analysis:")
                    print(f"Unique strategy profiles: {self.cycle_stats['profiles']}")
                    print("\nPayoff Statistics:")
                    for i in range(4):
                        print(f"Player {i + 1}:")
                        print(f"  Average: {self.cycle_stats['avg_payoffs'][i]:+.1f}")
                        print(
                            f"  Range: [{self.cycle_stats['min_payoffs'][i]:+.0f}, {self.cycle_stats['max_payoffs'][i]:+.0f}]"
                        )
                        print(f"  Variance: {self.cycle_stats['var_payoffs'][i]:.2f}")

                    print("\nCycle payoffs:")
                    for profile, payoffs in self.history[cycle_start : cycle_start + cycle_length]:
                        print("  " + " ".join(f"{p:+.0f}" for p in payoffs))
                    print(f"\nChosen profile with payoffs: {' '.join(f'{p:+.0f}' for p in best_payoffs)}")
                    print("(Selected based on weighted average of mean payoff, minimum payoff, and payoff stability)")

                for i, strategy in enumerate(best_profile):
                    self.players[i].current_strategy = strategy
                    self.players[i].payoffs.append(best_payoffs[i])

                return False

            old_payoffs = current_payoffs

            for i in range(4):
                best_strategy, _ = self.find_best_response(i)
                self.players[i].current_strategy = best_strategy

            new_payoffs = self.compute_payoffs()

            for i, payoff in enumerate(new_payoffs):
                self.players[i].payoffs.append(payoff)

            improved = any(new > old + 1e-6 for new, old in zip(new_payoffs, old_payoffs))

            if verbose and (iteration % 10 == 0 or not improved):
                elapsed = time.perf_counter() - start_time
                print(f"\nIteration {iteration} (elapsed: {elapsed:.2f}s)")
                print("Current payoffs:", " ".join(f"{p:+.0f}" for p in new_payoffs))

            if not improved:
                converged = True
                if verbose:
                    print("\nConverged to equilibrium!")

            iteration += 1

        if verbose and not converged:
            print(f"\nReached maximum iterations ({max_iter})")

        return converged


def parse_hand(hand_str: str) -> List[Card]:
    """
    Parse a space-separated string of cards into a list of Card objects.

    Args:
        hand_str: A string of 13 card representations (e.g., 'AS KH QD ...').

    Returns:
        A list of 13 Card objects.

    Raises:
        ValueError: If the hand does not contain exactly 13 cards or has invalid card formats.
    """
    try:
        cards = []
        for card_str in hand_str.split():
            if len(card_str) > 2:  # Handle '10'
                rank, suit = card_str[:-1], card_str[-1]
            else:
                rank, suit = card_str[0], card_str[1]
            cards.append(Card(rank, suit))

        if len(cards) != 13:
            raise ValueError("Each hand must contain exactly 13 cards")
        return cards
    except (IndexError, KeyError):
        raise ValueError(f"Invalid card format in hand: {hand_str}")


def main() -> None:
    """
    Main function to find an approximate Nash equilibrium in Chinese Poker.

    Handles argument parsing, hand generation or input, game setup, equilibrium search,
    and result display including timing information.
    """
    parser = argparse.ArgumentParser(description="Find Nash equilibrium for Chinese Poker")
    parser.add_argument("--random", action="store_true", help="Use random hands")
    parser.add_argument("--max_iter", type=int, default=100, help="Maximum iterations")
    parser.add_argument("--three_dict", default="three_card.pkl", help="Three-card dictionary file")
    parser.add_argument("--five_dict", default="five_card.pkl", help="Five-card dictionary file")
    args = parser.parse_args()

    total_start = time.perf_counter()

    print("Loading evaluation dictionaries...")
    dict_start = time.perf_counter()
    three_dict, five_dict = load_evaluation_dicts(args.three_dict, args.five_dict)
    dict_time = time.perf_counter() - dict_start

    hands_start = time.perf_counter()
    if args.random:
        print("\nDealing random hands...")
        deck = Deck()
        deck.shuffle()
        hands = deck.deal(4)
    else:
        print("\nEnter four 13-card hands (space-separated, e.g. 'AS KH QD...'):")
        hands = []
        for i in range(4):
            while True:
                try:
                    hand_str = input(f"Hand {i + 1}: ")
                    hand = parse_hand(hand_str)
                    hands.append(hand)
                    break
                except ValueError as e:
                    print(f"Error: {e}")
    hands_time = time.perf_counter() - hands_start

    print("\nFinding valid arrangements and computing equilibrium...")
    game_start = time.perf_counter()
    game = ChinesePokerGame(hands, three_dict, five_dict)
    init_time = time.perf_counter() - game_start

    equilibrium_start = time.perf_counter()
    is_nash = game.find_nash_equilibrium(max_iter=args.max_iter)
    equilibrium_time = time.perf_counter() - equilibrium_start

    print("\nFinal Game State:")
    print("=" * 60)
    for i, player in enumerate(game.players):
        print(f"\nPlayer {i + 1}:")
        print(player)

    print("\nFinal Payoffs:", " ".join(f"{p.payoffs[-1]:+.0f}" for p in game.players))

    total_time = time.perf_counter() - total_start
    print("\nTiming Information:")
    print(f"{'Dictionary loading:':<25} {dict_time:>8.3f}s")
    print(f"{'Hand generation:':<25} {hands_time:>8.3f}s")
    print(f"{'Finding arrangements:':<25} {init_time:>8.3f}s")
    print(f"{'Equilibrium search:':<25} {equilibrium_time:>8.3f}s")
    print(f"{'Total time:':<25} {total_time:>8.3f}s")

    if not is_nash:
        print("\nNote: No true Nash equilibrium found (detected cycle or hit iteration limit)")


if __name__ == "__main__":
    main()