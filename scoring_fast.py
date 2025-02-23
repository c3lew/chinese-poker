#!/usr/bin/env python3
"""
scoring_fast.py - Fast scoring module for Chinese Poker

Optimizations:
1. Uses __slots__ in Arrangement class for memory efficiency.
2. Optimized data structures for lookups.
3. Minimizes object creation and copying.
4. Provides fast pairwise comparison and bonus scoring.

This module:
- Loads precomputed evaluation dictionaries.
- Finds valid arrangements using arrange.py.
- Compares arrangements and calculates scores with bonuses.
- Supports a 4-player game simulation with performance benchmarking.
"""

import pickle
import random
import timeit
from dataclasses import dataclass
from typing import List, Tuple, Dict

from arrange import find_valid_arrangements as calc_find_arrangements
from poker import Card, Deck


@dataclass
class Arrangement:
    """Represents a valid arrangement of 13 cards into front, middle, and back hands.

    Attributes:
        front: List of 3 Card objects (front hand).
        middle: List of 5 Card objects (middle hand).
        back: List of 5 Card objects (back hand).
        front_eval: Evaluation score for the front hand.
        middle_eval: Evaluation score for the middle hand.
        back_eval: Evaluation score for the back hand.
    """

    __slots__ = ["front", "middle", "back", "front_eval", "middle_eval", "back_eval"]
    front: List[Card]
    middle: List[Card]
    back: List[Card]
    front_eval: float
    middle_eval: float
    back_eval: float

    def __str__(self) -> str:
        """Return a string representation of the arrangement."""
        return (
            f"Front  ({self.front_eval:.6f}): {' '.join(str(c) for c in self.front)}\n"
            f"Middle ({self.middle_eval:.6f}): {' '.join(str(c) for c in self.middle)}\n"
            f"Back   ({self.back_eval:.6f}): {' '.join(str(c) for c in self.back)}"
        )


def load_evaluation_dicts(three_card_file: str, five_card_file: str) -> Tuple[Dict, Dict]:
    """Load precomputed evaluation dictionaries from pickle files.

    Args:
        three_card_file: Path to the 3-card dictionary pickle file.
        five_card_file: Path to the 5-card dictionary pickle file.

    Returns:
        A tuple of (three_card_dict, five_card_dict) containing evaluation dictionaries.
    """
    with open(three_card_file, "rb") as f:
        three_card_dict = pickle.load(f)
    with open(five_card_file, "rb") as f:
        five_card_dict = pickle.load(f)
    return three_card_dict, five_card_dict


def parse_card_str(card_str: str) -> Card:
    """Parse a card string into a Card object, handling '10' as a special case.

    Args:
        card_str: A string representing a card (e.g., 'AS', '10D').

    Returns:
        A Card object.

    Raises:
        ValueError: If the card string format is invalid.
    """
    if len(card_str) < 2:
        raise ValueError(f"Invalid card format: {card_str}")
    if len(card_str) > 2:  # Handle '10x'
        rank, suit = card_str[:-1], card_str[-1]
    else:
        rank, suit = card_str[0], card_str[1]
    return Card(rank, suit)


def find_valid_arrangements(
        hand: List[Card], three_card_dict: Dict, five_card_dict: Dict
) -> List[Arrangement]:
    """Find all valid arrangements of a 13-card hand using arrange.py's implementation.

    Args:
        hand: List of 13 Card objects.
        three_card_dict: Dictionary mapping 3-card combinations to scores.
        five_card_dict: Dictionary mapping 5-card combinations to scores.

    Returns:
        A list of Arrangement objects representing valid hand splits.
    """
    arrangements = calc_find_arrangements(hand, three_card_dict, five_card_dict)
    valid_arrangements = [
        Arrangement(
            front=[parse_card_str(c) for c in arr["Front"].split()],
            middle=[parse_card_str(c) for c in arr["Middle"].split()],
            back=[parse_card_str(c) for c in arr["Back"].split()],
            front_eval=arr["Front Score"],
            middle_eval=arr["Middle Score"],
            back_eval=arr["Back Score"],
        )
        for arr in arrangements
    ]
    return valid_arrangements


def compare_arrangements_fast(arr1: Arrangement, arr2: Arrangement) -> Tuple[int, List[int]]:
    """Compare two arrangements efficiently and compute the score with bonuses.

    Args:
        arr1: First player's arrangement.
        arr2: Second player's arrangement.

    Returns:
        A tuple of (total_score, position_scores) where total_score is the net score for arr1,
        and position_scores is a list of [front_score, middle_score, back_score].
    """
    scores = [0, 0, 0]
    total_score = 0

    # Front comparison
    if arr1.front_eval > arr2.front_eval:
        scores[0] = 1 + (3 if int(arr1.front_eval) == 4 else 0)  # Three of a kind bonus
    elif arr1.front_eval < arr2.front_eval:
        scores[0] = -1 - (3 if int(arr2.front_eval) == 4 else 0)
    total_score += scores[0]

    # Middle comparison
    category1, category2 = int(arr1.middle_eval), int(arr2.middle_eval)
    if arr1.middle_eval > arr2.middle_eval:
        scores[1] = 1
        if category1 in (9, 10):  # Straight flush or royal flush
            scores[1] += 5
        elif category1 == 8:  # Four of a kind
            scores[1] += 4
        elif category1 == 7:  # Full house
            scores[1] += 2
    elif arr1.middle_eval < arr2.middle_eval:
        scores[1] = -1
        if category2 in (9, 10):
            scores[1] -= 5
        elif category2 == 8:
            scores[1] -= 4
        elif category2 == 7:
            scores[1] -= 2
    total_score += scores[1]

    # Back comparison
    category1, category2 = int(arr1.back_eval), int(arr2.back_eval)
    if arr1.back_eval > arr2.back_eval:
        scores[2] = 1
        if category1 in (9, 10):  # Straight flush or royal flush
            scores[2] += 5
        elif category1 == 8:  # Four of a kind
            scores[2] += 4
    elif arr1.back_eval < arr2.back_eval:
        scores[2] = -1
        if category2 in (9, 10):
            scores[2] -= 5
        elif category2 == 8:
            scores[2] -= 4
    total_score += scores[2]

    # Sweep bonus
    if all(s > 0 for s in scores):
        total_score += 3
    elif all(s < 0 for s in scores):
        total_score -= 3

    return total_score, scores


def calculate_overall_bonus_fast(arrangements: List[Arrangement]) -> List[int]:
    """Calculate the overall bonus for players with the best hands in all positions.

    Args:
        arrangements: List of Arrangement objects for all players.

    Returns:
        A list of bonus scores (+18 for the winner, -6 for others, or 0 if no winner).
    """
    n_players = len(arrangements)
    bonus_scores = [0] * n_players

    for i, arr_i in enumerate(arrangements):
        has_best = all(
            arr_i.front_eval > arr_j.front_eval
            and arr_i.middle_eval > arr_j.middle_eval
            and arr_i.back_eval > arr_j.back_eval
            for j, arr_j in enumerate(arrangements)
            if i != j
        )
        if has_best:
            bonus_scores[i] = 18
            for j in range(n_players):
                if j != i:
                    bonus_scores[j] = -6
            break

    return bonus_scores


def score_game(players_arrangements: List[Arrangement]) -> List[int]:
    """Score a single game given the arrangements for all players.

    Args:
        players_arrangements: List of Arrangement objects for each player.

    Returns:
        A list of final scores for each player.
    """
    n_players = len(players_arrangements)
    scores = [0] * n_players

    # Compute pairwise scores
    for i in range(n_players):
        for j in range(i + 1, n_players):
            score, _ = compare_arrangements_fast(players_arrangements[i], players_arrangements[j])
            scores[i] += score
            scores[j] -= score

    # Apply overall bonus
    bonus_scores = calculate_overall_bonus_fast(players_arrangements)
    return [s + b for s, b in zip(scores, bonus_scores)]


def main() -> None:
    """Main function to demonstrate scoring and benchmark performance."""
    print("Loading evaluation dictionaries...")
    three_card_dict, five_card_dict = load_evaluation_dicts("Dict/three_card.pkl", "Dict/five_card.pkl")

    print("\nGenerating test game...")
    deck = Deck()
    deck.shuffle()
    hands = deck.deal(4)

    players_arrangements = []
    for i, hand in enumerate(hands):
        print(f"\nPlayer {i + 1}'s hand ({len(hand)} cards):")
        print(" ".join(str(card) for card in sorted(hand, key=lambda c: (c.value, c.suit), reverse=True)))

        arrangements = find_valid_arrangements(hand, three_card_dict, five_card_dict)
        if not arrangements:
            print(f"No valid arrangements found for Player {i + 1}!")
            return

        chosen_arrangement = random.choice(arrangements)
        players_arrangements.append(chosen_arrangement)
        print("\nChosen arrangement:")
        print(chosen_arrangement)

    print("\nBenchmarking scoring speed...")

    def score_test() -> List[int]:
        return score_game(players_arrangements)

    num_rounds = 10_000
    total_time = timeit.timeit(score_test, number=num_rounds)
    avg_time = total_time / num_rounds

    print(f"\nScoring Speed:")
    print(f"Total time for {num_rounds:,} rounds: {total_time:.4f} seconds")
    print(f"Average time per game: {avg_time * 1000:.4f} milliseconds")
    print(f"Games per second: {1 / avg_time:,.1f}")

    print("\nTest Game Final Scores:")
    final_scores = score_test()
    for i, score in enumerate(final_scores):
        print(f"Player {i + 1}: {score:+d}")


if __name__ == "__main__":
    main()
