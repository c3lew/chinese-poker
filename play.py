#!/usr/bin/env python3
"""
play.py - Play a 4-player Chinese Poker game with optimal arrangements

This script:
1. Deals random hands to 4 players from a standard 52-card deck.
2. Finds the best arrangement for each player based on expected value using
   precomputed evaluation dictionaries and statistical data.
3. Scores the arrangements against each other according to Chinese Poker rules.
4. Displays each player's hand, their optimal arrangement, and the final scores.

Dependencies:
    - poker.py (for Deck and Card classes)
    - arrange_with_stats.py (for arrangement calculation and stats)
    - scoring_fast.py (for Arrangement class and scoring)
    - Precomputed dictionaries (three_card.pkl, five_card.pkl)
"""

from typing import List, Tuple, Dict, Any

from arrange_with_stats import HandStats, find_arrangements_with_stats, load_dictionaries
from poker import Card, Deck
from scoring_fast import Arrangement, score_game


def parse_card_str(card_str: str) -> Tuple[str, str]:
    """Parse a card string into rank and suit, handling '10' as a special case.

    Args:
        card_str: A string representing a card (e.g., 'AS', '10D').

    Returns:
        A tuple of (rank, suit).

    Raises:
        ValueError: If the card string format is invalid.
    """
    if len(card_str) < 2:
        raise ValueError(f"Invalid card format: {card_str}")
    if len(card_str) > 2:  # Handle '10x'
        return card_str[:-1], card_str[-1]
    return card_str[0], card_str[1]


def create_arrangement(arrangement_info: Dict[str, Any]) -> Arrangement:
    """Convert an arrangement info dictionary to an Arrangement object.

    Args:
        arrangement_info: Dictionary with keys 'front_cards', 'middle_cards', 'back_cards',
                          'front_score', 'middle_score', 'back_score'.

    Returns:
        An Arrangement object with card lists and evaluation scores.
    """
    front = [Card(*parse_card_str(c)) for c in arrangement_info["front_cards"].split()]
    middle = [Card(*parse_card_str(c)) for c in arrangement_info["middle_cards"].split()]
    back = [Card(*parse_card_str(c)) for c in arrangement_info["back_cards"].split()]
    return Arrangement(
        front=front,
        middle=middle,
        back=back,
        front_eval=float(arrangement_info["front_score"]),
        middle_eval=float(arrangement_info["middle_score"]),
        back_eval=float(arrangement_info["back_score"]),
    )


def find_best_arrangement(
    hand: List[Card], three_dict: Dict, five_dict: Dict, stats: HandStats
) -> Tuple[Dict[str, Any], Arrangement]:
    """Find the arrangement with the highest expected value for a hand.

    Args:
        hand: List of 13 Card objects.
        three_dict: Dictionary mapping 3-card combinations to scores.
        five_dict: Dictionary mapping 5-card combinations to scores.
        stats: HandStats object for statistical calculations.

    Returns:
        A tuple of (best arrangement info dictionary, corresponding Arrangement object).
    """
    arrangements = find_arrangements_with_stats(hand, three_dict, five_dict, stats)
    best_info = max(arrangements, key=lambda x: x["expected_value"])
    return best_info, create_arrangement(best_info)


def format_hand(hand: List[Card]) -> str:
    """Format a hand of cards for display.

    Args:
        hand: List of Card objects.

    Returns:
        A space-separated string of card representations, sorted by suit and value.
    """
    return " ".join(str(card) for card in sorted(hand, key=lambda c: (c.suit, c.value)))


def format_ev_breakdown(info: Dict[str, Any]) -> str:
    """Format the expected value breakdown for display.

    Args:
        info: Dictionary containing expected value components and win rates.

    Returns:
        A formatted string with detailed EV breakdown.
    """
    basic_ev = info["basic_ev"]
    bonus_ev = info["bonus_ev"]
    win_rates = {
        "front": info["front_win_rate"],
        "middle": info["middle_win_rate"],
        "back": info["back_win_rate"],
    }
    return (
        "Expected Values:\n"
        f"  Front:  Win Rate: {win_rates['front'] * 100:5.1f}%, "
        f"Basic EV: {basic_ev['front']:+.2f}, Bonus EV: {bonus_ev['front']:+.2f}\n"
        f"  Middle: Win Rate: {win_rates['middle'] * 100:5.1f}%, "
        f"Basic EV: {basic_ev['middle']:+.2f}, Bonus EV: {bonus_ev['middle']:+.2f}\n"
        f"  Back:   Win Rate: {win_rates['back'] * 100:5.1f}%, "
        f"Basic EV: {basic_ev['back']:+.2f}, Bonus EV: {bonus_ev['back']:+.2f}\n"
        f"  Sweep Bonus EV: {info['sweep_bonus_ev']:+.2f}\n"
        f"  Overall Bonus EV: {info['overall_bonus_ev']:+.2f}\n"
        f"  Total EV: {info['expected_value']:+.2f}"
    )


def main() -> None:
    """Main function to simulate and display a 4-player Chinese Poker game."""
    print("Loading dictionaries...")
    three_dict, five_dict = load_dictionaries("Dict/three_card.pkl", "Dict/five_card.pkl")
    stats = HandStats()

    print("\nDealing hands...")
    deck = Deck()
    deck.shuffle()
    hands = deck.deal(4)

    print("\nFinding best arrangements...")
    players_arrangements: List[Arrangement] = []
    for i, hand in enumerate(hands, 1):
        print(f"\nPlayer {i}")
        print(f"Hand: {format_hand(hand)}")
        best_info, best_arr = find_best_arrangement(hand, three_dict, five_dict, stats)
        players_arrangements.append(best_arr)
        print("Best arrangement:")
        print(str(best_arr))
        print(format_ev_breakdown(best_info))

    print("\nScoring game...")
    scores = score_game(players_arrangements)

    print("\nFinal Scores:")
    for i, score in enumerate(scores, 1):
        print(f"Player {i}: {score:+.1f}")


if __name__ == "__main__":
    main()