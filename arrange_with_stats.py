#!/usr/bin/env python3
"""
arrange_with_stats.py

Extended version of arrange.py that includes statistical analysis.
For each arrangement, calculates:
- Basic scores (from precomputed evaluation dictionaries)
- Win rates (from position info files)
- Category percentages (from position info files)
- Expected values (using ExpectedValueCalculator)

Usage:
    python arrange_with_stats.py --hand "2C 3C AS 10D 9H 8S 7D 6C 5H 4S 3D 2H KH"
"""

import argparse
import csv
import pickle
import time
from typing import List, Tuple, Dict, Generator, Union
from expected_value import ExpectedValueCalculator
from poker import Card, Deck, card_to_id


class HandStats:
    """
    Handles statistical calculations for hand arrangements in Chinese Poker.

    Loads statistical data from pickle files and provides methods to retrieve
    win rates and category percentages for given scores in front, middle,
    and back positions.

    Attributes:
        front_info (pd.DataFrame): Statistical data for front position (3 cards).
        middle_info (pd.DataFrame): Statistical data for middle position (5 cards).
        back_info (pd.DataFrame): Statistical data for back position (5 cards).
        front_stats (Dict[float, Dict[str, float]]): Lookup for front position stats.
        middle_stats (Dict[float, Dict[str, float]]): Lookup for middle position stats.
        back_stats (Dict[float, Dict[str, float]]): Lookup for back position stats.
        ev_calculator (ExpectedValueCalculator): Instance for calculating expected values.
    """

    def __init__(self) -> None:
        """Initialize HandStats by loading data and setting up lookups."""
        with open('Dict/front_info.pkl', 'rb') as f:
            self.front_info = pickle.load(f)
        with open('Dict/middle_info.pkl', 'rb') as f:
            self.middle_info = pickle.load(f)
        with open('Dict/back_info.pkl', 'rb') as f:
            self.back_info = pickle.load(f)

        self.front_stats = self._create_lookup(self.front_info)
        self.middle_stats = self._create_lookup(self.middle_info)
        self.back_stats = self._create_lookup(self.back_info)

        self.ev_calculator = ExpectedValueCalculator()

    def _create_lookup(self, df: 'pd.DataFrame') -> Dict[float, Dict[str, float]]:
        """
        Create a dictionary for efficient score-based lookups.

        Args:
            df (pd.DataFrame): DataFrame containing score, win_rate, and category_percentage.

        Returns:
            Dict[float, Dict[str, float]]: Mapping of scores to their statistics.
        """
        return {
            row['score']: {
                'win_rate': row['win_rate'],
                'category_percentage': row['category_percentage']
            }
            for _, row in df.iterrows()
        }

    def get_stats(self, score: float, position: str) -> Dict[str, float]:
        """
        Retrieve statistics for a score in a specific position.

        Args:
            score (float): The score to look up.
            position (str): Position ('front', 'middle', or 'back').

        Returns:
            Dict[str, float]: Dictionary with 'win_rate' and 'category_percentage'.
        """
        lookup = {
            'front': self.front_stats,
            'middle': self.middle_stats,
            'back': self.back_stats
        }[position]
        return lookup.get(score, {'win_rate': 0.0, 'category_percentage': 0.0})


def load_dictionaries(
    three_dict_file: str,
    five_dict_file: str
) -> Tuple[Dict[Tuple[int, ...], float], Dict[Tuple[int, ...], float]]:
    """
    Load precomputed evaluation dictionaries from pickle files.

    Args:
        three_dict_file (str): Path to the three-card dictionary pickle file.
        five_dict_file (str): Path to the five-card dictionary pickle file.

    Returns:
        Tuple[Dict[Tuple[int, ...], float], Dict[Tuple[int, ...], float]]: Three-card and five-card evaluation dictionaries.

    Raises:
        FileNotFoundError: If a dictionary file is missing.
        ValueError: If a dictionary file has an invalid format.
    """
    try:
        with open(three_dict_file, 'rb') as f:
            three_dict = pickle.load(f)
        with open(five_dict_file, 'rb') as f:
            five_dict = pickle.load(f)
        return three_dict, five_dict
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Dictionary file not found: {e.filename}")
    except pickle.UnpicklingError:
        raise ValueError("Invalid dictionary file format")


def parse_hand(hand_str: str) -> List[Card]:
    """
    Parse a space-separated hand string into a list of Card objects.

    Args:
        hand_str (str): Space-separated string of 13 cards (e.g., "2C 3C AS 10D 9H 8S 7D 6C 5H 4S 3D 2H KH").

    Returns:
        List[Card]: List of 13 Card objects.

    Raises:
        ValueError: If the hand does not contain exactly 13 cards or contains invalid card formats.
    """
    try:
        cards = hand_str.split()
        if len(cards) != 13:
            raise ValueError("Hand must contain exactly 13 cards")

        hand = []
        for card_str in cards:
            if len(card_str) > 2:  # Handle '10' rank
                rank = card_str[:-1]
                suit = card_str[-1]
            else:
                rank = card_str[0]
                suit = card_str[1]
            hand.append(Card(rank, suit))
        return hand
    except (IndexError, KeyError):
        raise ValueError("Invalid card format in hand")


def get_hand_ids(cards: List[Card]) -> Tuple[int, ...]:
    """
    Convert a list of Card objects to a sorted tuple of card IDs.

    Args:
        cards (List[Card]): List of Card objects.

    Returns:
        Tuple[int, ...]: Sorted tuple of card IDs.
    """
    return tuple(sorted(card_to_id(card) for card in cards))


def bit_combinations(n: int, r: int) -> Generator[int, None, None]:
    """
    Generate r-sized combinations of indices from 0 to n-1 using bit operations.

    Args:
        n (int): Total number of items.
        r (int): Size of combinations to generate.

    Yields:
        int: Bit pattern where each set bit (1) represents a selected index.
    """
    start = (1 << r) - 1
    max_val = 1 << n
    while start < max_val:
        yield start
        c = start & -start
        r = start + c
        start = (((r ^ start) >> 2) // c) | r


def find_arrangements_with_stats(
    hand: List[Card],
    three_dict: Dict[Tuple[int, ...], float],
    five_dict: Dict[Tuple[int, ...], float],
    stats: HandStats
) -> List[Dict[str, Union[str, float]]]:
    """
    Find all valid arrangements of a 13-card hand with statistical information.

    Args:
        hand (List[Card]): List of 13 Card objects.
        three_dict (Dict[Tuple[int, ...], float]): Dictionary mapping 3-card tuples to scores.
        five_dict (Dict[Tuple[int, ...], float]): Dictionary mapping 5-card tuples to scores.
        stats (HandStats): Instance providing statistical data for positions.

    Returns:
        List[Dict[str, Union[str, float]]]: List of dictionaries containing arrangement details,
        including card strings, scores, win rates, category percentages, and expected values.
    """
    arrangements: List[Dict[str, Union[str, float]]] = []

    # Generate all possible 3-card selections for the front
    for front_bits in bit_combinations(13, 3):
        front_cards = [hand[i] for i in range(13) if front_bits & (1 << i)]
        middle_back_cards = [hand[i] for i in range(13) if not (front_bits & (1 << i))]

        front_tuple = get_hand_ids(front_cards)
        if front_tuple not in three_dict:
            continue
        front_score = three_dict[front_tuple]
        front_stats = stats.get_stats(front_score, 'front')
        front_win_rate = front_stats['win_rate']
        front_category = front_stats['category_percentage']

        # Generate all possible 5-card selections for the middle from the remaining 10 cards
        for middle_bits in bit_combinations(10, 5):
            middle_cards = [middle_back_cards[i] for i in range(10) if middle_bits & (1 << i)]
            back_cards = [middle_back_cards[i] for i in range(10) if not (middle_bits & (1 << i))]

            middle_tuple = get_hand_ids(middle_cards)
            back_tuple = get_hand_ids(back_cards)
            if middle_tuple not in five_dict or back_tuple not in five_dict:
                continue

            middle_score = five_dict[middle_tuple]
            back_score = five_dict[back_tuple]
            middle_stats = stats.get_stats(middle_score, 'middle')
            middle_win_rate = middle_stats['win_rate']
            middle_category = middle_stats['category_percentage']
            back_stats = stats.get_stats(back_score, 'back')
            back_win_rate = back_stats['win_rate']
            back_category = back_stats['category_percentage']

            # Validate arrangement: back > middle > front
            if back_score > middle_score > front_score:
                arrangement = {
                    'front_score': front_score,
                    'front_win_rate': front_win_rate,
                    'middle_score': middle_score,
                    'middle_win_rate': middle_win_rate,
                    'back_score': back_score,
                    'back_win_rate': back_win_rate
                }
                ev = stats.ev_calculator.calculate_total_ev(arrangement)

                arrangement_info = {
                    'front_cards': ' '.join(str(card) for card in front_cards),
                    'middle_cards': ' '.join(str(card) for card in middle_cards),
                    'back_cards': ' '.join(str(card) for card in back_cards),
                    'front_score': front_score,
                    'middle_score': middle_score,
                    'back_score': back_score,
                    'front_win_rate': front_win_rate,
                    'middle_win_rate': middle_win_rate,
                    'back_win_rate': back_win_rate,
                    'front_category': front_category,
                    'middle_category': middle_category,
                    'back_category': back_category,
                    'expected_value': ev['total_ev'],
                    'basic_ev': ev['basic_ev'],
                    'bonus_ev': ev['bonus_ev'],
                    'sweep_bonus_ev': ev['sweep_bonus_ev'],
                    'overall_bonus_ev': ev['overall_bonus_ev']
                }
                arrangements.append(arrangement_info)

    return arrangements


def save_to_csv(arrangements: List[Dict[str, Union[str, float]]], filename: str) -> None:
    """
    Save arrangements to a CSV file, sorted by expected value in descending order.

    Args:
        arrangements (List[Dict[str, Union[str, float]]]): List of arrangement dictionaries.
        filename (str): Path to the output CSV file.
    """
    sorted_arrangements = sorted(arrangements, key=lambda x: x['expected_value'], reverse=True)
    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'front_cards', 'front_score', 'front_win_rate', 'front_category',
            'middle_cards', 'middle_score', 'middle_win_rate', 'middle_category',
            'back_cards', 'back_score', 'back_win_rate', 'back_category',
            'expected_value', 'basic_ev', 'bonus_ev', 'sweep_bonus_ev', 'overall_bonus_ev'
        ])
        writer.writeheader()
        writer.writerows(sorted_arrangements)


def main() -> None:
    """
    Main entry point for calculating Chinese Poker hand arrangements with statistics.

    Parses command-line arguments, processes a hand, finds valid arrangements,
    and optionally saves results to a CSV file.
    """
    parser = argparse.ArgumentParser(
        description="Calculate Chinese Poker hand arrangements with statistical analysis."
    )
    parser.add_argument(
        '--hand',
        help="Space-separated string of 13 cards (e.g., '2C 3C AS 10D 9H 8S 7D 6C 5H 4S 3D 2H KH')"
    )
    parser.add_argument(
        '--random',
        action='store_true',
        help="Generate a random 13-card hand"
    )
    parser.add_argument(
        '--csv',
        help="Output CSV file name (e.g., 'arrangements.csv')"
    )
    parser.add_argument(
        '--three_dict',
        default='three_card.pkl',
        help="Path to the three-card dictionary pickle file"
    )
    parser.add_argument(
        '--five_dict',
        default='five_card.pkl',
        help="Path to the five-card dictionary pickle file"
    )
    args = parser.parse_args()

    three_dict, five_dict = load_dictionaries(args.three_dict, args.five_dict)
    stats = HandStats()

    if args.random:
        deck = Deck()
        deck.shuffle()
        hand = deck.deal(1, cards_each=13)[0]
    elif args.hand:
        hand = parse_hand(args.hand)
    else:
        parser.error("Must specify either --hand or --random")

    print("Finding valid arrangements...")
    start_time = time.time()
    arrangements = find_arrangements_with_stats(hand, three_dict, five_dict, stats)
    end_time = time.time()

    print(f"\nFound {len(arrangements)} valid arrangements in {end_time - start_time:.2f} seconds")
    if args.csv:
        save_to_csv(arrangements, args.csv)
        print(f"Results saved to {args.csv}")


if __name__ == "__main__":
    main()