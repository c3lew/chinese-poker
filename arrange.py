#!/usr/bin/env python3
"""
arrange.py

This script loads precomputed evaluation dictionaries for 3-card and 5-card combinations,
accepts a 13-card hand as input (or generates a random one with --random),
and computes all valid arrangements into:
    - Front (3 cards)
    - Middle (5 cards)
    - Back (5 cards)
that satisfy the Chinese Poker ordering rule: eval(front) < eval(middle) <= eval(back).

For each valid arrangement, it retains evaluation scores from the dictionaries.
With the --csv option, results are saved to a CSV file with columns:
"Front", "Front Score", "Middle", "Middle Score", "Back", "Back Score".

Usage:
    # Specific hand:
    python arrange.py --hand "2C 3C AS 10D 9H 8S 7D 6C 5H 4S 3D 2H KH"
        [--three_dict three_card.pkl] [--five_dict five_card.pkl]

    # Random hand with CSV output:
    python arrange.py --random --csv valid.csv
        [--three_dict three_card.pkl] [--five_dict five_card.pkl]
"""

import argparse
import csv
import pickle
import time
from typing import List, Tuple, Dict, Generator, Union
from poker import Card, Deck, card_to_id


def load_dictionaries(
    three_dict_file: str,
    five_dict_file: str
) -> Tuple[Dict[Tuple[int, ...], float], Dict[Tuple[int, ...], float]]:
    """
    Load precomputed evaluation dictionaries from pickle files.

    Args:
        three_dict_file: Path to the 3-card dictionary pickle file.
        five_dict_file: Path to the 5-card dictionary pickle file.

    Returns:
        A tuple containing the 3-card and 5-card evaluation dictionaries.

    Raises:
        FileNotFoundError: If a dictionary file is missing.
        ValueError: If a dictionary file is corrupted or invalid.
    """
    try:
        with open(three_dict_file, "rb") as f:
            three_dict = pickle.load(f)
        with open(five_dict_file, "rb") as f:
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
        hand_str: A string of 13 cards (e.g., "2C 3C AS 10D 9H 8S 7D 6C 5H 4S 3D 2H KH").

    Returns:
        A list of 13 Card objects.

    Raises:
        ValueError: If the hand does not contain exactly 13 cards or has invalid formats.
    """
    try:
        cards = hand_str.split()
        if len(cards) != 13:
            raise ValueError("Hand must contain exactly 13 cards")

        hand = []
        for card_str in cards:
            if len(card_str) > 2:  # Handle '10' rank
                rank, suit = card_str[:-1], card_str[-1]
            else:
                rank, suit = card_str[0], card_str[1]
            hand.append(Card(rank, suit))
        return hand
    except (IndexError, KeyError):
        raise ValueError("Invalid card format in hand")


def get_hand_ids(cards: List[Card]) -> Tuple[int, ...]:
    """
    Convert a list of Card objects to a sorted tuple of card IDs.

    Args:
        cards: List of Card objects.

    Returns:
        A sorted tuple of card IDs.
    """
    return tuple(sorted(card_to_id(card) for card in cards))


def bit_combinations(n: int, r: int) -> Generator[int, None, None]:
    """
    Generate r-sized combinations of indices from 0 to n-1 using bit operations.

    Args:
        n: Total number of items.
        r: Size of combinations to generate.

    Yields:
        Integer bit pattern where each set bit represents a selected index.
    """
    start = (1 << r) - 1
    max_val = 1 << n
    while start < max_val:
        yield start
        c = start & -start
        r = start + c
        start = (((r ^ start) >> 2) // c) | r


def find_valid_arrangements(
    hand: List[Card],
    three_dict: Dict[Tuple[int, ...], float],
    five_dict: Dict[Tuple[int, ...], float]
) -> List[Dict[str, Union[str, float]]]:
    """
    Find all valid Chinese Poker arrangements for a 13-card hand.

    Args:
        hand: List of 13 Card objects.
        three_dict: Dictionary mapping 3-card ID tuples to scores.
        five_dict: Dictionary mapping 5-card ID tuples to scores.

    Returns:
        List of dictionaries with card strings and scores for valid arrangements.
    """
    # Precompute sorted card IDs and cards
    hand_ids = [(card_to_id(card), card) for card in hand]
    hand_ids.sort(key=lambda x: x[0])
    all_ids = [id_ for id_, _ in hand_ids]
    all_cards = [card for _, card in hand_ids]

    # Precompute 3-card scores
    three_scores = {}
    for bits in bit_combinations(13, 3):
        indices = [i for i in range(13) if bits & (1 << i)]
        ids = tuple(all_ids[i] for i in indices)
        try:
            three_scores[bits] = (three_dict[ids], indices)
        except KeyError:
            continue  # Skip invalid combinations

    # Precompute 5-card scores
    five_scores = {}
    for bits in bit_combinations(13, 5):
        indices = [i for i in range(13) if bits & (1 << i)]
        ids = tuple(all_ids[i] for i in indices)
        try:
            five_scores[bits] = (five_dict[ids], indices)
        except KeyError:
            continue  # Skip invalid combinations

    valid_arrangements: List[Dict[str, Union[str, float]]] = []

    # Iterate through all possible front combinations
    for front_bits in three_scores:
        front_score, front_indices = three_scores[front_bits]
        remaining_mask = ((1 << 13) - 1) ^ front_bits
        remaining_count = bin(remaining_mask).count("1")

        # Iterate through middle combinations from remaining cards
        for middle_bits in bit_combinations(remaining_count, 5):
            actual_middle = 0
            src_pos, dst_pos = 0, 0
            while src_pos < 13:
                if remaining_mask & (1 << src_pos):
                    if middle_bits & (1 << dst_pos):
                        actual_middle |= (1 << src_pos)
                    dst_pos += 1
                src_pos += 1

            if actual_middle not in five_scores:
                continue
            middle_score, middle_indices = five_scores[actual_middle]
            if middle_score <= front_score:
                continue

            # Compute back using remaining bits
            back_bits = remaining_mask ^ actual_middle
            if back_bits not in five_scores:
                continue
            back_score, back_indices = five_scores[back_bits]

            # Validate arrangement
            if middle_score <= back_score:
                arrangement = {
                    "Front": " ".join(str(all_cards[i]) for i in front_indices),
                    "Front Score": front_score,
                    "Middle": " ".join(str(all_cards[i]) for i in middle_indices),
                    "Middle Score": middle_score,
                    "Back": " ".join(str(all_cards[i]) for i in back_indices),
                    "Back Score": back_score,
                }
                valid_arrangements.append(arrangement)

    return valid_arrangements


def save_to_csv(arrangements: List[Dict[str, Union[str, float]]], filename: str) -> None:
    """
    Save valid arrangements to a CSV file.

    Args:
        arrangements: List of arrangement dictionaries.
        filename: Path to the output CSV file.

    Raises:
        IOError: If writing to the file fails.
    """
    fieldnames = ["Front", "Front Score", "Middle", "Middle Score", "Back", "Back Score"]
    try:
        with open(filename, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(arrangements)
    except IOError as e:
        raise IOError(f"Error writing to CSV file: {e}")


def main() -> None:
    """
    Main function to process a Chinese Poker hand and find valid arrangements.

    Supports command-line arguments for hand input, random generation, CSV output,
    and benchmarking. Prints results and optionally saves to a file.
    """
    parser = argparse.ArgumentParser(description="Calculate valid Chinese Poker hand arrangements.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--hand",
        help='13-card hand as space-separated string (e.g., "2C 3C AS 10D 9H 8S 7D 6C 5H 4S 3D 2H KH")',
    )
    group.add_argument("--random", action="store_true", help="Generate a random 13-card hand")
    parser.add_argument("--three_dict", default="three_card.pkl", help="Path to 3-card evaluation dictionary")
    parser.add_argument("--five_dict", default="five_card.pkl", help="Path to 5-card evaluation dictionary")
    parser.add_argument("--csv", help="Save results to a CSV file (e.g., 'valid.csv')")
    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmark")
    args = parser.parse_args()

    try:
        print("Loading dictionaries...")
        three_dict, five_dict = load_dictionaries(args.three_dict, args.five_dict)
        print("Dictionaries loaded.")

        if args.benchmark:
            num_runs = 5
            total_time, total_arrangements = 0.0, 0
            print(f"\nRunning benchmark with {num_runs} random hands...")

            for i in range(num_runs):
                deck = Deck()
                deck.shuffle()
                hand = deck.deal(1, cards_each=13)[0]

                start_time = time.perf_counter()
                arrangements = find_valid_arrangements(hand, three_dict, five_dict)
                end_time = time.perf_counter()

                run_time = end_time - start_time
                total_time += run_time
                total_arrangements += len(arrangements)
                print(f"Run {i + 1}: Found {len(arrangements)} arrangements in {run_time:.3f} seconds")

            avg_time = total_time / num_runs
            avg_arrangements = total_arrangements / num_runs
            print(f"\nBenchmark Results:")
            print(f"Average time: {avg_time:.3f} seconds")
            print(f"Average arrangements found: {avg_arrangements:.1f}")
            print(f"Arrangements per second: {avg_arrangements / avg_time:.1f}")
            return

        # Process hand
        if args.random:
            deck = Deck()  # Create a new Deck instance
            deck.shuffle()  # Shuffle the deck
            hand = deck.deal(1, cards_each=13)[0]  # Deal 13 cards
        else:
            hand = parse_hand(args.hand)  # Parse the provided hand

        # Compute arrangements
        start_time = time.perf_counter()
        valid_arrangements = find_valid_arrangements(hand, three_dict, five_dict)
        end_time = time.perf_counter()

        # Display results
        print(f"\nFound {len(valid_arrangements)} valid arrangements in {end_time - start_time:.3f} seconds")
        if not valid_arrangements:
            print("No valid arrangements found!")
            return

        print("\nExample arrangements:")
        for i, arr in enumerate(valid_arrangements[:3], 1):
            print(f"\nArrangement {i}:")
            print(f"Front:  {arr['Front']} (score: {arr['Front Score']:.2f})")
            print(f"Middle: {arr['Middle']} (score: {arr['Middle Score']:.2f})")
            print(f"Back:   {arr['Back']} (score: {arr['Back Score']:.2f})")

        # Save to CSV if requested
        if args.csv:
            save_to_csv(valid_arrangements, args.csv)
            print(f"\nSaved all arrangements to {args.csv}")

    except (FileNotFoundError, ValueError, IOError) as e:
        print(f"Error: {e}")
        return


if __name__ == "__main__":
    main()