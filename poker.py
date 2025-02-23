#!/usr/bin/env python3
"""
poker.py – A module for Chinese Poker

Features:
  • A Deck class that creates a standard 52–card deck, shuffles it,
    and deals 13 cards per player (for 1–4 players).
  • Hand evaluation for 5–card hands using standard poker ranking:
       Royal Flush > Straight Flush > Four of a Kind > Full House >
       Flush > Straight > Three of a Kind > Two Pair > Pair > High Card
  • Hand evaluation for 3–card hands (only three-of-a-kind, pair, high-card)
  • A system to generate all possible 3–card or 5–card combinations from
    a 52–card deck, evaluate (score) them, and save the mapping to file.
  • Optionally save the generated combinations to a CSV file for verification.

Usage:
  To deal hands:
      python poker.py --deal 4

  To generate combination dictionaries:
      3–card hands:
          python poker.py --generate 3 --outfile three_card.pkl [--csv three_card.csv]
      5–card hands:
          python poker.py --generate 5 --outfile five_card.pkl [--csv five_card.csv]
"""

import random
import itertools
import collections
import pickle
import argparse
import csv
from typing import List, Tuple, Dict, Optional

# Global definitions for card ranks and suits
RANKS = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]
RANK_VALUES = {rank: i for i, rank in enumerate(RANKS, start=2)}
SUITS = ["C", "D", "H", "S"]  # Clubs, Diamonds, Hearts, Spades


class Card:
    """A playing card with a rank, suit, and numerical value."""

    def __init__(self, rank: str, suit: str) -> None:
        if rank not in RANKS:
            raise ValueError(f"Invalid rank: {rank}")
        if suit not in SUITS:
            raise ValueError(f"Invalid suit: {suit}")
        self.rank = rank
        self.suit = suit
        self.value = RANK_VALUES[rank]

    def __repr__(self) -> str:
        return f"{self.rank}{self.suit}"

    def __hash__(self) -> int:
        return hash((self.rank, self.suit))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Card):
            return NotImplemented
        return (self.rank, self.suit) == (other.rank, other.suit)


class Deck:
    """A deck of 52 playing cards."""

    def __init__(self) -> None:
        self.cards: List[Card] = [Card(rank, suit) for suit in SUITS for rank in RANKS]

    def shuffle(self) -> None:
        random.shuffle(self.cards)

    def deal(self, num_players: int, cards_each: int = 13) -> List[List[Card]]:
        """
        Deal cards_each cards to each player (supports 1–4 players).
        Returns a list of lists, where each sublist is a player's hand.
        """
        if num_players * cards_each > len(self.cards):
            raise ValueError("Not enough cards in the deck to deal")
        hands: List[List[Card]] = []
        for i in range(num_players):
            hand = self.cards[i * cards_each : (i + 1) * cards_each]
            hands.append(hand)
        return hands


# -------------------- Hand Evaluation --------------------


def is_straight(values: List[int]) -> Tuple[bool, Optional[int]]:
    """
    Check if the list of card values (assumed sorted ascending) forms a straight.
    Returns (True, high_card) if yes. Special-case A,2,3,4,5: high_card is 5.
    """
    consecutive = all(values[i + 1] - values[i] == 1 for i in range(len(values) - 1))
    if consecutive:
        return True, values[-1]
    if values == [2, 3, 4, 5, 14]:
        return True, 5
    return False, None


def tuple_to_decimal(category: int, tiebreakers: Tuple[int, ...]) -> float:
    """Convert a category and tiebreaker tuple to a decimal score."""
    if not tiebreakers:
        return float(category)
    # Convert tiebreakers to string, pad each number to 2 digits
    decimal_part = "".join(str(x).zfill(2) for x in tiebreakers)
    return float(f"{category}.{decimal_part}")


def evaluate_five(cards: List[Card]) -> float:
    """
    Evaluate a 5–card hand.
    Returns a decimal number where:
    - Integer part is the category rank (1-10)
    - Decimal part contains tiebreaker values (e.g., high cards or kickers).
    Categories:
      10: Royal Flush
       9: Straight Flush
       8: Four of a Kind
       7: Full House
       6: Flush
       5: Straight
       4: Three of a Kind
       3: Two Pair
       2: Pair
       1: High Card
    """
    values = sorted([card.value for card in cards])
    counts = collections.Counter(card.value for card in cards)
    count_vals = sorted(counts.values(), reverse=True)
    is_flush = len({card.suit for card in cards}) == 1
    straight, high_straight = is_straight(sorted(values))

    if is_flush and straight:
        if high_straight == 14 and min(values) == 10:
            return tuple_to_decimal(10, ())
        else:
            return tuple_to_decimal(9, (high_straight,))
    if 4 in count_vals:
        quad = max(rank for rank, cnt in counts.items() if cnt == 4)
        kicker = max(rank for rank, cnt in counts.items() if cnt == 1)
        return tuple_to_decimal(8, (quad, kicker))
    if 3 in count_vals and 2 in count_vals:
        triple = max(rank for rank, cnt in counts.items() if cnt == 3)
        pair = max(rank for rank, cnt in counts.items() if cnt == 2)
        return tuple_to_decimal(7, (triple, pair))
    if is_flush:
        sorted_desc = tuple(sorted((card.value for card in cards), reverse=True))
        return tuple_to_decimal(6, sorted_desc)
    if straight:
        return tuple_to_decimal(5, (high_straight,))
    if 3 in count_vals:
        triple = max(rank for rank, cnt in counts.items() if cnt == 3)
        kickers = tuple(
            sorted((rank for rank, cnt in counts.items() if cnt == 1), reverse=True)
        )
        return tuple_to_decimal(4, (triple,) + kickers)
    if count_vals.count(2) == 2:
        pairs = tuple(
            sorted((rank for rank, cnt in counts.items() if cnt == 2), reverse=True)
        )
        kicker = max(rank for rank, cnt in counts.items() if cnt == 1)
        return tuple_to_decimal(3, pairs + (kicker,))
    if 2 in count_vals:
        pair = max(rank for rank, cnt in counts.items() if cnt == 2)
        kickers = tuple(
            sorted((rank for rank, cnt in counts.items() if cnt == 1), reverse=True)
        )
        return tuple_to_decimal(2, (pair,) + kickers)
    sorted_desc = tuple(sorted((card.value for card in cards), reverse=True))
    return tuple_to_decimal(1, sorted_desc)


def evaluate_three(cards: List[Card]) -> float:
    """
    Evaluate a 3–card hand for Chinese Poker.
    Returns a decimal number where:
    - Integer part is the category rank (1-4)
    - Decimal part contains tiebreaker values (e.g., high cards or kickers).
    Categories:
       4: Three of a Kind
       2: Pair
       1: High Card
    """
    counts = collections.Counter(card.value for card in cards)
    count_vals = sorted(counts.values(), reverse=True)
    if 3 in count_vals:
        triple = max(rank for rank, cnt in counts.items() if cnt == 3)
        return tuple_to_decimal(4, (triple,))
    if 2 in count_vals:
        pair = max(rank for rank, cnt in counts.items() if cnt == 2)
        kicker = max(rank for rank, cnt in counts.items() if cnt == 1)
        return tuple_to_decimal(2, (pair, kicker))
    sorted_desc = tuple(sorted((card.value for card in cards), reverse=True))
    return tuple_to_decimal(1, sorted_desc)


def evaluate_hand(cards: List[Card]) -> float:
    """
    Evaluate a hand (list of Card objects). Must be either 3 or 5 cards.
    Returns a decimal number.
    """
    if len(cards) == 5:
        return evaluate_five(cards)
    elif len(cards) == 3:
        return evaluate_three(cards)
    else:
        raise ValueError("Hand must contain either 3 or 5 cards")


def compare_hands(hand1: List[Card], hand2: List[Card]) -> int:
    """
    Compare two hands. Returns 1 if hand1 is stronger, -1 if hand2 is stronger, 0 if tie.
    """
    score1 = evaluate_hand(hand1)
    score2 = evaluate_hand(hand2)
    if score1 > score2:
        return 1
    elif score1 < score2:
        return -1
    else:
        return 0


# -------------------- Combination Dictionary Generation --------------------


def card_to_id(card: Card) -> int:
    """
    Convert a card to a unique ID (0 to 51) based on SUITS and RANKS order.
    """
    suit_index = SUITS.index(card.suit)
    rank_index = RANKS.index(card.rank)
    return suit_index * len(RANKS) + rank_index


def id_to_card(card_id: int) -> str:
    """
    Convert a card ID (0 to 51) back to its string representation.
    """
    num_ranks = len(RANKS)
    suit_index = card_id // num_ranks
    rank_index = card_id % num_ranks
    return f"{RANKS[rank_index]}{SUITS[suit_index]}"


def generate_combinations_dict(n: int) -> Dict[Tuple[int, ...], float]:
    """
    Generate a dictionary mapping every n–card combination (as a sorted tuple of card IDs)
    to its evaluation score.
    The keys are sorted tuples of card IDs (integers from 0 to 51) to ensure uniqueness.
    """
    deck: List[Card] = [Card(rank, suit) for suit in SUITS for rank in RANKS]
    comb_dict: Dict[Tuple[int, ...], float] = {}
    total = {3: 22100, 5: 2598960}[n]
    count = 0
    for combo in itertools.combinations(deck, n):
        key = tuple(sorted(card_to_id(card) for card in combo))
        score = evaluate_hand(combo)
        comb_dict[key] = score
        count += 1
        if count % 100000 == 0:
            print(f"Processed {count}/{total} combinations...")
    return comb_dict


def save_combinations_dict(n: int, filename: str) -> None:
    """
    Generate and save the n–card combination dictionary to a pickle file.
    """
    print(f"Generating dictionary for {n}–card combinations...")
    comb_dict = generate_combinations_dict(n)
    with open(filename, "wb") as f:
        pickle.dump(comb_dict, f)
    print(f"Saved {len(comb_dict)} combinations to {filename}")


def save_combinations_csv(n: int, filename: str) -> None:
    """
    Generate and save the n–card combination dictionary to a CSV file.
    The CSV will have columns: combination, score.
    """
    print(f"Generating CSV for {n}–card combinations...")
    deck: List[Card] = [Card(rank, suit) for suit in SUITS for rank in RANKS]
    total = {3: 22100, 5: 2598960}[n]
    count = 0
    with open(filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["combination", "score"])
        for combo in itertools.combinations(deck, n):
            key_ids = tuple(sorted(card_to_id(card) for card in combo))
            score = evaluate_hand(combo)
            # Convert the sorted IDs into a human-readable string of cards
            combo_cards = [id_to_card(cid) for cid in key_ids]
            combo_str = " ".join(combo_cards)
            writer.writerow([combo_str, score])
            count += 1
            if count % 100000 == 0:
                print(f"Processed {count}/{total} combinations...")
    print(f"Saved CSV for {count} combinations to {filename}")


# -------------------- Demo / Main --------------------


def demo_deal(num_players: int) -> List[List[Card]]:
    """
    Shuffle a deck and deal 13 cards to each of num_players.
    """
    deck = Deck()
    deck.shuffle()
    hands = deck.deal(num_players)
    for i, hand in enumerate(hands):
        print(f"Player {i + 1}'s hand ({len(hand)} cards): {hand}")
    return hands


def main() -> None:
    """Main entry point for the poker game."""
    parser = argparse.ArgumentParser(
        description="Chinese Poker Simulation and Hand Dictionary Generator"
    )
    parser.add_argument(
        "--deal",
        type=int,
        default=0,
        help="Deal 13 cards to the specified number of players (1–4).",
    )
    parser.add_argument(
        "--generate",
        type=int,
        choices=[3, 5],
        help="Generate a dictionary of all n–card combinations (n=3 or 5) and score them.",
    )
    parser.add_argument(
        "--outfile",
        help="Output file for the generated dictionary (required with --generate).",
    )
    parser.add_argument(
        "--csv",
        help="Also save the combinations to this CSV file for verification.",
    )

    args = parser.parse_args()

    if args.deal:
        if not 1 <= args.deal <= 4:
            parser.error("Number of players must be between 1 and 4")
        demo_deal(args.deal)
    elif args.generate:
        if not args.outfile:
            parser.error("--outfile is required with --generate")
        save_combinations_dict(args.generate, args.outfile)
        if args.csv:
            save_combinations_csv(args.generate, args.csv)


if __name__ == "__main__":
    main()