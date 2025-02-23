#!/usr/bin/env python3
"""
expected_value.py

Calculates expected values for Chinese Poker hands based on:
1. Basic scoring rules (+1/-1 for position wins)
2. Bonus scoring for special hands
3. Overall bonus for having the best hands in all positions

Uses statistical information from position info files to estimate probabilities
and calculate expected values.
"""

import pickle
from typing import Dict, Any
import pandas as pd


class ExpectedValueCalculator:
    """Calculates expected values for Chinese Poker hands.

    Attributes:
        front_info (pd.DataFrame): Statistical data for the front position (3 cards).
        middle_info (pd.DataFrame): Statistical data for the middle position (5 cards).
        back_info (pd.DataFrame): Statistical data for the back position (5 cards).
        front_categories (Dict[float, int]): Mapping of front scores to categories.
        middle_categories (Dict[float, int]): Mapping of middle scores to categories.
        back_categories (Dict[float, int]): Mapping of back scores to categories.
    """

    def __init__(self) -> None:
        """Initialize calculator with position statistics from pickle files."""
        with open('Dict/front_info.pkl', 'rb') as f:
            self.front_info: pd.DataFrame = pickle.load(f)
        with open('Dict/middle_info.pkl', 'rb') as f:
            self.middle_info: pd.DataFrame = pickle.load(f)
        with open('Dict/back_info.pkl', 'rb') as f:
            self.back_info: pd.DataFrame = pickle.load(f)

        self.front_categories = self._create_category_map(self.front_info)
        self.middle_categories = self._create_category_map(self.middle_info)
        self.back_categories = self._create_category_map(self.back_info)

    def _create_category_map(self, df: pd.DataFrame) -> Dict[float, int]:
        """Create a mapping of scores to their categories.

        Args:
            df: DataFrame containing 'score' column.

        Returns:
            A dictionary mapping rounded scores to their integer categories.
        """
        return {round(float(row['score'])): int(float(row['score'])) for _, row in df.iterrows()}

    def _get_category(self, score: float, position: str) -> int:
        """Get the category for a given score in a specific position.

        Args:
            score: The score to categorize.
            position: The position ('front', 'middle', or 'back').

        Returns:
            The integer category of the score.
        """
        rounded_score = round(score)
        if position == 'front':
            return self.front_categories.get(rounded_score, 0)
        elif position == 'middle':
            return self.middle_categories.get(rounded_score, 0)
        else:  # back
            return self.back_categories.get(rounded_score, 0)

    def _calculate_basic_ev(self, score: float, win_rate: float, position: str) -> float:
        """Calculate the basic expected value for a position.

        Args:
            score: The score of the hand in the position.
            win_rate: The win rate of the hand in the position.
            position: The position ('front', 'middle', or 'back').

        Returns:
            The basic expected value for the position.
        """
        if position == 'front':
            base = 3.0  # Win/lose against 3 opponents
            p = 0.0079  # Probability of three of a kind

            lose_value = (
                (1 - p) ** 3 * base
                + 3 * p * (1 - p) ** 2 * (base + 3)
                + 3 * p ** 2 * (1 - p) * (base + 6)
                + p ** 3 * (base + 9)
            )
            return win_rate * base - (1 - win_rate) * lose_value

        elif position == 'middle':
            base = 3.0
            p_fh = 0.0165  # Probability of full house
            p_4k = 0.00015  # Probability of four of a kind

            lose_value = (
                (1 - p_fh) ** 3 * (1 - p_4k) ** 3 * base
                + 3 * p_fh * (1 - p_fh) ** 2 * (1 - p_4k) ** 3 * (base + 2)
                + 3 * (1 - p_fh) ** 3 * p_4k * (1 - p_4k) ** 2 * (base + 4)
                + (1 - (1 - p_fh) ** 3) * (1 - (1 - p_4k) ** 3) * (base + 6)
            )
            return win_rate * base - (1 - win_rate) * lose_value

        else:  # back
            base = 3.0
            p_4k = 0.0327  # Probability of four of a kind
            p_rf = 0.00865 + 0.00178  # Combined probability of royal flush

            lose_value = (
                (1 - p_4k) ** 3 * (1 - p_rf) ** 3 * base
                + 3 * p_4k * (1 - p_4k) ** 2 * (1 - p_rf) ** 3 * (base + 4)
                + 3 * (1 - p_4k) ** 3 * p_rf * (1 - p_rf) ** 2 * (base + 5)
                + (1 - (1 - p_4k) ** 3) * (1 - (1 - p_rf) ** 3) * (base + 9)
            )
            return win_rate * base - (1 - win_rate) * lose_value

    def _calculate_bonus_ev(self, score: float, win_rate: float, position: str) -> float:
        """Calculate the bonus expected value for special hands.

        Args:
            score: The score of the hand in the position.
            win_rate: The win rate of the hand in the position.
            position: The position ('front', 'middle', or 'back').

        Returns:
            The bonus expected value for the position.
        """
        category = self._get_category(score, position)
        if position == 'front':
            if category == 4:  # Three of a kind
                return win_rate * 3 * 3  # Bonus * number of opponents
        elif position == 'middle':
            if category == 7:  # Full house
                return win_rate * 2 * 3
            elif category == 8:  # Four of a kind
                return win_rate * 4 * 3
            elif category in (9, 10):  # Straight flush or royal flush
                return win_rate * 5 * 3
        else:  # back
            if category == 8:  # Four of a kind
                return win_rate * 4 * 3
            elif category in (9, 10):  # Straight flush or royal flush
                return win_rate * 5 * 3
        return 0.0

    def _calculate_sweep_bonus_ev(self, front_wr: float, middle_wr: float, back_wr: float) -> float:
        """Calculate the expected value for sweeping opponents and being swept.

        Args:
            front_wr: Win rate for the front position.
            middle_wr: Win rate for the middle position.
            back_wr: Win rate for the back position.

        Returns:
            The expected value for sweep bonuses.
        """
        our_sweep_prob = front_wr * middle_wr * back_wr
        opp_sweep_prob = (1 - front_wr) * (1 - middle_wr) * (1 - back_wr)
        sweep_ev_per_opponent = 3 * our_sweep_prob - 3 * opp_sweep_prob
        return 3 * sweep_ev_per_opponent

    def _calculate_overall_bonus_ev(self, front_wr: float, middle_wr: float, back_wr: float) -> float:
        """Calculate the expected value for having the best hands in all positions.

        Args:
            front_wr: Win rate for the front position.
            middle_wr: Win rate for the middle position.
            back_wr: Win rate for the back position.

        Returns:
            The expected value for the overall bonus.
        """
        best_front_prob = front_wr ** 3
        best_middle_prob = middle_wr ** 3
        best_back_prob = back_wr ** 3
        best_all_prob = best_front_prob * best_middle_prob * best_back_prob
        return 18 * best_all_prob

    def calculate_total_ev(self, arrangement: Dict[str, float]) -> Dict[str, Any]:
        """Calculate the total expected value for an arrangement.

        Args:
            arrangement: A dictionary containing 'front_score', 'front_win_rate',
                         'middle_score', 'middle_win_rate', 'back_score', 'back_win_rate'.

        Returns:
            A dictionary with the total expected value and its components.
        """
        front_score = arrangement['front_score']
        front_wr = arrangement['front_win_rate']
        middle_score = arrangement['middle_score']
        middle_wr = arrangement['middle_win_rate']
        back_score = arrangement['back_score']
        back_wr = arrangement['back_win_rate']

        front_basic_ev = self._calculate_basic_ev(front_score, front_wr, 'front')
        middle_basic_ev = self._calculate_basic_ev(middle_score, middle_wr, 'middle')
        back_basic_ev = self._calculate_basic_ev(back_score, back_wr, 'back')

        front_bonus_ev = self._calculate_bonus_ev(front_score, front_wr, 'front')
        middle_bonus_ev = self._calculate_bonus_ev(middle_score, middle_wr, 'middle')
        back_bonus_ev = self._calculate_bonus_ev(back_score, back_wr, 'back')

        sweep_bonus_ev = self._calculate_sweep_bonus_ev(front_wr, middle_wr, back_wr)
        overall_bonus_ev = self._calculate_overall_bonus_ev(front_wr, middle_wr, back_wr)

        total_ev = (
            front_basic_ev + middle_basic_ev + back_basic_ev
            + front_bonus_ev + middle_bonus_ev + back_bonus_ev
            + sweep_bonus_ev + overall_bonus_ev
        )

        return {
            'total_ev': total_ev,
            'basic_ev': {
                'front': front_basic_ev,
                'middle': middle_basic_ev,
                'back': back_basic_ev,
            },
            'bonus_ev': {
                'front': front_bonus_ev,
                'middle': middle_bonus_ev,
                'back': back_bonus_ev,
            },
            'sweep_bonus_ev': sweep_bonus_ev,
            'overall_bonus_ev': overall_bonus_ev,
        }


def main() -> None:
    """Demonstrate the usage of ExpectedValueCalculator with an example arrangement."""
    calculator = ExpectedValueCalculator()

    # Example arrangement with bonus hands
    arrangement = {
        'front_score': 4.1,  # Three of a kind (category 4)
        'front_win_rate': 0.8,
        'middle_score': 7.2,  # Full house (category 7)
        'middle_win_rate': 0.7,
        'back_score': 10.3,  # Royal flush (category 10)
        'back_win_rate': 0.9,
    }

    ev = calculator.calculate_total_ev(arrangement)

    print("\nExpected Value Breakdown:")
    print(f"Basic EV:")
    print(f"  Front:  {ev['basic_ev']['front']:.3f}")
    print(f"  Middle: {ev['basic_ev']['middle']:.3f}")
    print(f"  Back:   {ev['basic_ev']['back']:.3f}")
    print(f"\nBonus EV:")
    print(f"  Front:  {ev['bonus_ev']['front']:.3f}")
    print(f"  Middle: {ev['bonus_ev']['middle']:.3f}")
    print(f"  Back:   {ev['bonus_ev']['back']:.3f}")
    print(f"\nSweep Bonus EV:    {ev['sweep_bonus_ev']:.3f}")
    print(f"Overall Bonus EV:  {ev['overall_bonus_ev']:.3f}")
    print(f"\nTotal Expected Value: {ev['total_ev']:.3f}")


if __name__ == "__main__":
    main()