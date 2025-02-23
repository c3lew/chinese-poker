#!/usr/bin/env python3
"""
analyze_scores.py

Analyzes hand scores from Chinese Poker games to calculate:
1. Win rates for each possible score
2. Category percentages based on score integer parts

Analyzes all three positions:
- Front (3 cards)
- Middle (5 cards)
- Back (5 cards)
"""

import pickle
import pandas as pd
from typing import List, Tuple, Optional


def load_scores(pkl_file: str) -> List[float]:
    """Load all possible scores from a pickle file and return them as a sorted list."""
    with open(pkl_file, 'rb') as f:
        score_data = pickle.load(f)
    # Extract unique scores and sort them
    all_scores = sorted(set(score_data.values()))
    return all_scores


def analyze_scores(
    results_file: str,
    score_column: str,
    possible_scores: Optional[List[float]] = None
) -> pd.DataFrame:
    """
    Analyze scores from game results for a specific position.

    Parameters:
    - results_file: Path to the CSV file containing game results.
    - score_column: Name of the column in the CSV that contains the scores.
    - possible_scores: Optional list of all possible scores. If not provided,
      unique scores from the results will be used.

    Returns:
    - A DataFrame with columns: score, count, win_rate, category_percentage.
    """
    # Load game results
    df = pd.read_csv(results_file)
    if score_column not in df.columns:
        raise ValueError(f"Column '{score_column}' not found in {results_file}")
    scores = df[score_column].values

    # If possible_scores not provided, get unique scores from results
    if possible_scores is None:
        possible_scores = sorted(set(scores))

    # Count occurrences of each score
    score_counts = pd.Series(scores).value_counts()
    total_count = len(scores)

    # Initialize results list
    results_list = []

    # Calculate metrics for each possible score
    for score in possible_scores:
        # Count occurrences of this score
        count = score_counts.get(score, 0)

        # Calculate win rate (proportion of scores that are smaller)
        smaller_count = sum(score_counts[score_counts.index < score])
        win_rate = smaller_count / total_count if total_count > 0 else 0.0

        # Calculate category percentage
        category = int(score)
        category_count = sum(
            score_counts[(score_counts.index >= category) & (score_counts.index < category + 1)]
        )
        category_percentage = (category_count / total_count) * 100 if total_count > 0 else 0.0

        results_list.append({
            'score': score,
            'count': count,
            'win_rate': win_rate,
            'category_percentage': category_percentage
        })

    # Convert to DataFrame
    results_df = pd.DataFrame(results_list)
    return results_df


def save_results(df: pd.DataFrame, position: str) -> None:
    """Save the results DataFrame to both CSV and pickle files."""
    # Generate filenames
    csv_file = f'{position}_info.csv'
    pkl_file = f'{position}_info.pkl'

    # Save CSV
    df.to_csv(csv_file, index=False)
    print(f"Results saved to {csv_file}")

    # Save pickle
    with open(pkl_file, 'wb') as f:
        pickle.dump(df, f)
    print(f"Results saved to {pkl_file}")


def analyze_position(
    position: str,
    pkl_file: str,
    score_column: str
) -> pd.DataFrame:
    """Analyze scores for a specific position and save the results."""
    print(f"\nAnalyzing {position} scores...")

    # Load possible scores
    print(f"Loading {position} scores from {pkl_file}...")
    possible_scores = load_scores(pkl_file)
    print(f"Found {len(possible_scores)} possible scores")

    # Analyze scores
    print(f"Analyzing {position} scores from game results...")
    results_df = analyze_scores('results.csv', score_column, possible_scores)

    # Print summary statistics
    print(f"\n{position} Summary Statistics:")
    print(f"Total unique scores analyzed: {len(results_df)}")
    print("\nWin Rate Statistics:")
    print(results_df['win_rate'].describe())
    print("\nCategory Percentage Statistics:")
    print(results_df['category_percentage'].describe())

    # Save results
    print(f"\nSaving {position} results...")
    save_results(results_df, position.lower())

    return results_df


def main() -> None:
    """Main entry point to analyze all three positions."""
    positions: List[Tuple[str, str, str]] = [
        ('Front', 'three_card.pkl', 'Front Score'),
        ('Middle', 'five_card.pkl', 'Middle Score'),
        ('Back', 'five_card.pkl', 'Back Score')
    ]

    for position, pkl_file, score_column in positions:
        analyze_position(position, pkl_file, score_column)


if __name__ == "__main__":
    main()