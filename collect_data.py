#!/usr/bin/env python3
"""
collect_data.py

Simulates a specified number of Chinese Poker games to gather data. For each game:
  - Shuffles a full deck and deals 4 hands (13 cards each).
  - Computes all valid arrangements (front/middle/back splits) for each hand using
    precomputed evaluation dictionaries.
  - Uses best-response dynamics (from game_equilibrium.py) to find a Nash equilibrium outcome.
  - Records each player's front (3 cards), middle (5 cards), and back (5 cards)
    evaluation scores, along with their final payoff.

Results are appended to a CSV file with columns:
  - Game, Player, Front Score, Middle Score, Back Score, Final Score

Usage:
    python collect_data.py --games 10 --csv results.csv --max_iter 100
"""

import argparse
import csv
import multiprocessing as mp
import os
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from queue import Empty
from typing import List, Dict, Tuple

from game_equilibrium import ChinesePokerGame, load_evaluation_dicts
from poker import Deck


@dataclass
class GameResult:
    """Stores the results of a single game for one player.

    Attributes:
        game_id: Unique identifier for the game.
        player_id: Player number (1 to 4).
        front_score: Evaluation score for the front hand (3 cards).
        middle_score: Evaluation score for the middle hand (5 cards).
        back_score: Evaluation score for the back hand (5 cards).
        final_score: Final payoff for the player.
    """

    game_id: int
    player_id: int
    front_score: float
    middle_score: float
    back_score: float
    final_score: float


def simulate_game(
    game_id: int, three_dict: Dict, five_dict: Dict, max_iter: int, verbose: bool = False
) -> List[GameResult]:
    """Simulate one game of Chinese Poker and return results for all players.

    Args:
        game_id: Unique identifier for the game.
        three_dict: Dictionary mapping 3-card combinations to scores.
        five_dict: Dictionary mapping 5-card combinations to scores.
        max_iter: Maximum iterations for Nash equilibrium search.
        verbose: If True, print progress during equilibrium search.

    Returns:
        A list of GameResult objects for all players in the game.
    """
    try:
        deck = Deck()
        deck.shuffle()
        hands = deck.deal(4)

        game = ChinesePokerGame(hands, three_dict, five_dict)
        game.find_nash_equilibrium(max_iter=max_iter, verbose=verbose)

        results = [
            GameResult(
                game_id=game_id,
                player_id=i + 1,
                front_score=player.arrangements[player.current_strategy].front_eval,
                middle_score=player.arrangements[player.current_strategy].middle_eval,
                back_score=player.arrangements[player.current_strategy].back_eval,
                final_score=player.payoffs[-1],
            )
            for i, player in enumerate(game.players)
        ]
        return results

    except Exception as e:
        print(f"Error in game {game_id}: {e}", file=sys.stderr)
        return []


def get_last_game_id(csv_file: str) -> int:
    """Get the last game ID from an existing CSV file.

    Args:
        csv_file: Path to the CSV file.

    Returns:
        The highest game ID found, or 0 if the file doesn't exist or is empty.
    """
    if not os.path.exists(csv_file):
        return 0

    try:
        with open(csv_file, "r", newline="") as f:
            next(f)  # Skip header
            return max((int(line.split(",")[0]) for line in f if line.strip()), default=0)
    except (ValueError, FileNotFoundError, StopIteration) as e:
        print(f"Warning: Error reading last game ID: {e}", file=sys.stderr)
        return 0


def write_results(results: List[GameResult], output_file: str, write_header: bool = False) -> None:
    """Write or append game results to a CSV file.

    Args:
        results: List of GameResult objects to write.
        output_file: Path to the output CSV file.
        write_header: If True, write the header row (used for new files).
    """
    mode = "w" if write_header else "a"
    headers = ["Game", "Player", "Front Score", "Middle Score", "Back Score", "Final Score"]

    with open(output_file, mode, newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(headers)
        for result in results:
            writer.writerow(
                [
                    result.game_id,
                    result.player_id,
                    str(result.front_score),
                    str(result.middle_score),
                    str(result.back_score),
                    f"{result.final_score:.1f}",
                ]
            )


def progress_monitor(total_games: int, result_queue: mp.Queue, done_event: threading.Event) -> int:
    """Monitor simulation progress across all processes.

    Args:
        total_games: Total number of games to simulate.
        result_queue: Queue for receiving completion signals from processes.
        done_event: Event to signal when all simulations are complete.

    Returns:
        The number of successfully completed games.
    """
    success_count = 0
    completed = 0
    update_batch = 0
    last_update = time.time()
    start_time = time.time()

    while not done_event.is_set() or not result_queue.empty():
        try:
            while True:
                success = result_queue.get_nowait()
                if success:
                    success_count += 1
                completed += 1
                update_batch += 1

                current_time = time.time()
                if current_time - last_update >= 0.5 and update_batch > 0:
                    elapsed = current_time - start_time
                    games_per_sec = completed / elapsed if elapsed > 0 else 0
                    eta = (total_games - completed) / games_per_sec if games_per_sec > 0 else 0
                    progress = completed / total_games * 100

                    msg = (
                        f"Progress: {progress:6.1f}% | "
                        f"Games: {completed:4d}/{total_games:<4d} | "
                        f"Success: {success_count:3d}/{completed:<3d} | "
                        f"Speed: {games_per_sec:5.1f} games/s | "
                        f"ETA: {eta:5.1f}s"
                    )
                    sys.stdout.write(f"\r{msg}")
                    sys.stdout.flush()

                    update_batch = 0
                    last_update = current_time

        except Empty:
            if update_batch > 0 and time.time() - last_update >= 0.5:
                elapsed = time.time() - start_time
                games_per_sec = completed / elapsed if elapsed > 0 else 0
                eta = (total_games - completed) / games_per_sec if games_per_sec > 0 else 0
                progress = completed / total_games * 100

                msg = (
                    f"Progress: {progress:6.1f}% | "
                    f"Games: {completed:4d}/{total_games:<4d} | "
                    f"Success: {success_count:3d}/{completed:<3d} | "
                    f"Speed: {games_per_sec:5.1f} games/s | "
                    f"ETA: {eta:5.1f}s"
                )
                sys.stdout.write(f"\r{msg}")
                sys.stdout.flush()

                update_batch = 0
                last_update = time.time()
            time.sleep(0.01)

    elapsed = time.time() - start_time
    games_per_sec = completed / elapsed if elapsed > 0 else 0
    msg = (
        f"\nCompleted!\n"
        f"Total Games: {completed}/{total_games}\n"
        f"Success Rate: {success_count}/{completed} ({100.0 * success_count / completed:.1f}%)\n"
        f"Average Speed: {games_per_sec:.1f} games/s\n"
        f"Total Time: {elapsed:.1f}s\n"
    )
    sys.stdout.write(msg)
    sys.stdout.flush()

    return success_count


def simulate_games_chunk(args: Tuple[List[int], Dict, Dict, int, bool]) -> List[GameResult]:
    """Simulate a chunk of games in a single process.

    Args:
        args: Tuple containing game_ids, three_dict, five_dict, max_iter, and verbose flag.

    Returns:
        A list of GameResult objects for the chunk.
    """
    game_ids, three_dict, five_dict, max_iter, verbose = args
    results = []
    for game_id in game_ids:
        game_results = simulate_game(game_id, three_dict, five_dict, max_iter, verbose)
        results.extend(game_results)
    return results


def run_simulation(num_games: int, output_file: str, max_iter: int = 100) -> None:
    """Run multiple Chinese Poker games in parallel and save results to a CSV file.

    Args:
        num_games: Number of games to simulate.
        output_file: Path to the output CSV file.
        max_iter: Maximum iterations for Nash equilibrium search.
    """
    print("Loading evaluation dictionaries...")
    three_dict, five_dict = load_evaluation_dicts("Dict/three_card.pkl", "Dict/five_card.pkl")

    start_game_id = get_last_game_id(output_file) + 1
    print(f"Starting from game ID: {start_game_id}")

    result_queue = mp.Queue()
    done_event = threading.Event()

    monitor_thread = threading.Thread(
        target=progress_monitor,
        args=(num_games, result_queue, done_event),
    )
    monitor_thread.start()

    try:
        num_processes = min(mp.cpu_count(), 8)
        chunk_size = max(10, num_games // (num_processes * 4))
        game_chunks = [
            (
                list(range(start_game_id + i, min(start_game_id + i + chunk_size, start_game_id + num_games))),
                three_dict,
                five_dict,
                max_iter,
                False,
            )
            for i in range(0, num_games, chunk_size)
        ]

        all_results = []
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            futures = [executor.submit(simulate_games_chunk, chunk) for chunk in game_chunks]
            for future in futures:
                chunk_results = future.result()
                if chunk_results:
                    write_header = not os.path.exists(output_file)
                    write_results(chunk_results, output_file, write_header)
                    all_results.extend(chunk_results)
                    games_in_chunk = len(chunk_results) // 4
                    for _ in range(games_in_chunk):
                        result_queue.put(True)

        done_event.set()
        monitor_thread.join()

        total_games = len(all_results) // 4
        if total_games > 0:
            print(f"\nResults written to: {output_file}")
        else:
            print("\nNo results generated!")

    except KeyboardInterrupt:
        print("\nSimulation interrupted by user.")
        done_event.set()
        monitor_thread.join()
    except Exception as e:
        print(f"\nError during simulation: {e}")
        done_event.set()
        monitor_thread.join()
        raise


def main() -> None:
    """Main entry point for simulating Chinese Poker games."""
    parser = argparse.ArgumentParser(description="Simulate multiple Chinese Poker games")
    parser.add_argument("--games", type=int, default=4000, help="Number of games to simulate")
    parser.add_argument("--csv", default="results.csv", help="Output CSV file")
    parser.add_argument(
        "--max_iter", type=int, default=100, help="Maximum iterations for Nash equilibrium search"
    )
    args = parser.parse_args()

    run_simulation(args.games, args.csv, args.max_iter)


if __name__ == "__main__":
    main()