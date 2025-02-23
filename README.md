# Chinese Poker

## Overview

Chinese Poker is a four-player card game where each player is dealt 13 cards and must arrange them into three hands:

- **Front** (3 cards)
- **Middle** (5 cards)
- **Back** (5 cards)

The goal is to create the strongest possible hand while following the ordering rule: `Front < Middle <= Back`. The game is based on the same rules as **開心鬥一番** and includes scoring mechanisms, expected value calculations, Nash equilibrium computations, and data collection to improve AI decision-making.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Find Hand Arrangements](#find-hand-arrangements)
  - [Play a Full Game](#play-a-full-game)
  - [Compute Expected Values](#compute-expected-values)
  - [Simulate Games for Data Collection](#simulate-games-for-data-collection)
  - [Analyze Score Distributions](#analyze-score-distributions)
- [Scoring System](#scoring-system)
- [AI and Game Strategy](#ai-and-game-strategy)
- [TODO List](#todo-list)
- [License](#license)

Chinese Poker is a four-player card game where each player is dealt 13 cards and must arrange them into three hands:

- **Front** (3 cards)
- **Middle** (5 cards)
- **Back** (5 cards)

The goal is to create the strongest possible hand while following the ordering rule: `Front < Middle <= Back`. The game is based on the same rules as **開心鬥一番** and includes scoring mechanisms, expected value calculations, Nash equilibrium computations, and data collection to improve AI decision-making.



## Features

- **Hand Arrangement**: Automatically finds valid arrangements for a given 13-card hand.
- **Scoring System**: Evaluates hands based on poker rules and assigns scores.
- **Nash Equilibrium Analysis**: Uses best-response dynamics to compute equilibrium strategies.
- **Expected Value Calculation**: Computes optimal arrangements using probability-based scoring.
- **Data Collection & Analysis**: Simulates games and stores results for statistical evaluation.
- **Fast Scoring Implementation**: Optimized hand evaluation and comparison functions.

## Project Structure

The following are the main components of the project:

- `analyze_scores.py` – Analyzes game results and score distributions.
- `arrange.py` – Computes valid hand arrangements.
- `arrange_with_stats.py` – Extended hand arrangement with statistical analysis.
- `collect_data.py` – Simulates games to collect training data.
- `expected_value.py` – Computes expected value for hands.
- `game_equilibrium.py` – Implements Nash equilibrium strategies.
- `play.py` – Simulates and evaluates a full game.
- `poker.py` – Poker deck and hand evaluation functions.
- `scoring_fast.py` – Optimized scoring and arrangement comparison.
- `Dict/` – Contains precomputed evaluation dictionaries (pkl files), which are too large for GitHub. Users can either generate their own pkl files and place them in the `Dict/` folder or download them from [Google Drive](https://drive.google.com/drive/folders/16xG5Q71OJtGZQXdtI4nxHSs1wa2llQZe?usp=drive_link).

The following are the main components of the project:

- `analyze_scores.py` – Analyzes game results and score distributions.
- `arrange.py` – Computes valid hand arrangements.
- `arrange_with_stats.py` – Extended hand arrangement with statistical analysis.
- `collect_data.py` – Simulates games to collect training data.
- `expected_value.py` – Computes expected value for hands.
- `game_equilibrium.py` – Implements Nash equilibrium strategies.
- `play.py` – Simulates and evaluates a full game.
- `poker.py` – Poker deck and hand evaluation functions.
- `scoring_fast.py` – Optimized scoring and arrangement comparison.
- `Dict/` – Precomputed evaluation dictionaries (pkl files).

```
Chinese Poker/
├── analyze_scores.py        # Analyzes hand scores and win rates
├── arrange.py               # Finds all valid hand arrangements
├── arrange_with_stats.py    # Extended arrangement with statistical analysis
├── collect_data.py          # Simulates games to collect training data
├── expected_value.py        # Computes expected value for hands
├── game_equilibrium.py      # Computes Nash equilibrium strategies
├── play.py                  # Simulates a full game with scoring
├── poker.py                 # Poker deck and hand evaluation functions
├── scoring_fast.py          # Optimized scoring and arrangement comparison
└── Dict/                    # Precomputed evaluation dictionaries (pkl files)
```

## Installation

### Prerequisites

- Python 3.7+
- Required dependencies (install via pip):
  ```bash
  pip install numpy pandas
  ```

## Usage

### Find Hand Arrangements

To find valid hand arrangements for a specific hand:

```bash
python arrange.py --hand "2C 3C AS 10D 9H 8S 7D 6C 5H 4S 3D 2H KH"
```

To generate a random hand and save valid arrangements to CSV:

```bash
python arrange.py --random --csv output.csv
```

### Play a Full Game

Run a simulation of a 4-player game with optimal arrangements:

```bash
python play.py
```

### Compute Expected Values

To evaluate hands based on statistical data:

```bash
python expected_value.py
```

### Simulate Games for Data Collection

To generate a dataset of simulated games:

```bash
python collect_data.py --games 100 --csv results.csv
```

### Analyze Score Distributions

To analyze hand scores and win rates:

```bash
python analyze_scores.py --results results.csv
```

## Scoring System

Scoring follows standard Chinese Poker rules:

- Players compare **Front, Middle, and Back** hands.
- Points are awarded based on who has the stronger hand in each position.
- Special bonuses apply for strong hands:
  - **Three of a Kind in Front**: +3 points
  - **Full House in Middle**: +2 points
  - **Four of a Kind in Middle/Back**: +4 points
  - **Straight Flush in Middle/Back**: +5 points
  - **Royal Flush in Back**: +10 points
- **Sweep Bonus** (+3 points) if a player wins all three hands.
- **Overall Bonus** (+18 points) for the best total hand.
- **Scoop Bonus** (+6 points) if a player wins all hands against all opponents.
- **Tied Hands** result in no points being awarded for that position.
- **Scoop Bonus** (+6 points) if a player wins all hands against all opponents.
- **Tied Hands** result in no points being awarded for that position.

- Players compare **Front, Middle, and Back** hands.
- Points are awarded based on who has the stronger hand.
- Special bonuses apply for strong hands (e.g., three of a kind in Front, four of a kind in Middle/Back).
- **Sweep Bonus** (+3 points) if a player wins all three hands.
- **Overall Bonus** (+18 points) for the best total hand.

## AI and Game Strategy

The project leverages:

- **Precomputed Evaluations**: Uses hand strength lookup tables.
- **Monte Carlo Simulations**: Estimates expected values for hands.
- **Best-Response Dynamics**: Finds Nash equilibrium strategies.
- **Game-Theoretic Analysis**: Optimizes play based on expected payoffs.

## TODO List

- Develop **Deep Learning AI** for optimal hand arrangement and implement 1v3 mode (AI vs three calculated stats).
- Gather real-world player data from **開心鬥一番** to analyze player performance.
- Integrate **image recognition** to extract gameplay data from **開心鬥一番**, enabling AI to compete against real players.
- Design and implement a **GUI interface** for enhanced user interaction with the tool.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

