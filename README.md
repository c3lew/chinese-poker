# Chinese Poker

## Overview

Chinese Poker is a four-player card game where each player is dealt 13 cards from a standard 52-card deck. The players must then arrange their cards into three separate hands:

- **Front Hand** (3 cards) – This is the weakest hand, often consisting of high-card hands or small pairs.
- **Middle Hand** (5 cards) – This must be stronger than the front hand but weaker than the back hand.
- **Back Hand** (5 cards) – The strongest of the three hands, typically containing the best possible five-card combination.

### Game Rules
After arranging their hands, players compare each of their three hands against the corresponding hands of the other players. The scoring system is based on who has the stronger hands, with additional bonuses awarded for special hands (e.g., straights, flushes, full houses). A player must ensure their hands follow the order rule: `Front < Middle <= Back`, or else their entire hand is considered fouled, resulting in an automatic loss.

The game is played in rounds, and each player accumulates points based on the strength of their hands relative to the other players. The player with the most points at the end of multiple rounds wins.

Chinese Poker is based on the same rules as **開心鬥一番** and incorporates advanced techniques such as scoring mechanisms, expected value calculations, Nash equilibrium computations, and AI-driven data collection to optimize gameplay and decision-making.

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

## Installation

### Prerequisites

- Python 3.7+
- Required dependencies:
  ```bash
  pip install numpy pandas
  ```

## Usage

### Find Hand Arrangements
```bash
python arrange.py --hand "2C 3C AS 10D 9H 8S 7D 6C 5H 4S 3D 2H KH"
```
Generate a random hand and save valid arrangements to CSV:
```bash
python arrange.py --random --csv output.csv
```

### Play a Full Game
```bash
python play.py
```

### Compute Expected Values
```bash
python expected_value.py
```

### Simulate Games for Data Collection
```bash
python collect_data.py --games 100 --csv results.csv
```

### Analyze Score Distributions
```bash
python analyze_scores.py --results results.csv
```

## Scoring System

Scoring follows standard Chinese Poker rules:

- Players compare **Front, Middle, and Back** hands.
- Points are awarded based on who has the stronger hand in each position:
  - **Winning a hand**: +1 point
  - **Losing a hand**: -1 point
  - **Tied hands**: 0 points awarded
- The total score is calculated by summing the points from all three hands.
- Special bonuses apply for strong hands:
  - **Three of a Kind in Front**: +3 points
  - **Full House in Middle**: +2 points
  - **Four of a Kind in Middle/Back**: +4 points
  - **Straight Flush in Middle/Back**: +5 points
  - **Royal Flush in Back**: +10 points
- **Sweep Bonus** (+3 points) if a player wins all three hands.
- **Overall Bonus** (+18 points) for the best total hand.
- **Scoop Bonus** (+6 points) if a player wins all hands against all opponents.

## AI and Game Strategy

### Challenges of Chinese Poker

Chinese Poker is an **imperfect information game**, meaning that players do not have full knowledge of their opponents' hands. This adds complexity to strategic decision-making compared to perfect information games like chess. Players must **estimate probabilities, assess risk**, and make decisions based on incomplete data.

Additionally, the game involves **multiple strategic layers**, including:
- Arranging hands optimally.
- Predicting opponent strategies.
- Adapting based on available information and scoring potential.

### How AI Helps Solve These Challenges

One key advantage of this approach is the ability to **calculate the expected value** of an initial hand efficiently. By generating all **15,000 to 20,000** possible valid hand arrangements, scoring them, and selecting the optimal configuration, the system can determine the best possible hand **in less than a second**. This rapid computation allows for strategic decision-making in real-time scenarios.

## TODO List

- Develop **Deep Learning AI** for optimal hand arrangement and implement 1v3 mode (AI vs three calculated stats).
- Gather real-world player data from **開心鬥一番** to analyze player performance.
- Integrate **image recognition** to extract gameplay data from **開心鬥一番**, enabling AI to compete against real players.
- Design and implement a **GUI interface** for enhanced user interaction.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

