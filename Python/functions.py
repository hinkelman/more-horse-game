"""
functions.py — Python port of functions.R

Key Python-vs-R differences to keep in mind:
  - Python uses 0-based indexing; R uses 1-based indexing.
  - Python dicts  ≈  R named vectors / lists.
  - numpy arrays  ≈  R numeric vectors.
  - pandas DataFrames ≈  R data.frames / tibbles.
  - `None`  ≈  R `NULL`.
  - List comprehensions / generator expressions ≈ R's sapply / lapply.
"""

import numpy as np
import pandas as pd
from itertools import product
from collections import Counter

# In R this is a named character vector: horse_colors["7"] == "#b15928"
# In Python we use a plain dict with the same string keys.
horse_colors = {
    "2": "#a6cee3",
    "12": "#1f78b4",
    "3": "#b2df8a",
    "11": "#33a02c",
    "4": "#fb9a99",
    "10": "#e31a1c",
    "5": "#fdbf6f",
    "9": "#ff7f00",
    "6": "#cab2d6",
    "8": "#6a3d9a",
    "7": "#b15928",
}

# Count how many of the 36 combinations give each sum (≈ dplyr::count).
roll_values = np.arange(2, 13)  # [2, 3, ..., 12]
roll_counts = Counter(d1 + d2 for d1, d2 in product(range(1, 7), range(1, 7)))
counts = np.array([roll_counts[k] for k in roll_values])

# "steps" is the number of spaces on the board assigned to each horse number.
# These are hard-coded in the original R code.
steps = np.array([3, 6, 8, 11, 14, 16, 14, 11, 8, 6, 3])

# Probabilities: fraction of the 36 outcomes for each roll value.
prob = counts / counts.sum()  # sums to 1.0
prob_steps = steps / steps.sum()  # alternative weighting (not used in sim)

rolls_df = pd.DataFrame(
    {
        "roll": roll_values,  # 2–12
        "n": counts,  # raw counts out of 36
        "steps": steps,  # board spaces per horse
        "prob": prob,  # P(roll)
        "prob_steps": prob_steps,  # P(roll) weighted by board steps
    }
)


def roll(n, replace=True, prob=None):
    if prob is None:
        prob = rolls_df["prob"].values  # .values converts pandas Series → numpy array

    # np.random.choice ≈ R's sample()
    return np.random.choice(roll_values, size=n, replace=replace, p=prob)


def get_kitty(base_value, scratches, rolls=None):
    # Starting kitty — same formula as R
    init = 4 * base_value * (4 + 3 + 2 + 1)  # = 4 * 0.25 * 10 = 10.0

    vals = []
    if rolls is not None and len(rolls) > 0:
        # Convert scratches to a list so we can use .index() for position lookup.
        scratches_list = list(scratches)
        for x in rolls:
            if x in scratches_list:
                # R's which() returns 1-based position; Python's list.index() is 0-based,
                # so we add 1 to match R's multiplier logic.
                mult = scratches_list.index(x) + 1  # 1-based multiplier
                vals.append(mult * base_value)
            else:
                vals.append(0)

    # sum([init] + vals)  ≈  R's sum(c(init, vals))
    return init + sum(vals)


def get_steps_remain(scratches, rolls, rdf=None):
    if rdf is None:
        rdf = rolls_df

    # Count how many times each roll value (2–12) has appeared.
    rolls_arr = (
        np.array(rolls, dtype=int) if len(rolls) > 0 else np.array([], dtype=int)
    )

    # Build a count vector indexed 2–12 (11 entries).
    counts = np.zeros(11, dtype=int)
    for r in rolls_arr:
        if 2 <= r <= 12:
            counts[r - 2] += 1  # shift: index 0 → roll value 2

    # Remaining steps = total steps − steps already taken.
    steps_arr = rdf["steps"].values  # numpy array, length 11
    remaining = steps_arr - counts  # element-wise subtraction

    # Build a dict {horse_str: remaining_steps_or_None}
    result = {}
    scratches_set = set(int(s) for s in scratches)  # fast membership test
    for i, horse in enumerate(range(2, 13)):  # horse numbers 2–12
        horse_str = str(horse)
        if horse in scratches_set:
            result[horse_str] = None
        else:
            result[horse_str] = int(remaining[i])

    return result


def sim_one_game(scratches, rolls, winner_only=True):
    steps_remain = get_steps_remain(scratches, rolls)
    active = {k: (v is not None) for k, v in steps_remain.items()}

    # sr holds mutable remaining step counts
    sr = {k: v for k, v in steps_remain.items()}  # shallow copy

    # Sanity check: no active horse should already be at 0 or below.
    for horse_str, remaining in sr.items():
        if active[horse_str] and remaining < 0:
            raise ValueError("Winner was already determined")

    # Pre-generate a pool of future rolls (avoids repeated numpy calls).
    # 200 is a conservative upper bound
    sim_rolls_pool = roll(200)

    future_rolls_used = []  # track which simulated rolls we actually consumed

    i = 0  # index into sim_rolls_pool
    # Keep rolling until at least one active horse reaches 0 remaining steps.
    while all(sr[h] > 0 for h in sr if active[h]):
        r = int(sim_rolls_pool[i])  # current roll, integer 2–12
        i += 1
        future_rolls_used.append(r)

        horse_str = str(r)
        # Only decrement if this horse is active in the current game.
        if active.get(horse_str, False):
            sr[horse_str] -= 1

    # The winner is the horse(s) whose remaining steps just hit 0.
    # In practice exactly one horse wins, but we handle ties gracefully.
    winner = [h for h, v in sr.items() if active[h] and v == 0]
    # R returns a character scalar; we return the first (and typically only) element.
    winner_str = winner[0] if len(winner) == 1 else winner

    if winner_only:
        return winner_str
    else:
        # Combine the historical rolls with the newly simulated ones.
        all_rolls = list(rolls) + future_rolls_used
        return {"winner": winner_str, "rolls": all_rolls}


def sim_win_prob(scratches, rolls=None, n_sim=1000):
    if rolls is None:
        rolls = []

    # Run n_sim simulations and collect the winner from each.
    # This is the Python equivalent of R's replicate().
    winners = [sim_one_game(scratches, rolls, winner_only=True) for _ in range(n_sim)]

    # Count wins per horse and normalise — ≈ R's table(winners) / n_sim.
    win_counts = {str(h): 0 for h in range(2, 13)}
    for w in winners:
        win_counts[str(w)] = win_counts.get(str(w), 0) + 1

    return {horse: count / n_sim for horse, count in win_counts.items()}


def sim_new_game(scratches, base_value=0.25):
    # Simulate from the beginning (no prior rolls).
    sim = sim_one_game(scratches, rolls=[], winner_only=False)

    # Calculate the kitty based on all rolls that occurred during the game.
    kitty = get_kitty(base_value, scratches, sim["rolls"])

    return {"winner": sim["winner"], "kitty": kitty}
