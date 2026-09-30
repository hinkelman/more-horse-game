"""
Key Python-vs-R translation notes:
  - pandas  ≈  dplyr / data.frame
  - tqdm progress bar  ≈  cli::cli_progress_along
  - List comprehensions / loops  ≈  lapply / sapply
"""

import pandas as pd
from tqdm import tqdm          # progress bar

from functions import roll, sim_new_game

N_REPS = 100_000      # number of games to simulate 
BASE_VALUE = 0.25     # coin denomination        

sim_list = [
    sim_new_game(scratches=roll(4, replace=False), base_value=BASE_VALUE)
    for _ in tqdm(range(N_REPS), desc="Simulating")
]

# use a list comprehension to extract each field,
# then pass the dict to pd.DataFrame() — one column per key.
sim_df = pd.DataFrame({
    "sim":    range(1, N_REPS + 1),
    "winner": [s["winner"] for s in sim_list],   # ≈ sapply(sim_list, `[[`, "winner")
    "kitty":  [s["kitty"]  for s in sim_list],   # ≈ sapply(sim_list, `[[`, "kitty")
})

# change factor levels to reflect paired probabilities
# pandas Categorical preserves order; ordered=True allows comparison operators.
# reversed to match R's rev(c(2, 12, 3, ...))
ordered_levels = [str(x) for x in reversed([2, 12, 3, 11, 4, 10, 5, 9, 6, 8, 7])]
sim_df["winner"] = pd.Categorical(sim_df["winner"], categories=ordered_levels, ordered=True)

winners = (
    sim_df["winner"]
    .value_counts(sort=False)            # count per level, in category order (≈ count())
    .rename("n")                         # name the column "n"
    .reset_index()                       # move the index (winner) into a column
)
winners["percent"] = (winners["n"] / winners["n"].sum() * 100).round(2)

kitty_avg = (
    sim_df
    .groupby("winner", observed=True)["kitty"]   # observed=True suppresses a pandas warning
    .mean()
    .rename("kitty_avg")
    .reset_index()
)

print(winners)
print(kitty_avg)
