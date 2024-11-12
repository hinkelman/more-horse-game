import numpy as np
import pandas as pd


rolls_df = (
    pd.merge(pd.DataFrame({"dice1": range(1, 7)}),
             pd.DataFrame({"dice2": range(1, 7)}),
             how="cross")
    .assign(roll=lambda df: df.dice1 + df.dice2)
    .groupby("roll")
    .size()
    .reset_index(name="n")
    .assign(
        steps=lambda df: [3, 6, 8, 11, 14, 16, 14, 11, 8, 6, 3],
        prob=lambda df: df.n/df.n.sum(),
        prob_steps=lambda df: df.steps/df.steps.sum(),
        roll=lambda df: pd.Categorical(df.roll, categories=range(2, 13))
    )
)


def roll(n, replace=True, rdf=rolls_df):
    return np.random.choice(rolls_df["roll"], size=n, replace=replace, p=rolls_df["prob"])


def get_kitty(base_value, scratches, rolls=None):
    init = 4 * base_value * (4 + 3 + 2 + 1)
    vals = []
    if rolls is not None and len(rolls) > 0:
        for x in rolls:
            ind = np.where(scratches == x)[0]
            val = ind[0] * base_value if len(ind) > 0 else 0
            vals.append(val)
    return np.cumsum([init] + vals)


def get_win_prob(scratches, rolls=None, rdf=rolls_df):
    if rolls is not None and len(rolls) > 0:
        counts = pd.Series(rolls).value_counts().reindex(
            rdf["roll"], fill_value=0)
        steps_remain = rdf["steps"] - counts.values
    else:
        steps_remain = rdf["steps"]
    return pd.DataFrame({
        "horse": pd.Categorical(np.arange(2, 13)),
        "steps_remain": steps_remain,
        "win_prob": calc_win_prob(rdf["prob"], steps_remain, scratches)
    })


def calc_win_prob(roll_prob, steps_remain, scratches):
    probs = roll_prob ** steps_remain
    probs.iloc[scratches - 2] = 0
    probs_scaled = probs / probs.sum()
    if steps_remain.min() < 1:
        probs_scaled = np.where(probs_scaled == probs_scaled.max(), 1, 0)
    return probs_scaled


def sim_game(base_value):
    scratches = roll(4, replace=False)
    rolls = roll(1)
    wp = get_win_prob(scratches, rolls)
    while wp["steps_remain"].min() > 0:
        rolls = np.append(rolls, roll(1))
        wp = get_win_prob(scratches, rolls)
    return pd.DataFrame({
        "rolls": [len(rolls)],
        "kitty": [get_kitty(base_value, scratches, rolls).max()],
        "winner": wp.loc[wp["steps_remain"] == 0, "horse"].values[0]
    })


reps = 50
base_value = 0.25
sim_df = sim_game(base_value)
for i in range(1, reps):
    sim_df = pd.concat([sim_df, sim_game(base_value)], ignore_index=True)
