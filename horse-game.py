import polars as pl

d1 = pl.DataFrame({"dice1": range(1, 7)})
d2 = pl.DataFrame({"dice2": range(1, 7)})

rolls_df = (
    d1.join(d2, how="cross")
    .with_columns((pl.col("dice1") + pl.col("dice2"))
                  .alias("roll"))
    .group_by("roll")
    .agg(pl.len().alias("n"))
    .sort("roll")
    .with_columns(
        pl.Series(name="steps",
                  values=[3, 6, 8, 11, 14, 16, 14, 11, 8, 6, 3]),
        (pl.col("n")/pl.col("n").sum()).alias("prob"),
        (pl.col("n")/pl.col("n").sum()).alias("prob_steps")
    )
)
