import knmi

df = knmi.get_day_data_dataframe(
    stations=[260],
    start="19570701",
    end="20190701",
    inseason=False,
    variables=["RH", "EV24"],
)
df.loc[df["RH"] < 0, "RH"] = 0.5
df["RH"] /= 10
df["EV24"] /= 10
df["NOV"] = df["RH"] - df["EV24"]
df["hydr_jaar"] = df.index.year
df.loc[df.index.month < 4, "hydr_jaar"] -= 1

df.groupby("hydr_jaar")["NOV"].sum().to_csv("data/neerslagoverschot.csv", header=True)
