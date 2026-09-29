# Where FIDE Ratings Miss

An out-of-sample test of how well published FIDE classical ratings predict game results, and which corrections improve them.

1,510,378 classical games from The Week in Chess, October 2014 to September 2026, checked against 169 FIDE monthly lists (September 2012 onward) so that rapid, blitz and Chess960 events are removed. Every correction is fitted on one season (October to September) and tested on the next, for 11 test seasons. Rule changes are then replayed month by month against FIDE's current rules, using the K-factor FIDE actually applied.

Main findings, pooled over the 11 test seasons:

- FIDE's conversion table overstates rating gaps by about 20% (fitted scale 0.71 to 0.87 by season; roughly a 500-point table instead of 400).
- Juniors play above their ratings (+51 Elo at ages 12 to 13 in 2024-25, peaking near +110 after the pandemic); players over 65 play about 70 below.
- Adults returning from a break play below their rating, more so the older they are and the longer the break: about -90 for ages 21 to 49 and -174 for over 50 after four or more years. Juniors returning play above it.
- Recent overperformance keeps predicting results, so ratings update too slowly.
- White is worth about 33 Elo points in every season.
- Every correction beat the published ratings in all 11 test seasons; together they cut Brier score by 3.9%.
- Replayed month by month, a stretched table plus an age-aware inactivity adjustment beat FIDE's current rules in 11 of 11 seasons (1.7% less error). A higher K-factor for juniors and an uncertainty-based K-factor were worse in every season.
- Glicko-2 and FIDE's Elo rule, rebuilt on the same games, were not consistently different (Glicko-2 better in 5 of 11 seasons).

By Nipun Anand.

## Run

Plain Python 3, no packages.

```
mkdir -p data/twic data/fide
curl -sf -o data/fide/players_list.zip https://ratings.fide.com/download/players_list.zip && unzip -oq data/fide/players_list.zip -d data/fide
python3 src/fetch_twic.py     # data/twic (TWIC issues 1038 to 1664, about 1.3 GB)
python3 src/prepare.py        # data/games.csv
python3 src/fetch_lists.py    # data/fide/monthly.csv (FIDE lists from Sep 2012)
python3 src/verify_events.py  # data/games_verified.csv
python3 src/backtest.py       # results/summary.json
python3 src/simulate.py       # results/simulation.json
python3 src/build_report.py   # report/index.html
```
