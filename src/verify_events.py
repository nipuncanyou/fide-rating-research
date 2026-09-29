"""Keep only events whose games show up in FIDE's classical game counts.

TWIC tags every game with the player's classical rating, even in rapid or
Chess960 events, so event names alone let some fast events through. Two
checks remove them.

First, classical events have at most two rounds a day, so an event where the
typical player plays three or more games in a day is dropped.

Second, a game played in month M is rated in FIDE's list for month M+1
(sometimes M+2). For each event, count the share of player-months where
FIDE's classical game count covers the games the player played in that event.
Classical events score near 1, fast events near 0. Events too recent for any
published list are judged on the first check alone.

Reads data/games.csv and data/fide/monthly.csv, writes data/games_verified.csv
and results/event_check.csv.
"""
import csv
import os
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
THRESHOLD = 0.6
MAX_PER_DAY = 2


def next_month(m, k=1):
    y, mo = int(m[:4]), int(m[5:7]) + k
    while mo > 12:
        y, mo = y + 1, mo - 12
    return f"{y}-{mo:02d}"


def main():
    gms = defaultdict(int)
    with open(os.path.join(ROOT, "data", "fide", "monthly.csv")) as f:
        for r in csv.DictReader(f):
            gms[(r["id"], r["month"])] = int(r["games"] or 0)
    last_list = max(m for _, m in gms)

    rows = []
    with open(os.path.join(ROOT, "data", "games.csv")) as f:
        rows = list(csv.DictReader(f))

    played = defaultdict(int)  # (event, player, month) -> games in that event
    for r in rows:
        m = r["date"][:7]
        for p in (r["white"], r["black"]):
            played[(r["event"], p, m)] += 1

    per_day = defaultdict(int)  # (event, player, date) -> games that day
    for r in rows:
        for p in (r["white"], r["black"]):
            per_day[(r["event"], p, r["date"])] += 1
    busiest = defaultdict(dict)  # event -> player -> most games in one day
    for (ev, p, d), n in per_day.items():
        busiest[ev][p] = max(busiest[ev].get(p, 0), n)
    day_rate = {ev: sorted(v.values())[len(v) // 2] for ev, v in busiest.items()}

    ok = defaultdict(lambda: [0, 0])
    for (ev, p, m), n in played.items():
        m1, m2 = next_month(m), next_month(m, 2)
        if m1 > last_list:
            continue  # not rated yet, no evidence either way
        cover = gms[(p, m1)] + (gms[(p, m2)] if m2 <= last_list else 0)
        ok[ev][0] += cover >= n
        ok[ev][1] += 1

    score = {ev: a / b for ev, (a, b) in ok.items() if b}
    kept_events = {ev for ev in day_rate
                   if day_rate[ev] <= MAX_PER_DAY and score.get(ev, 1.0) >= THRESHOLD}
    with open(os.path.join(ROOT, "results", "event_check.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["event", "player_months", "share_covered", "median_max_games_per_day", "kept"])
        for ev in sorted(day_rate, key=lambda e: score.get(e, 2)):
            sc = f"{score[ev]:.3f}" if ev in score else "not rated yet"
            w.writerow([ev, ok[ev][1], sc, day_rate[ev], ev in kept_events])

    kept = [r for r in rows if r["event"] in kept_events]
    with open(os.path.join(ROOT, "data", "games_verified.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(kept)

    hist = defaultdict(int)
    for s in score.values():
        hist[min(9, int(s * 10))] += 1
    print("events by share covered:", {f"{k / 10:.1f}": v for k, v in sorted(hist.items())})
    print(f"kept {len(kept_events)} of {len(score)} events, {len(kept)} of {len(rows)} games")


if __name__ == "__main__":
    main()
