"""Robustness checks for the paper, on the pooled test seasons (2015-16 to 2025-26).

1. Inactivity with controls: refit the return offsets while also controlling
   for age group and each player's recent form before the game, so the
   offsets are not just pre-existing decline or age.
2. Gap scale by rating band: fit the rating-gap scale separately by the
   higher-rated player's rating.
3. Inactivity by rating level: score against FIDE expectation by break
   length for adults, split by rating.
4. Without the pandemic: recompute pooled results leaving out 2019-20 and
   2020-21.

Writes results/robustness.json.
"""
import json
import os
from collections import defaultdict

from backtest import AGE_BUCKETS, add_form, bucket, fide_expected, fit
from simulate import (AGE_GROUPS, GAP_BUCKETS, N_RET, age_group, fit_generic, gap_bucket,
                      load_games, load_lists, ret_index)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_SEASONS = set(range(2015, 2026))
PANDEMIC = {2019, 2020}


def controlled_features(g):
    """White, age buckets, recent form, then one offset per age group x break length."""
    na = len(AGE_BUCKETS)
    x = [1.0] + [0.0] * na + [g["fw"] - g["fb"]] + [0.0] * N_RET
    for side, sign in (("w", 1), ("b", -1)):
        b = bucket(g[side + "a"])
        if b is not None:
            x[1 + b] += sign
        i = ret_index(g[side + "a"], g[side + "gap"], g["lookback"])
        if i is not None:
            x[2 + na + i] += sign
    return x


def main():
    active, gms, k_used, rating_at, flags = load_lists()
    games = load_games(active)
    add_form(games)
    test = [g for g in games if g["season"] in TEST_SEASONS]
    print(f"{len(test)} test-season games")

    # 1. inactivity offsets with and without controls
    from simulate import inactivity_features
    plain = fit_generic(test, inactivity_features, 1 + N_RET)
    ctrl = fit_generic(test, controlled_features, 2 + len(AGE_BUCKETS) + N_RET)
    base = 2 + len(AGE_BUCKETS)
    offsets = []
    for a, (_, _, alab) in enumerate(AGE_GROUPS):
        for b, (_, _, blab) in enumerate(GAP_BUCKETS[1:]):
            i = a * (len(GAP_BUCKETS) - 1) + b
            offsets.append({"age": alab, "gap": blab, "plain": plain[1 + i], "controlled": ctrl[base + i]})
    print("inactivity offsets, plain vs controlled for age and form:")
    for o in offsets:
        print(f"  {o['age']:>12} {o['gap']:>12}: {o['plain']:+7.1f} {o['controlled']:+7.1f}")

    # 2. gap scale by the higher-rated player's rating
    bands = [(0, 1799, "below 1800"), (1800, 2199, "1800-2199"), (2200, 2499, "2200-2499"), (2500, 9999, "2500+")]
    gap_by_band = []
    for lo, hi, lab in bands:
        sub = [g for g in test if lo <= max(g["we"], g["be"]) <= hi]
        theta = fit(sub, (True, False, False, True))
        gap_by_band.append({"band": lab, "games": len(sub), "gap_scale": 1 + theta[0], "white": theta[1]})
        print(f"gap scale, higher-rated player {lab:>10}: {1 + theta[0]:.3f} ({len(sub)} games)")

    # 3. inactivity by rating level, adults only, model-free
    levels = [(0, 1999, "below 2000"), (2000, 2299, "2000-2299"), (2300, 9999, "2300+")]
    cells = defaultdict(lambda: [0, 0.0])
    for g in test:
        e = fide_expected(g["we"] - g["be"])
        for side, r, rating in (("w", g["s"] - e, g["we"]), ("b", e - g["s"], g["be"])):
            age = g[side + "a"]
            if age is None or age < 21:
                continue
            b = gap_bucket(g[side + "gap"], g["lookback"])
            if b is None:
                continue
            lv = next(l for lo, hi, l in levels if lo <= rating <= hi)
            cells[(lv, b)][0] += 1
            cells[(lv, b)][1] += r
    by_level = [{"level": lv, "gap": GAP_BUCKETS[b][2], "player_games": n, "score_vs_expected_pct": 100 * s / n}
                for (lv, b), (n, s) in sorted(cells.items(), key=lambda kv: ([l for _, _, l in levels].index(kv[0][0]), kv[0][1]))]
    print("adults, score vs expectation by rating level and break:")
    for r in by_level:
        print(f"  {r['level']:>10} {r['gap']:>20}: {r['score_vs_expected_pct']:+6.2f}% ({r['player_games']})")

    # 4. pooled results without the pandemic seasons
    summ = json.load(open(os.path.join(ROOT, "results", "summary.json")))
    sim = json.load(open(os.path.join(ROOT, "results", "simulation.json")))
    no_pandemic = {"backtest": {}, "simulation": {}}
    rows = [r for r in summ["walk_forward"]["seasons"] if r["season"] not in PANDEMIC]
    n = sum(r["games"] for r in rows)
    for name in rows[0]["models"]:
        ll = sum(r["models"][name]["log_loss"] * r["games"] for r in rows) / n
        br = sum(r["models"][name]["brier"] * r["games"] for r in rows) / n
        d = sum(r["models"][name]["vs_fide"] * r["games"] for r in rows) / n
        no_pandemic["backtest"][name] = {"log_loss": ll, "brier": br, "vs_fide": d}
    per = {int(s): v for s, v in sim["simulation_by_season"].items()}
    counts = {int(s): c for s, c in sim["games_by_season"].items()}
    keep = [s for s in per if s not in PANDEMIC]
    n = sum(counts[s] for s in keep)
    for name in per[keep[0]]:
        br = sum(per[s][name]["brier"] * counts[s] for s in keep) / n
        d = sum(per[s][name]["vs_control"] * counts[s] for s in keep) / n
        no_pandemic["simulation"][name] = {"brier": br, "vs_control": d}
    print("without pandemic seasons:")
    for k, v in no_pandemic["backtest"].items():
        print(f"  backtest {k:34} {v['vs_fide']:+.5f}")
    for k, v in no_pandemic["simulation"].items():
        print(f"  replay   {k:36} {v['vs_control']:+.5f}")

    with open(os.path.join(ROOT, "results", "robustness.json"), "w") as f:
        json.dump({"inactivity_controlled": offsets, "gap_scale_by_band": gap_by_band,
                   "inactivity_by_rating_level": by_level, "without_pandemic": no_pandemic}, f, indent=2)


if __name__ == "__main__":
    main()
