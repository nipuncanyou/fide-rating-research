"""Month-by-month simulation of proposed FIDE rule changes, plus the inactivity test.

Difference replay: the official rating in each game's tag already reflects
every rated game in the world, most of which are not in this sample. So each
proposed system keeps, per player, a running difference (delta) from the
official rating. Each month:

  proposed rating  = official rating + delta
  after the month  delta += K_new * (score - E_new) - K_fide * (score - E_fide)

K_fide is the K-factor FIDE actually used, read from the monthly list. Under
FIDE's own rules delta stays at zero, which is the control.

Walk-forward: the replay runs from October 2014. Season 2014-15 is warm-up
with no rule changes. From 2015-16 on, each season's rules use parameters
fitted on the season before it (the table divisor comes from backtest.py's
walk-forward fits), and each season is scored as a test season.
"""
import bisect
import csv
import json
import math
import os
from collections import defaultdict

from backtest import C, fide_expected, logistic, losses, paired_ci, season_of, solve

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIRST_SEASON, LAST_SEASON = 2014, 2025
FIRST_LIST = "2012-09"  # FIDE lists are monthly from here

GAP_BUCKETS = [(1, 2, "active (0-2 months)"), (3, 5, "3-5 months"), (6, 11, "6-11 months"),
               (12, 23, "1-2 years"), (24, 47, "2-4 years"), (48, 10 ** 6, "4+ years")]
AGE_GROUPS = [(0, 20, "under 21"), (21, 49, "21-49"), (50, 200, "50 and over")]
K_MIN, K_MAX = 10, 40
RD_MIN, RD_MAX = math.sqrt(K_MIN / C), math.sqrt(K_MAX / C)
MEAN_INFO = 0.21  # average E(1-E) per game, for the uncertainty update


def midx(m):
    return int(m[:4]) * 12 + int(m[5:7]) - 1


def load_lists():
    active = defaultdict(list)  # player -> sorted month indexes with rated classical games
    gms, k_used, rating_at, flags = {}, {}, {}, {}
    with open(os.path.join(ROOT, "data", "fide", "monthly.csv")) as f:
        for r in csv.DictReader(f):
            i = midx(r["month"])
            n = int(r["games"] or 0)
            if n > 0:
                active[r["id"]].append(i - 1)  # list M covers games played in M-1
                gms[(r["id"], i)] = n
            if r["k"].isdigit():
                k_used[(r["id"], i)] = int(r["k"])
            rating_at[(r["id"], i)] = int(r["rating"])
            flags[(r["id"], i)] = r["flag"]
    for v in active.values():
        v.sort()
    return active, gms, k_used, rating_at, flags


def load_games(active):
    first = midx(FIRST_LIST)
    games = []
    with open(os.path.join(ROOT, "data", "games_verified.csv")) as f:
        for r in csv.DictReader(f):
            m = r["date"][:7]
            season = season_of(m)
            if not FIRST_SEASON <= season <= LAST_SEASON:
                continue
            y = int(m[:4])
            g = {"date": r["date"], "month": m, "mi": midx(m), "season": season,
                 "w": r["white"], "b": r["black"],
                 "we": int(r["white_elo"]), "be": int(r["black_elo"]), "s": float(r["score"]),
                 "wa": y - int(r["white_by"]) if r["white_by"] else None,
                 "ba": y - int(r["black_by"]) if r["black_by"] else None,
                 "event": m + "|" + r["event"]}
            g["lookback"] = g["mi"] - first
            for side in ("w", "b"):
                acts = active.get(g[side], [])
                j = bisect.bisect_right(acts, g["mi"] - 1)
                g[side + "gap"] = g["mi"] - acts[j - 1] if j else None
            games.append(g)
    games.sort(key=lambda g: g["date"])
    return games


def gap_bucket(gap, lookback):
    if gap is None:
        # no rated game since the lists began: a 4+ year break only if the lists reach back that far
        return len(GAP_BUCKETS) - 1 if lookback >= 48 else None
    for i, (lo, hi, _) in enumerate(GAP_BUCKETS):
        if lo <= gap <= hi:
            return i
    return 0


def age_group(age):
    if age is None:
        return None
    for i, (lo, hi, _) in enumerate(AGE_GROUPS):
        if lo <= age <= hi:
            return i
    return None


# ---------- inactivity: does coming back from a break cost rating points? ----------

N_RET = (len(GAP_BUCKETS) - 1) * len(AGE_GROUPS)


def ret_index(age, gap, lookback):
    b, a = gap_bucket(gap, lookback), age_group(age)
    if not b or a is None:
        return None
    return a * (len(GAP_BUCKETS) - 1) + (b - 1)


def inactivity_features(g):
    x = [1.0] + [0.0] * N_RET  # white advantage, then one offset per age group x break length
    for side, sign in (("w", 1), ("b", -1)):
        i = ret_index(g[side + "a"], g[side + "gap"], g["lookback"])
        if i is not None:
            x[1 + i] += sign
    return x


def fit_generic(games, featfn, k, ridge=1e-3):
    theta = [0.0] * k
    for _ in range(10):
        grad = [ridge * t for t in theta]
        hess = [[ridge if i == j else 0.0 for j in range(k)] for i in range(k)]
        for g in games:
            x = featfn(g)
            nz = [i for i in range(k) if x[i]]
            d = max(-400, min(400, g["we"] - g["be"])) + sum(theta[i] * x[i] for i in nz)
            p = logistic(d)
            e, w = (p - g["s"]) * C, p * (1 - p) * C * C
            for i in nz:
                grad[i] += e * x[i]
                for j in nz:
                    hess[i][j] += w * x[i] * x[j]
        step = solve(hess, grad)
        theta = [t - s for t, s in zip(theta, step)]
        if max(abs(s) for s in step) < 1e-3:
            break
    return theta


def returns_from(theta):
    """Offsets applied on return, for breaks of a year or more only."""
    return {i: theta[1 + i] for i in range(N_RET) if (i % (len(GAP_BUCKETS) - 1)) + 1 >= 3}


# ---------- rule systems ----------

def expect_for(divisor):
    def e(d):
        d = max(-divisor, min(divisor, d))  # the 400-point rule stretches with the table
        return 1 / (1 + 10 ** (-d / divisor))
    return e


def k_age(age, k_fide):
    if age is None:
        return k_fide
    if age <= 17:
        return 40
    if age <= 20:
        return max(k_fide, 30)
    return k_fide


def simulate(games, k_used, gms, params, table=False, age_k=False, returns=False, unc_c=None,
             start_season=FIRST_SEASON + 1):
    """params: season -> {"divisor", "returns"}; rules act from start_season on.

    unc_c: monthly growth of rating uncertainty (Elo points). When set, each
    player's K is q * RD^2, clamped to FIDE's range of 10 to 40.
    """
    delta = defaultdict(float)
    rd, last_mi = {}, {}
    preds = []
    by_month = defaultdict(list)
    for g in games:
        by_month[g["mi"]].append(g)
    adjusted = 0
    for mi in sorted(by_month):
        month = by_month[mi]
        season = month[0]["season"]
        on = season >= start_season
        p_season = params.get(season, {})
        expect = expect_for(p_season.get("divisor", 400)) if (on and table) else fide_expected
        ret = p_season.get("returns", {}) if (on and returns) else {}
        k_new = {}
        seen = set()
        for g in month:
            for side in ("w", "b"):
                p = g[side]
                if p in seen:
                    continue
                seen.add(p)
                if ret:
                    gap = g[side + "gap"]
                    if gap is None or gap >= 12:
                        i = ret_index(g[side + "a"], gap, g["lookback"])
                        if i is not None and i in ret:
                            delta[p] += ret[i]
                            adjusted += 1
                if unc_c is not None:
                    kf = k_used.get((p, mi), 20)
                    if p not in rd:
                        rd[p] = math.sqrt(kf / C)
                    else:
                        rd[p] = min(RD_MAX, math.sqrt(rd[p] ** 2 + unc_c ** 2 * (mi - last_mi[p])))
                    k_new[p] = max(K_MIN, min(K_MAX, C * rd[p] ** 2))
        change = defaultdict(float)
        for g in month:
            rw, rb = g["we"] + delta[g["w"]], g["be"] + delta[g["b"]]
            e_new = expect(rw - rb)
            e_old = fide_expected(g["we"] - g["be"])
            preds.append(e_new)
            for side, s, en, eo, r_off in (("w", g["s"], e_new, e_old, g["we"]),
                                           ("b", 1 - g["s"], 1 - e_new, 1 - e_old, g["be"])):
                p = g[side]
                kf = k_used.get((p, mi), 20 if r_off < 2400 else 10)
                if not on:
                    kn = kf
                elif unc_c is not None:
                    kn = k_new[p]
                elif age_k:
                    kn = k_age(g[side + "a"], kf)
                else:
                    kn = kf
                change[p] += kn * (s - en) - kf * (s - eo)
        for p, c in change.items():
            delta[p] += c
        if unc_c is not None:
            for p in seen:
                # every rated game since the last update counts, including games outside this sample
                start = last_mi.get(p, mi - 1) + 1
                n = sum(gms.get((p, i + 1), 0) for i in range(start, mi + 1)) or 1
                rd[p] = max(RD_MIN, 1 / math.sqrt(1 / rd[p] ** 2 + C * C * MEAN_INFO * n))
                last_mi[p] = mi
    return preds, delta, adjusted


def evaluate(preds, games, seasons):
    """Per-season and pooled losses over the test seasons."""
    per, ll_all, br_all, ev_all = defaultdict(lambda: [0.0, 0.0, 0]), [], [], []
    for p, g in zip(preds, games):
        if g["season"] not in seasons:
            continue
        ll, br = losses(p, g["s"])
        c = per[g["season"]]
        c[0] += ll
        c[1] += br
        c[2] += 1
        ll_all.append(ll)
        br_all.append(br)
        ev_all.append(g["event"])
    return per, ll_all, br_all, ev_all


def main():
    active, gms, k_used, rating_at, flags = load_lists()
    games = load_games(active)
    by_season = defaultdict(list)
    for g in games:
        by_season[g["season"]].append(g)
    test_seasons = list(range(FIRST_SEASON + 1, LAST_SEASON + 1))
    print("games per season:", {s: len(v) for s, v in sorted(by_season.items())})

    # 1. inactivity diagnostics, pooled over the test seasons
    cells = defaultdict(lambda: [0, 0.0])
    for g in games:
        if g["season"] not in test_seasons:
            continue
        e = fide_expected(g["we"] - g["be"])
        for side, r in (("w", g["s"] - e), ("b", e - g["s"])):
            a, b = age_group(g[side + "a"]), gap_bucket(g[side + "gap"], g["lookback"])
            if a is None or b is None:
                continue
            cells[(a, b)][0] += 1
            cells[(a, b)][1] += r
    residuals = [{"age": AGE_GROUPS[a][2], "gap": GAP_BUCKETS[b][2], "player_games": n,
                  "score_vs_expected_pct": 100 * s / n}
                 for (a, b), (n, s) in sorted(cells.items())]

    # offsets fitted on all seasons together: the sizes proposed as a rule
    theta_all = fit_generic(games, inactivity_features, 1 + N_RET)
    offsets = []
    for a, (_, _, alab) in enumerate(AGE_GROUPS):
        for b, (_, _, blab) in enumerate(GAP_BUCKETS[1:]):
            i = a * (len(GAP_BUCKETS) - 1) + b
            offsets.append({"age": alab, "gap": blab, "elo_points": theta_all[1 + i],
                            "player_games": cells[(a, b + 1)][0]})

    # 2. per-season parameters, each fitted on the season before
    walk = {r["season"]: r for r in json.load(open(os.path.join(ROOT, "results", "summary.json")))
            ["walk_forward"]["seasons"]}
    params = {}
    for s in test_seasons:
        theta = fit_generic(by_season[s - 1], inactivity_features, 1 + N_RET)
        params[s] = {"divisor": round(400 / walk[s]["gap_scale"] / 10) * 10 if s in walk else 400,
                     "returns": returns_from(theta)}
        print(f"season {s}: divisor {params[s]['divisor']}", flush=True)

    # uncertainty growth: chosen on the warm-up season, with the rule switched on from the start
    warm = by_season[FIRST_SEASON]
    best_c = None
    for c in (0.0, 5.0, 10.0, 20.0, 40.0):
        pr, _, _ = simulate(warm, k_used, gms, {}, unc_c=c, start_season=FIRST_SEASON)
        ll = sum(losses(p, g["s"])[0] for p, g in zip(pr, warm)) / len(warm)
        print(f"  uncertainty growth {c}: warm-up log loss {ll:.5f}")
        if best_c is None or ll < best_c[0]:
            best_c = (ll, c)
    unc_c = best_c[1]

    systems = {
        "FIDE rules (control)": dict(),
        "Stretched table": dict(table=True),
        "K-factor by age": dict(age_k=True),
        "Inactivity adjustment": dict(returns=True),
        "Uncertainty K-factor": dict(unc_c=unc_c),
        "Table + inactivity adjustment": dict(table=True, returns=True),
        "Table + inactivity + uncertainty K": dict(table=True, returns=True, unc_c=unc_c),
    }
    results, by_season_out, deltas = [], defaultdict(dict), {}
    base_ll = None
    for name, kw in systems.items():
        preds, delta, adjusted = simulate(games, k_used, gms, params, **kw)
        per, ll, br, ev = evaluate(preds, games, test_seasons)
        if base_ll is None:
            base_ll, base_per = ll, per
        d, lo, hi = paired_ci(base_ll, ll, ev)
        better = 0
        for s, (sl, sb, n) in sorted(per.items()):
            diff = (sl - base_per[s][0]) / n
            by_season_out[s][name] = {"log_loss": sl / n, "brier": sb / n, "vs_control": diff}
            better += diff < 0
        results.append({"system": name, "log_loss": sum(ll) / len(ll), "brier": sum(br) / len(br),
                        "vs_control": d, "ci_low": lo, "ci_high": hi,
                        "returns_adjusted": adjusted, "seasons_better": better})
        deltas[name] = delta
        print(f"  {name:36} {d:+.5f} [{lo:+.5f}, {hi:+.5f}] better in {better}/{len(per)} seasons",
              flush=True)

    last_age = {}
    for g in games:
        for side in ("w", "b"):
            last_age[g[side]] = g[side + "a"]
    shift = []
    for name in list(systems)[1:]:
        by = defaultdict(list)
        for p, dv in deltas[name].items():
            a = age_group(last_age.get(p))
            if a is not None:
                by[a].append(dv)
        shift.append({"system": name, **{AGE_GROUPS[a][2]: sum(v) / len(v) for a, v in sorted(by.items())}})

    # 3. who is actually sitting on a rating: 2700+ on the September 2026 list
    sep26 = midx("2026-09")
    top = []
    for (p, i), r in rating_at.items():
        if i == sep26 and r >= 2700:
            last = active.get(p, [])
            top.append({"id": p, "rating": r, "flag": flags[(p, i)],
                        "classical_games_last_12_months": sum(gms.get((p, j), 0) for j in range(sep26 - 11, sep26 + 1)),
                        "months_since_last_game": (sep26 - last[-1]) if last else None})
    want = {t["id"] for t in top}
    names, births = {}, {}
    with open(os.path.join(ROOT, "data", "fide", "players_list_foa.txt"), encoding="latin-1") as f:
        header = f.readline()
        b0 = header.index("B-day")
        for line in f:
            fid = line[:15].strip()
            if fid in want:
                names[fid] = line[15:76].strip()
                if line[b0:b0 + 4].strip().isdigit():
                    births[fid] = int(line[b0:b0 + 4])
    for t in top:
        t["name"] = names.get(t["id"], t["id"])
        t["age"] = 2026 - births[t["id"]] if t["id"] in births else None
        gap = t["months_since_last_game"]
        i = ret_index(t["age"], gap, sep26 - midx(FIRST_LIST))
        t["return_adjustment"] = theta_all[1 + i] if i is not None and (gap is None or gap >= 12) else 0.0
    top.sort(key=lambda t: -t["rating"])

    naka = [{"list": f"{i // 12}-{i % 12 + 1:02d}", "rating": rating_at[("2016192", i)],
             "games": gms.get(("2016192", i), 0)}
            for i in range(midx("2024-01"), sep26 + 1) if ("2016192", i) in rating_at]

    out = {"divisor": params[LAST_SEASON]["divisor"],
           "divisor_by_season": {s: p["divisor"] for s, p in params.items()},
           "uncertainty_growth": unc_c, "test_seasons": test_seasons,
           "inactivity_residuals": residuals, "inactivity_offsets": offsets,
           "white_advantage": theta_all[0], "simulation": results,
           "simulation_by_season": by_season_out, "rating_shift_by_age": shift,
           "top_2700": top, "nakamura_lists": naka,
           "games_by_season": {s: len(v) for s, v in sorted(by_season.items())}}
    with open(os.path.join(ROOT, "results", "simulation.json"), "w") as f:
        json.dump(out, f, indent=2)

    print("\nScore vs FIDE expectation by age and time since last rated game (test seasons):")
    for r in residuals:
        print(f"  {r['age']:>12} | {r['gap']:>20}: {r['score_vs_expected_pct']:+6.2f}%  ({r['player_games']})")
    print("\nOffsets on return, Elo points (all seasons):")
    for o in offsets:
        print(f"  {o['age']:>12} | {o['gap']:>12}: {o['elo_points']:+6.1f}  (n={o['player_games']})")
    print(f"\nSimulation over {len(test_seasons)} test seasons (uncertainty growth {unc_c}):")
    for r in results:
        print(f"  {r['system']:36} ll {r['log_loss']:.5f} brier {r['brier']:.5f} "
              f"{r['vs_control']:+.5f} [{r['ci_low']:+.5f}, {r['ci_high']:+.5f}] "
              f"better {r['seasons_better']}/{len(test_seasons)} returns {r['returns_adjusted']}")
    print("\nMean rating change by the end, by age group:")
    for s in shift:
        print("  ", {k: (round(v, 1) if isinstance(v, float) else v) for k, v in s.items()})
    print("\nInactive 2700+ players, September 2026 list:")
    for t in top:
        if t["flag"] == "i" or t["id"] == "2016192":
            print(f"  {t['name'][:28]:28} {t['rating']} flag={t['flag'] or '-'} games(12m)={t['classical_games_last_12_months']} "
                  f"since={t['months_since_last_game']} age={t['age']} adj={t['return_adjustment']:+.0f}")


if __name__ == "__main__":
    main()
