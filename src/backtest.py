"""Out-of-sample backtest of FIDE ratings against candidate improvements.

A season runs October to September. The detailed analysis trains on the
2024-25 season and tests on 2025-26. The walk-forward analysis repeats this
for every season from 2015-16 to 2025-26, each time fitting only on the
season before, and pools the eleven test seasons.

Every model predicts White's expected score before the game. Losses are
computed on the actual score (1, 0.5, 0).
"""
import csv
import json
import math
import os
import random
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GAMES = os.path.join(ROOT, "data", "games_verified.csv")
OUT = os.path.join(ROOT, "results")

FIRST_SEASON, LAST_SEASON = 2014, 2025  # season 2014 = Oct 2014 to Sep 2015
TRAIN_SEASON, TEST_SEASON = 2024, 2025
C = math.log(10) / 400
AGE_BUCKETS = [(0, 11, "11 and under"), (12, 13, "12-13"), (14, 15, "14-15"),
               (16, 17, "16-17"), (18, 20, "18-20"), (50, 64, "50-64"),
               (65, 200, "65+")]  # 21-49 is the reference group


def fide_expected(d):
    d = max(-400, min(400, d))  # FIDE's 400-point rule
    return 1 / (1 + 10 ** (-d / 400))


def logistic(d):
    return 1 / (1 + math.exp(-C * d))


def season_of(m):
    y, mo = int(m[:4]), int(m[5:7])
    return y if mo >= 10 else y - 1


def load():
    rows = []
    with open(GAMES) as f:
        for r in csv.DictReader(f):
            m = r["date"][:7]
            season = season_of(m)
            if not FIRST_SEASON <= season <= LAST_SEASON:
                continue
            y = int(r["date"][:4])
            rows.append({
                "date": r["date"], "month": m, "w": r["white"], "b": r["black"],
                "we": int(r["white_elo"]), "be": int(r["black_elo"]),
                "s": float(r["score"]),
                "wa": y - int(r["white_by"]) if r["white_by"] else None,
                "ba": y - int(r["black_by"]) if r["black_by"] else None,
                "season": season, "test": season == TEST_SEASON, "event": m + "|" + r["event"],
            })
    rows.sort(key=lambda g: g["date"])
    return rows


def bucket(age):
    if age is None:
        return None
    for i, (lo, hi, _) in enumerate(AGE_BUCKETS):
        if lo <= age <= hi:
            return i
    return None


# ---------- recent form: causal, per-player running residual vs FIDE ----------

def add_form(games, decay=0.9, shrink=4.0):
    s, n = defaultdict(float), defaultdict(float)
    i = 0
    while i < len(games):
        j = i
        while j < len(games) and games[j]["date"] == games[i]["date"]:
            j += 1
        day = games[i:j]
        for g in day:  # use only information from earlier dates
            g["fw"] = s[g["w"]] / (n[g["w"]] + shrink)
            g["fb"] = s[g["b"]] / (n[g["b"]] + shrink)
        for g in day:
            r = g["s"] - fide_expected(g["we"] - g["be"])
            for p, sign in ((g["w"], 1), (g["b"], -1)):
                s[p] = decay * s[p] + sign * r
                n[p] = decay * n[p] + 1
        i = j


# ---------- adjustment models: FIDE rating difference + fitted offsets ----------

def features(g, use_colour, use_age, use_form, use_scale=False):
    x = []
    if use_scale:
        x.append(float(base_diff(g)))  # theta here stretches or shrinks the rating gap
    if use_colour:
        x.append(1.0)
    if use_age:
        v = [0.0] * len(AGE_BUCKETS)
        bw, bb = bucket(g["wa"]), bucket(g["ba"])
        if bw is not None:
            v[bw] += 1
        if bb is not None:
            v[bb] -= 1
        x.extend(v)
    if use_form:
        x.append(g["fw"] - g["fb"])
    return x


def base_diff(g):
    return max(-400, min(400, g["we"] - g["be"]))


def solve(a, b):
    n = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(m[r][c]))
        m[c], m[p] = m[p], m[c]
        if abs(m[c][c]) < 1e-12:
            continue
        for r in range(n):
            if r != c:
                f = m[r][c] / m[c][c]
                for k in range(c, n + 1):
                    m[r][k] -= f * m[c][k]
    return [m[i][n] / m[i][i] if abs(m[i][i]) > 1e-12 else 0.0 for i in range(n)]


def fit(train, spec, ridge=1e-3):  # ~120 games of prior weight at zero
    k = len(features(train[0], *spec))
    theta = [0.0] * k
    for _ in range(12):  # Newton steps on fractional log loss
        grad = [ridge * t for t in theta]
        hess = [[ridge if i == j else 0.0 for j in range(k)] for i in range(k)]
        for g in train:
            x = features(g, *spec)
            d = base_diff(g) + sum(t * xi for t, xi in zip(theta, x))
            p = logistic(d)
            e = (p - g["s"]) * C
            w = p * (1 - p) * C * C
            for i in range(k):
                if x[i]:
                    grad[i] += e * x[i]
                    for j in range(k):
                        if x[j]:
                            hess[i][j] += w * x[i] * x[j]
        step = solve(hess, grad)
        theta = [t - s for t, s in zip(theta, step)]
        if max(abs(s) for s in step) < 1e-4:
            break
    return theta


def predict_adjusted(g, theta, spec):
    x = features(g, *spec)
    return logistic(base_diff(g) + sum(t * xi for t, xi in zip(theta, x)))


# ---------- from-scratch systems on the TWIC games only ----------

def run_elo(games):
    """FIDE's update rule, monthly, seeded from each player's first published rating."""
    rating, peak = {}, {}
    preds = []
    by_month = defaultdict(list)
    for g in games:
        by_month[g["month"]].append(g)
    for m in sorted(by_month):
        delta = defaultdict(float)
        for g in by_month[m]:
            for p, e in ((g["w"], g["we"]), (g["b"], g["be"])):
                if p not in rating:
                    rating[p] = peak[p] = float(e)
            rw, rb = rating[g["w"]], rating[g["b"]]
            ew = fide_expected(rw - rb)
            preds.append(ew)
            for p, age, r, sc, ex in ((g["w"], g["wa"], rw, g["s"], ew),
                                      (g["b"], g["ba"], rb, 1 - g["s"], 1 - ew)):
                if age is not None and age < 18 and r < 2300:
                    k = 40
                elif peak[p] >= 2400:
                    k = 10
                else:
                    k = 20
                delta[p] += k * (sc - ex)
        for p, dv in delta.items():
            rating[p] = max(1400.0, rating[p] + dv)
            peak[p] = max(peak[p], rating[p])
    return preds


def run_glicko2(games, init_rd=150.0, tau=0.5, init_vol=0.06):
    q = 173.7178
    mu, phi, vol = {}, {}, {}

    def g_(ph):
        return 1 / math.sqrt(1 + 3 * ph * ph / math.pi ** 2)

    preds = []
    by_month = defaultdict(list)
    for g in games:
        by_month[g["month"]].append(g)
    for m in sorted(by_month):
        results = defaultdict(list)
        for g in by_month[m]:
            for p, e in ((g["w"], g["we"]), (g["b"], g["be"])):
                if p not in mu:
                    mu[p], phi[p], vol[p] = (e - 1500) / q, init_rd / q, init_vol
            a, b = g["w"], g["b"]
            comb = math.sqrt(phi[a] ** 2 + phi[b] ** 2)
            preds.append(1 / (1 + math.exp(-g_(comb) * (mu[a] - mu[b]))))
            results[a].append((mu[b], phi[b], g["s"]))
            results[b].append((mu[a], phi[a], 1 - g["s"]))
        new = {}
        for p, res in results.items():
            v_inv, dsum = 0.0, 0.0
            for mj, pj, s in res:
                gj = g_(pj)
                e = 1 / (1 + math.exp(-gj * (mu[p] - mj)))
                v_inv += gj * gj * e * (1 - e)
                dsum += gj * (s - e)
            v = 1 / v_inv
            delta = v * dsum
            a = math.log(vol[p] ** 2)
            ph2 = phi[p] ** 2

            def f(x):
                ex = math.exp(x)
                return (ex * (delta ** 2 - ph2 - v - ex) / (2 * (ph2 + v + ex) ** 2)
                        - (x - a) / tau ** 2)

            A = a
            if delta ** 2 > ph2 + v:
                B = math.log(delta ** 2 - ph2 - v)
            else:
                k = 1
                while f(a - k * tau) < 0:
                    k += 1
                B = a - k * tau
            fa, fb = f(A), f(B)
            for _ in range(100):
                if abs(B - A) < 1e-6:
                    break
                Cx = A + (A - B) * fa / (fb - fa)
                fc = f(Cx)
                if fc * fb <= 0:
                    A, fa = B, fb
                else:
                    fa /= 2
                B, fb = Cx, fc
            nv = math.exp(A / 2)
            pstar = math.sqrt(ph2 + nv * nv)
            nphi = 1 / math.sqrt(1 / pstar ** 2 + 1 / v)
            new[p] = (mu[p] + nphi ** 2 * dsum, nphi, nv)
        for p in mu:
            if p in new:
                mu[p], phi[p], vol[p] = new[p]
            else:
                phi[p] = min(350 / q, math.sqrt(phi[p] ** 2 + vol[p] ** 2))
    return preds


# ---------- scoring ----------

def losses(p, s):
    p = min(max(p, 1e-6), 1 - 1e-6)
    return -(s * math.log(p) + (1 - s) * math.log(1 - p)), (p - s) ** 2


def score(preds, games):
    ll, br = [], []
    for p, g in zip(preds, games):
        a, b = losses(p, g["s"])
        ll.append(a)
        br.append(b)
    return ll, br


def paired_ci(a, b, events, reps=1000, seed=7):
    """Bootstrap CI for mean(b - a), resampling whole events (games in one event are correlated)."""
    rnd = random.Random(seed)
    groups = defaultdict(lambda: [0.0, 0])
    for x, y, ev in zip(a, b, events):
        groups[ev][0] += y - x
        groups[ev][1] += 1
    groups = list(groups.values())
    n = len(groups)
    means = []
    for _ in range(reps):
        tot = cnt = 0.0
        for _ in range(n):
            s, c = groups[rnd.randrange(n)]
            tot += s
            cnt += c
        means.append(tot / cnt)
    means.sort()
    total = sum(s for s, _ in groups) / sum(c for _, c in groups)
    return total, means[int(0.025 * reps)], means[int(0.975 * reps)]


def walk_forward(games, specs, elo_all, gl, idx):
    """Fit on each season, test on the next; pool all test seasons."""
    by_season = defaultdict(list)
    for g in games:
        by_season[g["season"]].append(g)
    seasons, pooled = [], defaultdict(lambda: [[], []])
    pooled_events = []
    for s in range(FIRST_SEASON + 1, LAST_SEASON + 1):
        tr, te = by_season[s - 1], by_season[s]
        if len(tr) < 1000 or len(te) < 1000:
            continue
        preds = {"FIDE as published": [fide_expected(g["we"] - g["be"]) for g in te]}
        params = {}
        for name, spec in specs.items():
            theta = fit(tr, spec)
            params[name] = theta
            preds[name] = [predict_adjusted(g, theta, spec) for g in te]
        preds["Elo rebuilt on this data"] = [elo_all[idx[id(g)]] for g in te]
        preds["Glicko-2 rebuilt on this data"] = [gl[idx[id(g)]] for g in te]
        base_ll = score(preds["FIDE as published"], te)[0]
        row = {"season": s, "games": len(te), "models": {},
               "gap_scale": 1 + params["FIDE + white + gap scale"][0],
               "white": params["FIDE + white advantage"][0],
               "age_12_13": params["FIDE + white + age"][1 + 1]}
        for name, p in preds.items():
            ll, br = score(p, te)
            row["models"][name] = {"log_loss": sum(ll) / len(ll), "brier": sum(br) / len(br),
                                   "vs_fide": (sum(ll) - sum(base_ll)) / len(ll)}
            pooled[name][0].extend(ll)
            pooled[name][1].extend(br)
        pooled_events.extend(g["event"] for g in te)
        seasons.append(row)
    base = pooled["FIDE as published"][0]
    out = {}
    for name, (ll, br) in pooled.items():
        d, lo, hi = paired_ci(base, ll, pooled_events)
        out[name] = {"log_loss": sum(ll) / len(ll), "brier": sum(br) / len(br),
                     "vs_fide": d, "ci_low": lo, "ci_high": hi,
                     "seasons_better": sum(r["models"][name]["vs_fide"] < 0 for r in seasons)}
    return {"seasons": seasons, "pooled": out, "test_games": len(base)}


def main():
    os.makedirs(OUT, exist_ok=True)
    games = load()
    add_form(games)
    train = [g for g in games if g["season"] == TRAIN_SEASON]
    test = [g for g in games if g["season"] == TEST_SEASON]
    print(f"train {len(train)} games, test {len(test)} games, "
          f"players {len({g['w'] for g in games} | {g['b'] for g in games})}")

    models = {}
    models["FIDE as published"] = [fide_expected(g["we"] - g["be"]) for g in test]

    specs = {
        "FIDE + white advantage": (True, False, False, False),
        "FIDE + white + gap scale": (True, False, False, True),
        "FIDE + white + age": (True, True, False, False),
        "FIDE + white + recent form": (True, False, True, False),
        "FIDE + white + gap + age + form": (True, True, True, True),
    }
    fitted = {}
    for name, spec in specs.items():
        theta = fit(train, spec)
        fitted[name] = theta
        models[name] = [predict_adjusted(g, theta, spec) for g in test]

    idx = {id(g): i for i, g in enumerate(games)}
    elo_all = run_elo(games)
    models["Elo rebuilt on this data"] = [elo_all[idx[id(g)]] for g in test]

    # Glicko-2 settings are chosen on the first season only, which is never a test season
    first = [g for g in games if g["season"] == FIRST_SEASON]
    best = None
    for rd in (100.0, 150.0, 250.0):
        for tau in (0.3, 0.6):
            p = run_glicko2(first, init_rd=rd, tau=tau)
            ll = sum(losses(q, g["s"])[0] for q, g in zip(p, first)) / len(first)
            if best is None or ll < best[0]:
                best = (ll, rd, tau)
    _, rd, tau = best
    gl = run_glicko2(games, init_rd=rd, tau=tau)
    models[f"Glicko-2 rebuilt on this data"] = [gl[idx[id(g)]] for g in test]

    events = [g["event"] for g in test]
    base_ll, base_br = score(models["FIDE as published"], test)
    table = []
    for name, preds in models.items():
        ll, br = score(preds, test)
        mean_d, lo, hi = paired_ci(base_ll, ll, events)
        table.append({
            "model": name, "log_loss": sum(ll) / len(ll), "brier": sum(br) / len(br),
            "vs_fide_log_loss": mean_d, "ci_low": lo, "ci_high": hi,
        })
    walk = walk_forward(games, specs, elo_all, gl, idx)
    elo_ll = score(models["Elo rebuilt on this data"], test)[0]
    gl_ll = score(models["Glicko-2 rebuilt on this data"], test)[0]
    gvse = paired_ci(elo_ll, gl_ll, events)

    # diagnostics on the test year, measured against FIDE as published
    calib = defaultdict(lambda: [0, 0.0, 0.0])
    for g in test:
        d = g["we"] - g["be"]
        b = max(-400, min(400, int(math.floor(d / 50.0)) * 50))
        c = calib[b]
        c[0] += 1
        c[1] += fide_expected(d)
        c[2] += g["s"]
    calibration = [{"diff_from": b, "games": c[0], "expected": c[1] / c[0],
                    "actual": c[2] / c[0]} for b, c in sorted(calib.items()) if c[0] >= 200]

    ages = defaultdict(lambda: [0, 0.0])
    for g in test:
        e = fide_expected(g["we"] - g["be"])
        for age, r in ((g["wa"], g["s"] - e), (g["ba"], e - g["s"])):
            if age is None:
                continue
            lab = next((l for lo, hi, l in AGE_BUCKETS if lo <= age <= hi), "21-49")
            ages[lab][0] += 1
            ages[lab][1] += r
    age_order = [l for _, _, l in AGE_BUCKETS[:5]] + ["21-49"] + [l for _, _, l in AGE_BUCKETS[5:]]
    age_table = [{"age": l, "player_games": ages[l][0],
                  "score_vs_expected_pct": 100 * ages[l][1] / ages[l][0]}
                 for l in age_order if ages[l][0]]

    theta_age = fitted["FIDE + white + age"]
    age_points = {"white advantage": theta_age[0]}
    for (_, _, lab), t in zip(AGE_BUCKETS, theta_age[1:]):
        age_points[lab] = t
    full = fitted["FIDE + white + gap + age + form"]
    full_params = {"gap scale": 1 + full[0], "white advantage": full[1]}
    for (_, _, lab), t in zip(AGE_BUCKETS, full[2:2 + len(AGE_BUCKETS)]):
        full_params[lab] = t
    full_params["form"] = full[-1]
    gap_only = fitted["FIDE + white + gap scale"]

    band = defaultdict(lambda: [0, 0.0])
    for g in test:
        e = fide_expected(g["we"] - g["be"])
        for r, res in ((g["we"], g["s"] - e), (g["be"], e - g["s"])):
            k = min(2700, max(1400, r // 200 * 200))
            band[k][0] += 1
            band[k][1] += res
    band_table = [{"band": f"{k}-{k + 199}" if k < 2700 else "2700+", "player_games": v[0],
                   "score_vs_expected_pct": 100 * v[1] / v[0]} for k, v in sorted(band.items())]

    # calibration from the higher-rated player's side, FIDE vs the gap-scale model
    spec_gap = specs["FIDE + white + gap scale"]
    strong = defaultdict(lambda: [0, 0.0, 0.0, 0.0])
    for g, p_gap in zip(test, models["FIDE + white + gap scale"]):
        d = g["we"] - g["be"]
        if d == 0:
            continue
        flip = d < 0
        b = min(400, abs(d) // 50 * 50)
        c = strong[b]
        c[0] += 1
        c[1] += fide_expected(abs(d))
        c[2] += (1 - p_gap) if flip else p_gap
        c[3] += (1 - g["s"]) if flip else g["s"]
    calibration_stronger = [{"gap_from": b, "games": c[0], "fide": c[1] / c[0],
                             "gap_model": c[2] / c[0], "actual": c[3] / c[0]}
                            for b, c in sorted(strong.items())]

    ratings = sorted([g["we"] for g in test] + [g["be"] for g in test])
    known_age = sum((g["wa"] is not None) + (g["ba"] is not None) for g in test)
    composition = {
        "players": len({g["w"] for g in games} | {g["b"] for g in games}),
        "events": len({g["event"] for g in games}),
        "median_rating": ratings[len(ratings) // 2],
        "p10_rating": ratings[len(ratings) // 10],
        "p90_rating": ratings[len(ratings) * 9 // 10],
        "share_known_age": known_age / (2 * len(test)),
        "draw_rate": sum(g["s"] == 0.5 for g in test) / len(test),
        "white_score": sum(g["s"] for g in test) / len(test),
    }

    summary = {
        "train_games": len(train), "test_games": len(test),
        "composition": composition, "calibration_stronger": calibration_stronger,
        "models": table,
        "glicko_vs_elo_same_data": {"mean": gvse[0], "ci_low": gvse[1], "ci_high": gvse[2]},
        "glicko_params": {"initial_rd": rd, "tau": tau},
        "fitted_elo_points": age_points, "full_model_params": full_params,
        "gap_scale_model": {"gap scale": 1 + gap_only[0], "white advantage": gap_only[1]},
        "form_coefficients": {n: fitted[n][-1] for n in fitted if "form" in n},
        "calibration": calibration, "age_residuals": age_table, "rating_band_residuals": band_table,
        "walk_forward": walk,
    }
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'model':34} {'log loss':>9} {'brier':>8} {'vs FIDE (95% CI)':>28}")
    for r in table:
        print(f"{r['model']:34} {r['log_loss']:9.5f} {r['brier']:8.5f} "
              f"{r['vs_fide_log_loss']:+.5f} [{r['ci_low']:+.5f}, {r['ci_high']:+.5f}]")
    print(f"\nGlicko-2 minus Elo, same data: {gvse[0]:+.5f} [{gvse[1]:+.5f}, {gvse[2]:+.5f}]"
          f"  (RD {rd:.0f}, tau {tau})")
    print("\nFitted offsets in Elo points:", {k: round(v, 1) for k, v in age_points.items()})
    print("Form coefficients:", {k: round(v, 1) for k, v in summary["form_coefficients"].items()})
    print("Gap-scale model:", {k: round(v, 3) for k, v in summary["gap_scale_model"].items()})
    print("Full model:", {k: round(v, 3) for k, v in full_params.items()})
    print("\nScore vs FIDE expectation by age (test year):")
    for r in age_table:
        print(f"  {r['age']:>13}: {r['score_vs_expected_pct']:+6.2f}%  ({r['player_games']} player-games)")
    print("\nBy rating band:")
    for r in band_table:
        print(f"  {r['band']:>10}: {r['score_vs_expected_pct']:+6.2f}%  ({r['player_games']})")
    print("\nWalk-forward, change in log loss vs FIDE by test season:")
    for srow in walk["seasons"]:
        print(f"  {srow['season']}-{(srow['season'] + 1) % 100:02d} ({srow['games']:6d} games): " +
              "  ".join(f"{k.replace('FIDE + ', '')[:18]} {v['vs_fide']:+.4f}" for k, v in srow["models"].items()
                        if k != "FIDE as published") + f"  gap scale {srow['gap_scale']:.3f}")
    print("Pooled:", {k: f"{v['vs_fide']:+.5f} [{v['ci_low']:+.5f}, {v['ci_high']:+.5f}]"
                      for k, v in walk["pooled"].items()})
    print("\nCalibration (rating diff bin: expected vs actual):")
    for r in calibration:
        print(f"  {r['diff_from']:+5d}: {r['expected']:.3f} vs {r['actual']:.3f}  ({r['games']})")


if __name__ == "__main__":
    main()
