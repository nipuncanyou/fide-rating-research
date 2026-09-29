"""Generate LaTeX macros, tables and plot data for the paper from the results files.

Every number in paper/main.tex comes from here, so the paper can be rebuilt
after any rerun without retyping results. Writes paper/generated/*.tex.
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "paper", "generated")

S = json.load(open(os.path.join(ROOT, "results", "summary.json")))
M = json.load(open(os.path.join(ROOT, "results", "simulation.json")))
R = json.load(open(os.path.join(ROOT, "results", "robustness.json")))
W = S["walk_forward"]
WS, WP = W["seasons"], W["pooled"]


def n(x):
    return f"{x:,}".replace(",", "{,}")


def sgn(x, d=0):
    v = round(x, d)
    s = f"{abs(v):.{d}f}"
    return f"$+{s}$" if v > 0 else (f"$-{s}$" if v < 0 else f"${s}$")


def pct(x, d=1):
    return f"{100 * x:.{d}f}\\%"


def season(s):
    return f"{s}--{(s + 1) % 100:02d}"


def write(name, text):
    with open(os.path.join(OUT, name), "w") as f:
        f.write(text)


def off(age, gap):
    return next(o["elo_points"] for o in M["inactivity_offsets"] if o["age"] == age and o["gap"] == gap)


def main():
    os.makedirs(OUT, exist_ok=True)
    E = S["fitted_elo_points"]
    C = S["composition"]
    sim = {r["system"]: r for r in M["simulation"]}
    cs = {r["gap_from"]: r for r in S["calibration_stronger"]}
    scales = [r["gap_scale"] for r in WS]
    mean_scale = sum(scales) / len(scales)
    base_b = WP["FIDE as published"]["brier"]
    full = WP["FIDE + white + gap + age + form"]
    ctrl_b = sim["FIDE rules (control)"]["brier"]
    best = sim["Table + inactivity adjustment"]
    glicko_wins = sum(r["models"]["Glicko-2 rebuilt on this data"]["log_loss"]
                      < r["models"]["Elo rebuilt on this data"]["log_loss"] for r in WS)
    np_bt = R["without_pandemic"]["backtest"]
    np_sim = R["without_pandemic"]["simulation"]
    ctrl_off = {(o["age"], o["gap"]): o for o in R["inactivity_controlled"]}

    macros = {
        "NGames": n(sum(M["games_by_season"].values())),
        "NTestGames": n(W["test_games"]),
        "NPlayers": n(C["players"]),
        "NEvents": n(C["events"]),
        "NSeasons": str(len(WS)),
        "MedianRating": str(C["median_rating"]),
        "PTenRating": str(C["p10_rating"]),
        "PNinetyRating": str(C["p90_rating"]),
        "DrawRate": pct(C["draw_rate"]),
        "WhiteScore": pct(C["white_score"]),
        "GapScaleLatest": f"{S['gap_scale_model']['gap scale']:.2f}",
        "GapScaleMin": f"{min(scales):.2f}",
        "GapScaleMax": f"{max(scales):.2f}",
        "GapScaleMean": f"{mean_scale:.2f}",
        "GapWidePct": f"{round(100 * (1 - mean_scale))}\\%",
        "DivisorMean": str(round(400 / mean_scale / 10) * 10),
        "DivisorMin": str(min(M["divisor_by_season"].values())),
        "DivisorMax": str(max(M["divisor_by_season"].values())),
        "FideAtTwoHundred": pct(cs[200]["fide"]),
        "ActualAtTwoHundred": pct(cs[200]["actual"]),
        "FideAtFourHundred": pct(cs[400]["fide"]),
        "ActualAtFourHundred": pct(cs[400]["actual"]),
        "AgeTwelve": sgn(E["12-13"]),
        "AgeEighteen": sgn(E["18-20"]),
        "AgeFifty": sgn(E["50-64"]),
        "AgeSixtyFive": sgn(E["65+"]),
        "AgeTwelvePeak": sgn(max(r["age_12_13"] for r in WS)),
        "WhiteElo": str(round(E["white advantage"])),
        "WhiteMin": str(round(min(r["white"] for r in WS))),
        "WhiteMax": str(round(max(r["white"] for r in WS))),
        "FormTenth": str(round(S["form_coefficients"]["FIDE + white + recent form"] * 0.1)),
        "BrierFide": f"{base_b:.5f}",
        "BrierFull": f"{full['brier']:.5f}",
        "WFGain": pct(1 - full["brier"] / base_b),
        "SimGain": pct(1 - best["brier"] / ctrl_b),
        "SimTableGain": pct(1 - sim["Stretched table"]["brier"] / ctrl_b),
        "SimBestSeasons": str(best["seasons_better"]),
        "ReturnsAdjusted": n(sim["Inactivity adjustment"]["returns_adjusted"]),
        "GlickoWins": str(glicko_wins),
        "UncGrowth": f"{M['uncertainty_growth']:.0f}",
        "DecayMidOneTwo": sgn(off("21-49", "1-2 years")),
        "DecayMidTwoFour": sgn(off("21-49", "2-4 years")),
        "DecayMidFour": sgn(off("21-49", "4+ years")),
        "DecayOldOneTwo": sgn(off("50 and over", "1-2 years")),
        "DecayOldTwoFour": sgn(off("50 and over", "2-4 years")),
        "DecayOldFour": sgn(off("50 and over", "4+ years")),
        "DecayYoungOneTwo": sgn(off("under 21", "1-2 years")),
        "DecayMidFourCtrl": sgn(ctrl_off[("21-49", "4+ years")]["controlled"]),
        "DecayOldFourCtrl": sgn(ctrl_off[("50 and over", "4+ years")]["controlled"]),
        "DecayYoungOneTwoCtrl": sgn(ctrl_off[("under 21", "1-2 years")]["controlled"]),
        "NoPandemicFull": f"{np_bt['FIDE + white + gap + age + form']['vs_fide']:+.4f}".replace("-", "$-$"),
        "NoPandemicSim": f"{np_sim['Table + inactivity adjustment']['vs_control']:+.4f}".replace("-", "$-$"),
        "GapBandLow": f"{R['gap_scale_by_band'][0]['gap_scale']:.2f}",
        "GapBandTop": f"{R['gap_scale_by_band'][-1]['gap_scale']:.2f}",
        "FullLogLoss": f"{full['vs_fide']:+.4f}".replace("-", "$-$"),
        "SimLogLoss": f"{best['vs_control']:+.4f}".replace("-", "$-$"),
    }
    write("macros.tex", "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in macros.items()))

    # Table: pooled walk-forward backtest
    labels = [("FIDE as published", "Published FIDE ratings"),
              ("FIDE + white advantage", "+ white advantage"),
              ("FIDE + white + gap scale", "+ white, gap scale"),
              ("FIDE + white + age", "+ white, age offsets"),
              ("FIDE + white + recent form", "+ white, recent form"),
              ("FIDE + white + gap + age + form", "+ all corrections"),
              ("Elo rebuilt on this data", "Elo rebuilt on sample"),
              ("Glicko-2 rebuilt on this data", "Glicko-2 rebuilt on sample")]
    rows = []
    for k, lab in labels:
        r = WP[k]
        if k == "FIDE as published":
            rows.append(f"{lab} & {r['log_loss']:.5f} & {r['brier']:.5f} & baseline & \\\\")
        else:
            rows.append(f"{lab} & {r['log_loss']:.5f} & {r['brier']:.5f} & "
                        f"${r['vs_fide']:+.4f}$ [${r['ci_low']:+.4f}$, ${r['ci_high']:+.4f}$] & "
                        f"{r['seasons_better']}/{len(WS)} \\\\")
    write("table_backtest.tex", "\n".join(rows) + "\n")

    # Table: per season
    rows = []
    by = {r["season"]: r for r in WS}
    for s in range(WS[0]["season"] - 1, WS[-1]["season"] + 1):
        t, nx = by.get(s), by.get(s + 1)
        gain = pct(1 - t["models"]["FIDE + white + gap + age + form"]["brier"] / t["models"]["FIDE as published"]["brier"]) if t else "warm-up"
        rows.append(f"{season(s)} & {n(M['games_by_season'][str(s)])} & {gain} & "
                    f"{nx['gap_scale']:.2f} & {sgn(nx['age_12_13'])} & {nx['white']:.0f} \\\\" if nx else
                    f"{season(s)} & {n(M['games_by_season'][str(s)])} & {gain} & & & \\\\")
    write("table_seasons.tex", "\n".join(rows) + "\n")

    # Table: inactivity offsets, plain and controlled
    rows = []
    gaps = ["3-5 months", "6-11 months", "1-2 years", "2-4 years", "4+ years"]
    glab = {"3-5 months": "3--5 months", "6-11 months": "6--11 months", "1-2 years": "1--2 years",
            "2-4 years": "2--4 years", "4+ years": "4+ years"}
    for age, alab in (("under 21", "Under 21"), ("21-49", "21--49"), ("50 and over", "50 and over")):
        for i, g in enumerate(gaps):
            o = next(x for x in M["inactivity_offsets"] if x["age"] == age and x["gap"] == g)
            c = ctrl_off[(age, g)]
            rows.append(f"{alab if i == 0 else ''} & {glab[g]} & {n(o['player_games'])} & "
                        f"{sgn(o['elo_points'])} & {sgn(c['controlled'])} \\\\")
        rows.append("\\addlinespace")
    write("table_inactivity.tex", "\n".join(rows[:-1]) + "\n")

    # Table: simulation
    simlab = [("FIDE rules (control)", "Current FIDE rules"),
              ("Stretched table", "Stretched expectancy table"),
              ("K-factor by age", "Age-based K-factor"),
              ("Inactivity adjustment", "Inactivity adjustment"),
              ("Uncertainty K-factor", "Uncertainty-based K-factor"),
              ("Table + inactivity adjustment", "Table + inactivity"),
              ("Table + inactivity + uncertainty K", "Table + inactivity + uncertainty K")]
    rows = []
    for k, lab in simlab:
        r = sim[k]
        if k == "FIDE rules (control)":
            rows.append(f"{lab} & {r['log_loss']:.5f} & {r['brier']:.5f} & baseline & \\\\")
        else:
            rows.append(f"{lab} & {r['log_loss']:.5f} & {r['brier']:.5f} & "
                        f"${r['vs_control']:+.4f}$ [${r['ci_low']:+.4f}$, ${r['ci_high']:+.4f}$] & "
                        f"{r['seasons_better']}/{len(WS)} \\\\")
    write("table_simulation.tex", "\n".join(rows) + "\n")

    # Table: gap scale by rating band
    rows = [f"{b['band'].replace('-', '--')} & {n(b['games'])} & {b['gap_scale']:.2f} & {b['white']:.0f} \\\\"
            for b in R["gap_scale_by_band"]]
    write("table_gapband.tex", "\n".join(rows) + "\n")

    # Table: inactivity by rating level (adults)
    lv = {}
    for r in R["inactivity_by_rating_level"]:
        lv.setdefault(r["gap"], {})[r["level"]] = r
    order = ["active (0-2 months)"] + gaps
    glab2 = {"active (0-2 months)": "Active (0--2 months)", **glab}
    levels = ["below 2000", "2000-2299", "2300+"]
    rows = []
    for g in order:
        cells = []
        for l in levels:
            c = lv.get(g, {}).get(l)
            cells.append(f"{c['score_vs_expected_pct']:+.1f} ({n(c['player_games'])})".replace("-", "$-$") if c else "")
        rows.append(f"{glab2[g]} & " + " & ".join(cells) + " \\\\")
    write("table_level.tex", "\n".join(rows) + "\n")

    # Top-rated inactive players
    rows = []
    for t in M["top_2700"]:
        if t["flag"] != "i" and t["id"] != "2016192":
            continue
        name = " ".join(reversed([p.strip() for p in t["name"].split(",")])) if "," in t["name"] else t["name"]
        since = "168+" if t["months_since_last_game"] is None else str(t["months_since_last_game"])  # lists begin Sep 2012
        adj = sgn(t["return_adjustment"]) if t["return_adjustment"] else "none"
        rows.append(f"{name} & {t['rating']} & {t['age']} & {'inactive' if t['flag'] == 'i' else 'active'} & "
                    f"{t['classical_games_last_12_months']} & {since} & {adj} \\\\")
    write("table_top.tex", "\n".join(rows) + "\n")

    # Plot data
    write("plot_calib.dat", "gap fide model actual\n" + "".join(
        f"{r['gap_from'] + 25} {100 * r['fide']:.2f} {100 * r['gap_model']:.2f} {100 * r['actual']:.2f}\n"
        for r in S["calibration_stronger"]))
    ages = [("11 and under", "$\\leq$11"), ("12-13", "12--13"), ("14-15", "14--15"), ("16-17", "16--17"),
            ("18-20", "18--20"), ("50-64", "50--64"), ("65+", "65+")]
    coords = " ".join(f"({i},{E[k]:.1f})" for i, (k, _) in enumerate(ages, 1))
    ticks = ",".join(lab for _, lab in ages)
    write("fig_age.tex", "\\begin{tikzpicture}\n\\begin{axis}[ybar, width=0.8\\linewidth, height=5.8cm, bar width=16pt, "
          "ylabel={Elo points vs.\\ rating}, xlabel={Age group (reference: 21--49)}, "
          f"xtick={{1,...,{len(ages)}}}, xticklabels={{{ticks}}}, xmin=0.4, xmax={len(ages) + 0.6}, "
          "ymin=-85, ymax=70, grid=major, grid style={gray!25}, tick label style={font=\\small}, "
          "nodes near coords, nodes near coords style={font=\\scriptsize}, "
          "every node near coord/.append style={/pgf/number format/fixed, /pgf/number format/precision=0}]\n"
          f"\\addplot[fill=teal!45, draw=teal!70!black] coordinates {{{coords}}};\n"
          "\\end{axis}\n\\end{tikzpicture}\n")
    res = {(r["age"], r["gap"]): r for r in M["inactivity_residuals"]}
    for age, fname in (("under 21", "young"), ("21-49", "mid"), ("50 and over", "old")):
        pts = [(i, res[(age, g)]) for i, g in enumerate(order) if (age, g) in res and res[(age, g)]["player_games"] >= 150]
        write(f"plot_decay_{fname}.dat", "x y\n" + "".join(f"{i} {r['score_vs_expected_pct']:.2f}\n" for i, r in pts))
    lines = {"full": "FIDE + white + gap + age + form", "age": "FIDE + white + age",
             "gap": "FIDE + white + gap scale", "white": "FIDE + white advantage"}
    for key, model in lines.items():
        write(f"plot_seasons_{key}.dat", "x y\n" + "".join(
            f"{r['season']} {100 * (1 - r['models'][model]['brier'] / r['models']['FIDE as published']['brier']):.3f}\n"
            for r in WS))
    print(f"wrote {len(os.listdir(OUT))} files to paper/generated")


if __name__ == "__main__":
    main()
