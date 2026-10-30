"""Build an IPL second-innings win-probability dataset from Cricsheet CSVs.

Expected layout (recommended):
    project_folder/
        build.py
        ipl_csv2/
            <match_id>.csv
            <match_id>_info.csv
            ...

The loader searches subfolders recursively, so the IPL folder can have another name.
Run from any working directory with: python build.py

Outputs are written beside this script:
    second_innings.csv       Main Power BI fact table; includes win_prob, wpa, winner
    all_balls.csv            Cleaned delivery-level table for innings 1 and 2
    model_coefficients.csv   Standardisation values and logistic regression coefficients
    model_metrics.csv        Held-out evaluation metrics
    model_comparison.csv     Three-feature vs four-feature comparison
    calibration.csv          Predicted probability vs observed win-rate bins
    data_quality.csv         Row counts and validation checks
"""
from __future__ import annotations

import csv
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
RANDOM_SEED = 42
MAX_LEGAL_BALLS = 120
FEATURES = ["runs_left", "balls_left", "wickets_left", "rrr"]
BASE_FEATURES = ["runs_left", "balls_left", "wickets_left"]
OUTPUT_NAMES = {
    "second_innings.csv", "all_balls.csv", "model_coefficients.csv",
    "model_metrics.csv", "model_comparison.csv", "calibration.csv",
    "data_quality.csv",
}
REQUIRED_SOURCE_COLUMNS = {"innings", "batting_team", "bowling_team", "runs_off_bat"}

TEAM_MAP = {
    "Delhi Daredevils": "Delhi Capitals",
    "Kings XI Punjab": "Punjab Kings",
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
    "Rising Pune Supergiant": "Rising Pune Supergiants",
}
SEASON_MAP = {"2007/08": "2008", "2009/10": "2010", "2020/21": "2020"}
VENUE_MAP = {
    "M.Chinnaswamy Stadium": "M Chinnaswamy Stadium",
    "Feroz Shah Kotla": "Arun Jaitley Stadium",
    "Punjab Cricket Association Stadium": "Punjab Cricket Association IS Bindra Stadium",
    "Sardar Patel Stadium": "Narendra Modi Stadium",
    "Zayed Cricket Stadium": "Sheikh Zayed Stadium",
}


def clean_venue(value: object) -> str:
    if pd.isna(value) or not str(value).strip():
        return "Unknown"
    name = str(value).split(",")[0].strip()
    return VENUE_MAP.get(name, name)


def read_match_winner(info_path: Path) -> str | None:
    """Read the winner from Cricsheet's key/value *_info.csv file safely."""
    if not info_path.exists():
        return None
    try:
        with info_path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.reader(handle):
                if len(row) >= 3 and row[0].strip().lower() == "info" and row[1].strip().lower() == "winner":
                    winner = ",".join(row[2:]).strip().strip('"')
                    return winner or None
    except (OSError, UnicodeError, csv.Error) as exc:
        warnings.warn(f"Could not read winner from {info_path}: {exc}")
    return None


def find_delivery_files() -> list[Path]:
    """Find raw Cricsheet delivery CSVs in any subfolder, not generated outputs."""
    candidates: list[Path] = []
    for path in ROOT.rglob("*.csv"):
        if path.parent == ROOT and path.name.lower() in OUTPUT_NAMES:
            continue
        if path.name.lower().endswith("_info.csv") or path.name.startswith("."):
            continue
        # Do not mistake a prior generated all_balls.csv for a raw match file.
        if path.name.lower() in OUTPUT_NAMES:
            continue
        try:
            header = set(pd.read_csv(path, nrows=0, encoding="utf-8-sig").columns.astype(str))
        except Exception:
            continue
        if REQUIRED_SOURCE_COLUMNS.issubset(header):
            candidates.append(path)
    # Keep deterministic order, while preserving each file's own delivery row order.
    return sorted(candidates, key=lambda p: str(p.relative_to(ROOT)).lower())


def load_raw_cricsheet(paths: list[Path]) -> pd.DataFrame:
    if not paths:
        raise FileNotFoundError(
            "No raw Cricsheet delivery CSVs were found under the folder containing build.py.\n"
            "Put your downloaded IPL CSV files in a subfolder (for example, ipl_csv2).\n"
            "The delivery files must be named <match_id>.csv and have matching <match_id>_info.csv files."
        )
    frames: list[pd.DataFrame] = []
    skipped_missing_winner = 0
    for path in paths:
        try:
            frame = pd.read_csv(path, low_memory=False, encoding="utf-8-sig")
        except Exception as exc:
            warnings.warn(f"Skipping unreadable file {path}: {exc}")
            continue
        if not REQUIRED_SOURCE_COLUMNS.issubset(frame.columns):
            continue
        match_id = path.stem
        info_path = path.with_name(f"{match_id}_info.csv")
        winner = read_match_winner(info_path)
        if winner is None:
            skipped_missing_winner += 1
        frame["match_id"] = match_id
        frame["winner"] = winner
        frame["_source_order"] = np.arange(len(frame), dtype=np.int64)
        frame["_source_file"] = str(path.relative_to(ROOT))
        frames.append(frame)
    if not frames:
        raise ValueError("CSV files were found, but none had the expected Cricsheet delivery columns.")
    data = pd.concat(frames, ignore_index=True, sort=False)
    print(f"Raw delivery files loaded: {len(frames):,}")
    if skipped_missing_winner:
        print(f"Files without a readable winner (will be excluded from modelling): {skipped_missing_winner:,}")
    return data


def prepare_data(source: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, set[str], dict]:
    missing = sorted(REQUIRED_SOURCE_COLUMNS - set(source.columns))
    if missing:
        raise ValueError(f"Raw data is missing required columns: {missing}")
    data = source.copy()
    if "match_id" not in data:
        raise ValueError("Internal data error: match_id was not assigned to the source rows.")

    # Standardise optional/messy columns so different Cricsheet CSV releases can be handled.
    for col in ["runs_off_bat", "extras", "wides", "noballs"]:
        if col not in data.columns:
            data[col] = 0
        data[col] = pd.to_numeric(data[col], errors="coerce").fillna(0)
    for col in ["innings"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    for col in ["match_id", "batting_team", "bowling_team"]:
        data[col] = data[col].astype("string").str.strip()
    if "winner" not in data:
        data["winner"] = pd.NA
    data["winner"] = data["winner"].astype("string").str.strip()
    for col in ["batting_team", "bowling_team", "winner"]:
        data[col] = data[col].replace(TEAM_MAP)

    if "season" not in data.columns:
        data["season"] = "Unknown"
    data["season"] = data["season"].astype("string").replace(SEASON_MAP)
    if "venue" not in data.columns:
        data["venue"] = "Unknown"
    data["venue"] = data["venue"].map(clean_venue)
    if "start_date" not in data.columns:
        data["start_date"] = pd.NaT
    data["start_date"] = pd.to_datetime(data["start_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    data["start_date"] = data["start_date"].fillna("")
    for col in ["striker", "bowler", "wicket_type"]:
        if col not in data.columns:
            data[col] = pd.NA

    # Keep only completed-result matches with first and second innings. Super Overs (innings 3+)
    # are intentionally not included in this standard 20-over chase model.
    valid = data["winner"].notna() & data["winner"].ne("") & data["innings"].isin([1, 2])
    data = data.loc[valid].copy()
    if data.empty:
        raise ValueError(
            "No rows remain after filtering. Check that matching *_info.csv files exist and contain an info,winner row."
        )

    data["total_runs"] = data["runs_off_bat"] + data["extras"]
    wicket_type = data["wicket_type"].astype("string").str.strip().str.lower()
    data["is_wicket"] = (wicket_type.notna() & wicket_type.ne("") & wicket_type.ne("retired hurt")).astype(int)
    data["is_legal"] = ((data["wides"] == 0) & (data["noballs"] == 0)).astype(int)
    data = data.sort_values(["match_id", "innings", "_source_order"], kind="stable").reset_index(drop=True)

    # Calculate each innings' running state using source delivery order.
    grp = data.groupby(["match_id", "innings"], sort=False, observed=True)
    data["delivery_no"] = grp.cumcount() + 1
    data["ball_no"] = data["delivery_no"]  # delivery count, including wides/no-balls; not an over.ball label
    data["cum_runs"] = grp["total_runs"].cumsum()
    data["cum_wkts"] = grp["is_wicket"].cumsum()
    data["legal_balls"] = grp["is_legal"].cumsum()

    first_innings = data.loc[data["innings"].eq(1)].groupby("match_id", sort=False).agg(
        first_innings_runs=("total_runs", "sum"),
        first_innings_legal_balls=("legal_balls", "max"),
        first_innings_wickets=("cum_wkts", "max"),
    )
    data["target"] = data["match_id"].map(first_innings["first_innings_runs"] + 1)
    data = data.loc[data["target"].notna()].copy()

    # The model assumes a 120-legal-ball innings and a conventional target. Exclude likely
    # shortened innings rather than pretending their states are directly comparable.
    short_first = set(first_innings.index[
        (first_innings["first_innings_legal_balls"] < MAX_LEGAL_BALLS)
        & (first_innings["first_innings_wickets"] < 10)
    ].astype(str))
    last_second = data.loc[data["innings"].eq(2)].groupby("match_id", sort=False).tail(1).set_index("match_id")
    short_chase = set(last_second.index[
        ((last_second["target"] - last_second["cum_runs"]) > 0)
        & (last_second["cum_wkts"] < 10)
        & (last_second["legal_balls"] < MAX_LEGAL_BALLS)
    ].astype(str))
    excluded = short_first | short_chase
    if excluded:
        data = data.loc[~data["match_id"].astype(str).isin(excluded)].copy()

    data = data.sort_values(["match_id", "innings", "_source_order"], kind="stable").reset_index(drop=True)
    grp = data.groupby(["match_id", "innings"], sort=False, observed=True)
    data["delivery_no"] = grp.cumcount() + 1
    data["ball_no"] = data["delivery_no"]
    data["cum_runs"] = grp["total_runs"].cumsum()
    data["cum_wkts"] = grp["is_wicket"].cumsum()
    data["legal_balls"] = grp["is_legal"].cumsum()
    first_runs = data.loc[data["innings"].eq(1)].groupby("match_id", sort=False)["total_runs"].sum()
    data["target"] = data["match_id"].map(first_runs + 1)

    chase = data.loc[data["innings"].eq(2) & data["legal_balls"].le(MAX_LEGAL_BALLS)].copy()
    chase["runs_left"] = (chase["target"] - chase["cum_runs"]).clip(lower=0)
    chase["balls_left"] = (MAX_LEGAL_BALLS - chase["legal_balls"]).clip(lower=0)
    chase["wickets_left"] = (10 - chase["cum_wkts"]).clip(lower=0)
    chase["rrr"] = chase["runs_left"] * 6 / chase["balls_left"].replace(0, np.nan)
    chase["rrr"] = chase["rrr"].replace([np.inf, -np.inf], np.nan).fillna(0)
    chase["crr"] = chase["cum_runs"] * 6 / chase["legal_balls"].replace(0, np.nan)
    chase["crr"] = chase["crr"].replace([np.inf, -np.inf], np.nan).fillna(0)
    chase["batting_won"] = chase["batting_team"].eq(chase["winner"]).astype(int)
    chase["season_year"] = pd.to_numeric(chase["season"].astype(str).str.extract(r"(\d{4})", expand=False), errors="coerce")
    chase = chase.replace([np.inf, -np.inf], np.nan)
    chase = chase.dropna(subset=FEATURES + ["batting_won"])
    if chase.empty or chase["batting_won"].nunique() < 2:
        raise ValueError("Not enough eligible second-innings data or winner variation remains to train the model.")

    diagnostics = {
        "source_rows": len(source),
        "rows_after_result_and_innings_filter": int(valid.sum()),
        "cleaned_all_ball_rows": len(data),
        "eligible_chase_rows": len(chase),
        "eligible_matches": int(chase["match_id"].nunique()),
        "excluded_shortened_matches": len(excluded),
        "matches_without_winner_in_source": int(source["winner"].isna().sum()) if "winner" in source else 0,
    }
    return data, chase, excluded, diagnostics


def fit_model(train: pd.DataFrame, features: list[str]):
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train[features])
    model = LogisticRegression(max_iter=3000, random_state=RANDOM_SEED)
    model.fit(x_train, train["batting_won"])
    return scaler, model


def metric_values(y_true: pd.Series, probabilities: np.ndarray) -> dict:
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-7, 1 - 1e-7)
    y = np.asarray(y_true, dtype=int)
    return {
        "accuracy": round(float(accuracy_score(y, p >= 0.5)), 5),
        "auc": round(float(roc_auc_score(y, p)), 5) if np.unique(y).size > 1 else np.nan,
        "log_loss": round(float(log_loss(y, p, labels=[0, 1])), 5),
        "brier": round(float(brier_score_loss(y, p)), 5),
    }


def evaluate_split(train: pd.DataFrame, test: pd.DataFrame, split_name: str, features: list[str]) -> tuple[dict, np.ndarray]:
    if train.empty or test.empty or train["batting_won"].nunique() < 2:
        raise ValueError(f"Cannot evaluate split '{split_name}': empty data or training data has only one outcome class.")
    scaler, model = fit_model(train, features)
    probabilities = model.predict_proba(scaler.transform(test[features]))[:, 1]
    row = metric_values(test["batting_won"], probabilities)
    baseline_p = np.full(len(test), float(train["batting_won"].mean()))
    row.update({
        "split": split_name,
        "features": "+".join(features),
        "train_matches": int(train["match_id"].nunique()),
        "test_matches": int(test["match_id"].nunique()),
        "test_rows": len(test),
        "baseline_log_loss": round(float(log_loss(test["batting_won"], np.clip(baseline_p, 1e-7, 1 - 1e-7), labels=[0, 1])), 5),
    })
    return row, probabilities


def make_calibration(test: pd.DataFrame, probabilities: np.ndarray, split_name: str) -> pd.DataFrame:
    frame = test[["match_id", "season", "batting_won"]].copy()
    frame["predicted_probability"] = probabilities
    frame["probability_bin"] = pd.cut(
        frame["predicted_probability"], bins=np.linspace(0, 1, 11), include_lowest=True
    ).astype(str)
    result = frame.groupby("probability_bin", observed=True).agg(
        n=("predicted_probability", "size"),
        avg_predicted_probability=("predicted_probability", "mean"),
        observed_win_rate=("batting_won", "mean"),
    ).reset_index()
    result.insert(0, "split", split_name)
    return result.round(5)


def write_csv(frame: pd.DataFrame, filename: str) -> None:
    frame.to_csv(ROOT / filename, index=False, encoding="utf-8-sig")


def main() -> None:
    files = find_delivery_files()
    source = load_raw_cricsheet(files)
    all_balls, chase, excluded, diagnostics = prepare_data(source)
    if chase["match_id"].nunique() < 10:
        raise ValueError(f"Only {chase['match_id'].nunique()} eligible matches remain; at least 10 are required.")

    # Split by match, never by individual delivery: a match cannot leak across train and test.
    match_ids = np.array(sorted(chase["match_id"].astype(str).unique()))
    rng = np.random.RandomState(RANDOM_SEED)
    rng.shuffle(match_ids)
    n_test = max(1, int(round(0.2 * len(match_ids))))
    random_test_ids = set(match_ids[:n_test])
    random_train = chase.loc[~chase["match_id"].astype(str).isin(random_test_ids)].copy()
    random_test = chase.loc[chase["match_id"].astype(str).isin(random_test_ids)].copy()

    train_chrono = chase.loc[chase["season_year"].notna() & chase["season_year"].le(2023)].copy()
    test_chrono = chase.loc[chase["season_year"].notna() & chase["season_year"].ge(2024)].copy()
    splits = [("random 80/20 by match", random_train, random_test)]
    if not train_chrono.empty and not test_chrono.empty and train_chrono["batting_won"].nunique() == 2:
        splits.append(("train <=2023, test 2024+", train_chrono, test_chrono))
    else:
        warnings.warn("Chronological evaluation unavailable: the data must contain seasons both through 2023 and from 2024 onward.")

    metric_rows, comparison_rows, calibration_frames = [], [], []
    for split_name, train, test in splits:
        metrics, probabilities = evaluate_split(train, test, split_name, FEATURES)
        metric_rows.append(metrics)
        calibration_frames.append(make_calibration(test, probabilities, split_name))
        for model_name, feature_set in [("3-feature baseline", BASE_FEATURES), ("4-feature model", FEATURES)]:
            comparison, _ = evaluate_split(train, test, split_name, feature_set)
            comparison["model"] = model_name
            comparison_rows.append(comparison)

    # Final model is fit on all eligible chase deliveries for historical replay in Power BI.
    # These are fitted historical probabilities, not out-of-sample predictions for each row.
    scaler, model = fit_model(chase, FEATURES)
    chase = chase.copy()
    chase["win_prob"] = model.predict_proba(scaler.transform(chase[FEATURES]))[:, 1]
    chase["win_prob"] = chase["win_prob"].clip(0, 1)
    won_now = chase["runs_left"].le(0)
    innings_finished_without_win = (chase["wickets_left"].le(0) | chase["legal_balls"].ge(MAX_LEGAL_BALLS)) & ~won_now
    chase.loc[won_now, "win_prob"] = 1.0
    chase.loc[innings_finished_without_win, "win_prob"] = 0.0

    # A tie is only labelled as a tie when the match result metadata explicitly says so;
    # standard Cricsheet winner-based training normally excludes tied/no-result games.
    starts = chase.groupby("match_id", sort=False).agg(target=("target", "first")).reset_index()
    starts["runs_left"] = starts["target"]
    starts["balls_left"] = MAX_LEGAL_BALLS
    starts["wickets_left"] = 10
    starts["rrr"] = starts["runs_left"] * 6 / MAX_LEGAL_BALLS
    starts["start_prob"] = model.predict_proba(scaler.transform(starts[FEATURES]))[:, 1]
    start_lookup = dict(zip(starts["match_id"], starts["start_prob"]))
    chase["prev_prob"] = chase.groupby("match_id", sort=False)["win_prob"].shift(1)
    chase["prev_prob"] = chase["prev_prob"].fillna(chase["match_id"].map(start_lookup))
    chase["wpa"] = chase["win_prob"] - chase["prev_prob"]

    coefficient_rows = [{"feature": "intercept", "coef": float(model.intercept_[0]), "mean": 0.0, "std": 1.0}]
    for index, feature in enumerate(FEATURES):
        coefficient_rows.append({
            "feature": feature,
            "coef": float(model.coef_[0][index]),
            "mean": float(scaler.mean_[index]),
            "std": float(scaler.scale_[index]),
        })

    # Stable, documented columns for Power BI. These are deliberately created in this script.
    second_columns = [
        "match_id", "season", "start_date", "venue", "innings", "batting_team", "bowling_team",
        "striker", "bowler", "ball_no", "delivery_no", "legal_balls", "runs_off_bat", "extras",
        "total_runs", "is_wicket", "wicket_type", "cum_runs", "cum_wkts", "target", "runs_left",
        "balls_left", "wickets_left", "rrr", "crr", "win_prob", "prev_prob", "wpa", "winner",
        "batting_won",
    ]
    for col in second_columns:
        if col not in chase.columns:
            chase[col] = pd.NA
    second = chase[second_columns].copy()
    write_csv(second, "second_innings.csv")

    # Export both innings with common cumulative fields for general match-level analysis.
    all_columns = [
        "match_id", "season", "start_date", "venue", "innings", "delivery_no", "ball_no",
        "legal_balls", "batting_team", "bowling_team", "striker", "bowler", "runs_off_bat",
        "extras", "total_runs", "is_wicket", "wicket_type", "cum_runs", "cum_wkts", "target", "winner",
    ]
    for col in all_columns:
        if col not in all_balls.columns:
            all_balls[col] = pd.NA
    write_csv(all_balls[all_columns], "all_balls.csv")
    write_csv(pd.DataFrame(coefficient_rows), "model_coefficients.csv")
    write_csv(pd.DataFrame(metric_rows), "model_metrics.csv")
    write_csv(pd.DataFrame(comparison_rows), "model_comparison.csv")
    write_csv(pd.concat(calibration_frames, ignore_index=True) if calibration_frames else pd.DataFrame(
        columns=["split", "probability_bin", "n", "avg_predicted_probability", "observed_win_rate"]
    ), "calibration.csv")

    quality_rows = [{"check": key, "value": value, "notes": "Build diagnostic"} for key, value in diagnostics.items()]
    quality_rows.extend([
        {"check": "source_delivery_files", "value": len(files), "notes": "Cricsheet delivery CSV files discovered recursively"},
        {"check": "missing_win_prob", "value": int(second["win_prob"].isna().sum()), "notes": "Expected 0"},
        {"check": "win_prob_outside_0_1", "value": int((~second["win_prob"].between(0, 1)).sum()), "notes": "Expected 0"},
        {"check": "missing_wpa", "value": int(second["wpa"].isna().sum()), "notes": "Expected 0; first delivery uses estimated pre-innings probability"},
        {"check": "missing_winner", "value": int((second["winner"].isna() | second["winner"].eq("")).sum()), "notes": "Expected 0 for included completed-result matches"},
        {"check": "duplicate_match_delivery_keys", "value": int(second.duplicated(["match_id", "delivery_no"]).sum()), "notes": "Expected 0"},
        {"check": "minimum_season_year", "value": int(chase["season_year"].min()) if chase["season_year"].notna().any() else "Unknown", "notes": "Parsed from season label"},
        {"check": "maximum_season_year", "value": int(chase["season_year"].max()) if chase["season_year"].notna().any() else "Unknown", "notes": "Parsed from season label"},
    ])
    write_csv(pd.DataFrame(quality_rows), "data_quality.csv")

    # Fail the build loudly if required output columns or core values are missing.
    required_output = {"win_prob", "wpa", "winner", "match_id", "target", "runs_left", "balls_left", "wickets_left"}
    missing_output = sorted(required_output - set(second.columns))
    if missing_output:
        raise RuntimeError(f"second_innings.csv is missing required columns: {missing_output}")
    if second.empty:
        raise RuntimeError("second_innings.csv is empty; no eligible chases were produced.")
    if second["win_prob"].isna().any() or not second["win_prob"].between(0, 1).all():
        raise RuntimeError("Probability validation failed: missing or out-of-range win_prob values.")

    print("\nEvaluation metrics:")
    print(pd.DataFrame(metric_rows).to_string(index=False))
    print("\nModel comparison:")
    print(pd.DataFrame(comparison_rows)[["model", "split", "auc", "log_loss", "brier"]].to_string(index=False))
    print("\nBUILD SUCCESSFUL")
    print(f"Source delivery files: {len(files):,}")
    print(f"Eligible matches: {chase['match_id'].nunique():,}")
    print(f"Rows in second_innings.csv: {len(second):,}")
    print("Confirmed columns: win_prob, wpa, winner")
    print("Created CSVs: second_innings.csv, all_balls.csv, model_coefficients.csv, model_metrics.csv,")
    print("              model_comparison.csv, calibration.csv, data_quality.csv")


if __name__ == "__main__":
    main()
