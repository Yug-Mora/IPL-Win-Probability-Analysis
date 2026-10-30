# IPL Win Probability Prediction & Analytics Dashboard

An end-to-end cricket analytics project that transforms historical IPL
ball-by-ball data into delivery-level win-probability estimates and an
interactive Power BI dashboard.

The project uses Python for data preparation, feature engineering,
logistic-regression modelling and evaluation, and Power BI for
interactive analysis and visual storytelling.

## Project highlights

-   Processes raw Cricsheet-style IPL delivery CSV files and their
    matching match-information files.
-   Builds a structured second-innings dataset with match state, win
    probability (`win_prob`) and win-probability added (`wpa`).
-   Trains a logistic regression model using match-state features.
-   Evaluates model performance using random match-level and, when the
    data supports it, chronological holdout splits.
-   Exports model metrics, model comparisons, calibration data and
    data-quality checks.
-   Presents historical match and model insights in an interactive Power
    BI report.

## Repository structure

Keep the Python script and raw-data folder together. The generated CSVs
and Power BI report can be stored in the repository as shown below.

``` text
IPL-Win-Probability/
├── build.py
├── ipl_csv2/
│   ├── <match_id>.csv
│   ├── <match_id>_info.csv
│   └── ...
├── second_innings.csv
├── all_balls.csv
├── model_coefficients.csv
├── model_metrics.csv
├── model_comparison.csv
├── calibration.csv
├── data_quality.csv
├── IPL_Win_Probability.pbix
├── screenshots/
│   └── dashboard-overview.png
└── README.md
```

The `ipl_csv2` folder name is an example. The script searches
recursively beneath the folder containing `build.py` for raw delivery
CSVs.

> **Data note:** Raw Cricsheet files are not included in this repository
> by default. Obtain the data from Cricsheet and follow its current
> terms of use and attribution requirements. Do not commit raw data
> unless you have checked that redistribution is permitted.

## Requirements

-   Python 3.9 or later
-   Power BI Desktop (to open and interact with the `.pbix` report)
-   Python packages:
    -   pandas
    -   numpy
    -   scikit-learn

## Run the Python pipeline

1.  Clone or download this repository.

2.  Place the raw IPL delivery CSV files and their corresponding
    `_info.csv` files inside `ipl_csv2` (or another subfolder under the
    project directory).

3.  Open a terminal in the project directory.

4.  Install the dependencies:

    ``` bash
    python -m pip install pandas numpy scikit-learn
    ```

5.  Run the build script:

    ``` bash
    python build.py
    ```

The script writes the generated CSV files beside `build.py`.

### Expected raw-data file pairing

For each match, the delivery file must have a matching information file
in the same folder. For example:

``` text
ipl_csv2/
├── 1234567.csv
└── 1234567_info.csv
```

The delivery CSV must contain the expected Cricsheet-style columns,
including `innings`, `batting_team`, `bowling_team` and `runs_off_bat`.
The information file must contain the match winner record.

## Generated outputs

  -----------------------------------------------------------------------
  File                                Purpose
  ----------------------------------- -----------------------------------
  `second_innings.csv`                Main Power BI table for
                                      second-innings delivery analysis.
                                      Includes `win_prob`, `prev_prob`,
                                      `wpa`, `winner`, match state and
                                      team fields.

  `all_balls.csv`                     Cleaned delivery-level data for the
                                      first and second innings, with
                                      cumulative runs and wickets.

  `model_coefficients.csv`            Logistic-regression coefficients
                                      and feature standardisation values.

  `model_metrics.csv`                 Held-out model evaluation metrics.

  `model_comparison.csv`              Comparison of the three-feature
                                      baseline and four-feature model.

  `calibration.csv`                   Predicted probability versus
                                      observed win rate grouped into
                                      probability bins.

  `data_quality.csv`                  Build diagnostics and validation
                                      checks.
  -----------------------------------------------------------------------

## Modelling approach

### Prediction target

The model estimates the probability that the team batting second (the
chasing team) wins the match, based on the state of the chase at a given
delivery.

### Features

The four model features are:

-   `runs_left`: runs still required to reach the target.
-   `balls_left`: legal balls remaining in a standard 120-ball innings.
-   `wickets_left`: wickets remaining.
-   `rrr`: required run rate.

A `StandardScaler` is fitted on the training data, followed by a
scikit-learn `LogisticRegression` model.

### Evaluation

The script evaluates the model using:

1.  **Random 80/20 split by match:** matches, rather than individual
    delivery rows, are assigned to train and test groups to reduce
    match-level leakage.
2.  **Chronological split, when available:** seasons through 2023 are
    used for training and 2024 onward for testing, provided both groups
    contain sufficient outcome variation.

Reported metrics include:

-   Accuracy
-   ROC-AUC
-   Log loss
-   Brier score

The script also compares the four-feature model with a three-feature
baseline that excludes required run rate.

## Win probability and WPA

-   `win_prob` is the model-estimated probability of the chasing team
    winning at a delivery state.
-   `prev_prob` is the preceding delivery's estimated probability. For
    the first delivery, the script uses an estimated pre-innings
    probability.
-   `wpa` is calculated as `win_prob - prev_prob`.

Positive WPA means the estimated chance of the chasing team winning
increased between the previous state and the current state. Negative WPA
means it decreased.

WPA is a change in modelled probability; it should not be interpreted as
proof that a delivery caused the change.

## Power BI dashboard

The Power BI report is designed around four analysis areas:

1.  **IPL Overview** --- match counts, chase wins, chase win rate,
    scoring trends and season-wise comparisons.
2.  **Win Probability** --- match selection, delivery-level chasing-team
    win probability and WPA.
3.  **Team & Venue Insights** --- comparisons of chasing performance,
    targets and venues.
4.  **Model Reliability** --- evaluation metrics, model comparison,
    calibration and data-quality checks.

Open `IPL_Win_Probability.pbix` in Power BI Desktop. If you regenerate
the CSV outputs, refresh the data sources in Power BI and verify that
the file paths still point to the project folder.

## Important limitations

-   This is a historical analysis and replay project, not a live IPL
    prediction system.
-   The model is designed for conventional 20-over (120 legal-ball)
    chases.
-   The pipeline excludes matches without a readable winner and limits
    modelling to innings 1 and 2. Super Overs are not modelled.
-   A heuristic excludes likely shortened innings; this is not a full
    implementation of every historical DLS or revised-target rule.
-   The final `win_prob` values exported for historical replay are
    generated by a model fitted on all eligible chase deliveries. They
    are fitted historical estimates, not out-of-sample predictions for
    every individual delivery.
-   Use `model_metrics.csv` and `calibration.csv` to assess held-out
    performance. Prefer the chronological evaluation when it is
    available and relevant to the intended use.
-   Ties, Super Over outcomes, abandoned matches and unusual match
    conditions may not be represented perfectly by a standard chase
    model.
-   Results depend on the raw-data release and the matches available in
    the local dataset.

## Suggested repository improvements

Before sharing the repository publicly:

-   Add a screenshot of the Power BI overview page to `screenshots/`.
-   Include the `.pbix` file if its size is practical and you are
    comfortable sharing it.
-   Do not upload raw data unless redistribution is permitted by its
    source terms.
-   Add a short note identifying the dataset source and the date/version
    downloaded.
-   Check that the README, script and dashboard do not contain personal
    information or local file paths.
-   If the `.pbix` is too large for GitHub's normal file limit, use Git
    LFS or share the report through an appropriate alternative.

## Author

**Yug Mora**

Built as a portfolio project demonstrating data preparation, feature
engineering, predictive modelling, model evaluation and business
intelligence reporting.
