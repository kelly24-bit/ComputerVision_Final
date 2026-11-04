# ADSP 32023 Final Project: Football Formation & Play Recognition

Computer vision on football footage to read tactical structure: what kind of play is happening, and what shape a team is in.

**Status:** data checked and explored (EDA done for both datasets). No models trained yet.

## What we're solving

1. **Perception (shared front-end):** turn a broadcast frame into player positions on a top-down pitch (detect, track, calibrate). Everything else depends on it.
2. **Play recognition (classification):** label a clip as corner, free-kick, throw-in, shot and so on. Evaluated with F1, confusion matrix and ROC.
3. **Formation recognition (probabilistic):** predict a team's shape as a probability distribution over formations, since one snapshot is ambiguous. Evaluated on accuracy and calibration (Brier score, reliability curves).
4. **Headline question:** does a model trained on synthetic formations hold up on real footage? We measure the synthetic-to-real gap.

Problems 2 and 3 are the two graded cognitive problems, kept separate on purpose (classification vs calibrated prediction). The intended users are analysts and coaching staff, with broadcasters and newer fans as a wider audience.

## Getting started

Tested on Python 3.13 (the pinned numpy and pandas are recent, so an older Python may not work). Run everything from the repo root.

```bash
# 1. create and activate a virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1          # Windows PowerShell
# source .venv/bin/activate         # macOS / Linux

# 2. install pinned dependencies
pip install -r requirements.txt

# 3. check StatsBomb open data loads (quick)
python scripts/check_statsbomb.py

# 4. download and check SoccerNet features + labels (about 4 minutes)
python scripts/download_soccernet.py
```

Both datasets are open, so no account or login is needed. Step 4 downloads the whole `valid` split (100 games), not a single game, into `data/raw/soccernet/`.

Healthy output for step 3 ends with `StatsBomb data confirmed working.` (80 competitions, 64 World Cup matches, 360 frames loaded). Step 4 ends with `SoccerNet features + labels confirmed working.` (a `(5399, 512)` feature array and 195 annotations for the first game).

The EDA notebooks are `notebooks/eda.ipynb` (SoccerNet) and `notebooks/eda_statsbomb.ipynb` (StatsBomb). The StatsBomb one pulls 64 matches on its first run (about 8 minutes) and caches them in `data/raw/statsbomb/`, so later runs are fast.

## Troubleshooting

These show up on a normal, working run. None of them is a bug.

| You see | Why | What to do |
|---|---|---|
| `NoAuthWarning: credentials were not supplied. open data access only` (StatsBomb) | We use the free open data, which needs no credentials. | Nothing. |
| `You are sending unauthenticated requests to the HF Hub` (SoccerNet download) | The datasets are open. Without a token you only get lower rate limits. | Nothing, or run `hf auth login` / set `HF_TOKEN` if downloads are slow. |
| `huggingface-hub ... does not provide the extra 'cli'` (during `pip install`) | A harmless pip notice for the pinned huggingface_hub version. | Nothing. |
| `LF will be replaced by CRLF the next time Git touches it` | Git is converting line endings on Windows. | Nothing. |

If the SoccerNet download does fail, check your network first. The script's error message says the same.

## Architecture

```
Play recognition   SoccerNet features (ResNet, 512-d) + Labels-v2  ->  17-class classifier
Formation          synthetic player coordinates  ->  classifier  ->  tested on StatsBomb World Cup 360 data
Perception         broadcast frame -> detect / track / calibrate -> top-down positions  (sn-gamestate, later)
```

- **Play recognition** uses the pre-extracted features and `Labels-v2.json`, so no raw video is needed. The 17 classes are very imbalanced (Ball out of play is 29% of annotations, Yellow->red card under 0.1%), so we plan on macro-F1 and game-level splits.
- **Formation** trains on our own procedural generator and tests on the 2022 men's World Cup via `statsbombpy`. That gives only 128 independent team-matches (64 matches x 2 teams) with a starting formation, even though there are about 400k team snapshots (200k freeze-frames). Each freeze-frame shows around half of a team's players, so models have to handle partial views.
- **Perception** (sn-gamestate) is the shared front-end. It's gated and we haven't checked access yet.

Datasets and what we take from each:

| Piece | Source | Access |
|---|---|---|
| Play recognition | `SoccerNet/SN-Features` + `SN-Labels` (Hugging Face) | Open |
| Formation (training) | our own generator | none |
| Formation (real test set) | StatsBomb open data via `statsbombpy` (`frames()` for 360 coordinates, `events()` for formations) | Open |
| Perception | `SoccerNet/sn-gamestate` (+ sn-calibration, sn-tracking) | Gated on Hugging Face |

Not used: sn-caption, sn-echoes, sn-depth, sn-mvfoul, sn-jersey, sn-reid, sn-vqa, sn-banner, sn-nvs, the 12-class ball-action set, anything tagged `-2027`, and `SoccerNet_raw_HQ` (NDA-gated raw video).

## Milestones

Fill in the checkpoint column with the class dates.

| Milestone | Checkpoint | Status |
|---|---|---|
| Data access verified (StatsBomb, SoccerNet features/labels) | TBD | Done. sn-gamestate not checked yet |
| EDA: SoccerNet class balance and data quality | TBD | Done |
| EDA: StatsBomb formations, 360 coverage, independent sample size | TBD | Done |
| Synthetic formation generator | TBD | Not started |
| Play-recognition classifier (17 classes) | TBD | Not started |
| Formation classifier with calibration | TBD | Not started |
| Synthetic-to-real gap evaluation on StatsBomb | TBD | Not started |
| Perception front-end (sn-gamestate) | TBD | Not started |
| Written report | TBD | Not started |

## Repo layout

```
data/raw, data/processed   downloaded and derived data (git-ignored)
scripts/                   download and check scripts
src/                       data_loaders.py, statsbomb_loaders.py
notebooks/                 EDA notebooks
reports/figures/           figures saved by the notebooks
src/perception             planned: detect/track/calibrate front-end
src/play_recognition       planned: 17-class action classifier
src/formation              planned: synthetic generator + formation classifier
```

The `report/` folder from the original plan isn't created yet.

## Notes

Never commit tokens or keys. Pushing to `github.com/kelly24-bit/ComputerVision_Final` needs your own GitHub auth (`gh auth login` or a personal access token).
