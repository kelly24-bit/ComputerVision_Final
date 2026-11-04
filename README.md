# Soccer Tactical CV — Project Scaffold

A computer-vision project that reads tactical structure from football footage:
what kind of play is happening, and how a team is shaped.

> **Current stage: data verification.**
> Before any EDA or modelling, the only goal is to confirm every data source
> actually loads on our machines. Nothing in this repo trains a model yet.

---

## The problems we're solving

1. **Perception (shared front-end)** — turn a broadcast frame into player
   positions on a top-down pitch (detect → track → camera-calibrate).
   Everything downstream depends on this.
2. **Play recognition (classification)** — classify a clip as corner / free-kick /
   throw-in / open play / shot, etc. Evaluated with F1, confusion matrix, ROC.
3. **Formation recognition (probabilistic)** — predict the team's shape as a
   *probability distribution* over formations, since one snapshot is ambiguous.
   Evaluated on accuracy **and** calibration (Brier score, reliability curves).
4. **Methodological question (headline)** — can a model trained on *synthetic*
   formations generalize to real footage? We measure the synthetic-to-real gap.

Problems 2 and 3 are our two graded cognitive problems (kept distinct:
classification vs. calibrated prediction). Problem 1 is the shared front-end.

## Users

Primary: analysts and coaching staff (tactical shape + play tagging as decision
support). Broader audience: broadcasters, and the newer fans a home World Cup
brings in.

---

## Dataset map (what feeds what)

| Piece | Source | Access | What we take |
|-------|--------|--------|--------------|
| Play recognition | `SoccerNet/SN-Features` + `SoccerNet/SN-Labels` (Hugging Face) | Open, no login needed | Pre-extracted ResNet features (`resnet-tf2-pca512`) + `Labels-v2.json` (17 classes). **No raw video needed.** |
| Perception | `github.com/SoccerNet/sn-gamestate` (+ sn-calibration, sn-tracking) | Dataset gated via HF | End-to-end detect/track/calibrate → top-down minimap |
| Formation — training | our own procedural generator | none | Synthetic player coordinates |
| Formation — real test set | StatsBomb open data (`statsbombpy`) | **Open, no NDA** | `sb.frames()` 360 coordinates + `sb.lineups()` formation labels |

**Ignore** (other SoccerNet challenges, not ours): sn-caption, sn-echoes,
sn-depth, sn-mvfoul, sn-jersey, sn-reid, sn-vqa, sn-banner, sn-nvs, the
ball-action 12-class set, anything tagged `-2027`, and `SoccerNet_raw_HQ`
(NDA-gated raw video we don't need).

---

## What "the data works" means (our definition of done for this stage)

| Source | Verifiable from a fresh machine? | Success looks like |
|--------|----------------------------------|--------------------|
| StatsBomb | **Yes — open, no access needed** | `sb.competitions()` returns rows; a World Cup match's `frames()` and `lineups()` load, with a formation label present |
| SoccerNet features/labels | Yes, open, no login needed | `*.npy` feature files + `Labels-v2.json` download and open with sensible shapes (checked: `(5399, 512)` features and 195 annotations for one game) |
| sn-gamestate | Only after HF gated-access approval | baseline runs on one sample clip and emits minimap coordinates |

StatsBomb and the SoccerNet features/labels both work on a fresh machine, and
each script prints what a good result looks like. sn-gamestate hasn't been
checked yet and may be gated.

## Access notes

1. SoccerNet features/labels (SN-Features, SN-Labels) are open, so no Hugging
   Face login or approval is needed. If you want higher rate limits and faster
   downloads, run `hf auth login` or set `HF_TOKEN`. Without it you just get an
   "unauthenticated requests" warning.
2. sn-gamestate may be gated on Hugging Face. If the download is refused,
   request access on its dataset page.
3. StatsBomb: accept the open-data user agreement in their repo. No login.

**Push target:** `github.com/kelly24-bit` — set the git remote here. The
username alone grants no access; authenticate separately (`gh auth login` or a
personal access token). Never commit tokens or keys.

## Repo layout

```
data/raw, data/processed   # downloaded + derived data (git-ignored)
src/perception             # detect/track/calibrate front-end
src/play_recognition       # 17-class action classifier
src/formation              # synthetic generator + formation classifier
scripts/                   # data download + verification scripts
notebooks/                 # EDA (later)
report/                    # written deliverable (later)
```
