# Streamlit UI

Local-only Streamlit UI for creating and generating training videos.

## Run

```powershell
python -m pip install -r requirements.txt
streamlit run app.py
```

## How it works

- **Create** tab writes a new run folder under `runs/` (e.g., `storyboard.json`, `script.json`, `prompts.json`, `prompts_readable.md`).
- **Generate** tab starts `python cli.py generate --run <run_dir>` as a background process.
- **Stop after current clip** writes `STOP_AFTER_CURRENT` into the run folder; the CLI pauses after finishing the current clip.
- **Results** tab stitches clips into `final_video.mp4`.
