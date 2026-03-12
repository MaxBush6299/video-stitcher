"""Optional one-time setup for local Streamlit runs."""

from pathlib import Path


def main() -> None:
	# Ensure runs directory exists (the app writes artifacts here)
	Path("runs").mkdir(exist_ok=True)

	# Optional Streamlit server config (no theme overrides)
	streamlit_dir = Path(".streamlit")
	streamlit_dir.mkdir(exist_ok=True)
	config_path = streamlit_dir / "config.toml"
	if not config_path.exists():
		config_path.write_text(
			"""[server]
port = 8501
""",
			encoding="utf-8",
		)

	print("Setup complete. Run: streamlit run app.py")


if __name__ == "__main__":
	main()
