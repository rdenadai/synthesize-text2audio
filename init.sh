#!/bin/bash

apt update & apt install -y curl ffmpeg
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env 
uv sync
uv run python -m nltk.downloader punkt
uv run python -m nltk.downloader punkt_tab
uv run uvicorn src.app:app --host 0.0.0.0 --port 8000 --reload --workers 1 --log-level info