# Synthesize Text2Audio

![alt text](image.png)

## Install dependencies

If you don't have `uv` installed, you can install it curling the following command [Installing uv](https://docs.astral.sh/uv/getting-started/installation/):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

After that in order to run the application, you need to install the required dependencies. You can do this using `uv` (a virtual environment manager).

```bash
uv sync
```

> P.S.: Take a look at the `pyproject.toml` file, in particular the [[tool.uv.index]] section, because in my machine I had to add the `pytorch-cu118` index to be able to install the `torch` package. If you have a new NVIDIA GPU, you might need to change the index URL to match your CUDA version (or remove it if you don't need it).

Install `ffmpeg` if you don't have it installed yet. This is required to convert audio files to mp3 format.

```bash
sudo apt install ffmpeg
```

## Create a .env file

Create a `.env` file in the root directory of the project.

This file should contain the Hugging Face model names and other configurations. Here is an example of what your `.env` file might look like:

The bellow values are the default ones used in the project, but you can change them to use different models or configurations as needed.

```bash
HUGGINGFACE_TTS_MODEL=suno/bark
HUGGINGFACE_VOICE=v2/pt_speaker_0
HUGGINGFACE_MAX_LENGTH=5120
```

## Run the server

To use the application you should run as a server.

```bash
uv run uvicorn src.app:app --host 0.0.0.0 --port 8000 --reload
```
