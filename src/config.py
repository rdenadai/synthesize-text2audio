from pydantic_settings import BaseSettings, SettingsConfigDict


class HuggingFaceSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="huggingface_")
    tts_model: str = "suno/bark"
    voice: str = "v2/pt_speaker_0"
    max_length: int = 5120


class Settings(BaseSettings):
    huggingface: HuggingFaceSettings = HuggingFaceSettings()


settings = Settings()
