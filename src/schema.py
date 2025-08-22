from pydantic import BaseModel, model_validator

from src.config import settings


class InputProcessedText(BaseModel):
    raw_text: str | None
    url: str | None
    voice: str = settings.huggingface.voice

    @model_validator(mode="after")
    def validate(self):
        if not self.raw_text and not self.url:
            raise ValueError("Either raw_text or url must be provided.")
        return self


class OutputProcessedText(BaseModel):
    content: str
    summary: str
    voice: str = settings.huggingface.voice
