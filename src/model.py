import nltk
import torch
from transformers import BarkModel, BarkProcessor

from src.config import settings

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


class Text2AudioModel:
    def __init__(self):
        self.processor = BarkProcessor.from_pretrained(settings.huggingface.tts_model)
        self.model = BarkModel.from_pretrained(settings.huggingface.tts_model, low_cpu_mem_usage=True).to(DEVICE)
        self.model.generation_config.pad_token_id = self.processor.tokenizer.eos_token_id

    def synthesize(self, text: str, voice: str = settings.huggingface.voice) -> torch.Tensor:
        text = text.replace("\n", " ").strip()
        sentences = nltk.sent_tokenize(text, language="portuguese" if "v2/pt" in voice else "english")
        silence = torch.zeros((1, int(self.get_sample_rate() * 0.25)), dtype=torch.float32).to(DEVICE)

        pieces = []
        for sentence in sentences:
            inputs = self.processor(
                sentence,
                max_length=settings.huggingface.max_length,
                voice_preset=voice,
                return_tensors="pt",
            )
            for key in inputs:
                if hasattr(inputs[key], "to"):
                    inputs[key] = inputs[key].to(DEVICE)

            audio_array = self.model.generate(
                **inputs,
                do_sample=True,
                temperature=0.6,
                min_eos_p=0.05,
            )
            pieces += [audio_array, silence.clone()]

        return torch.cat(pieces, dim=-1).squeeze()

    def get_sample_rate(self):
        return self.model.generation_config.sample_rate


text2audio_model = Text2AudioModel()
