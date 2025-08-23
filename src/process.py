import asyncio
import os
import re
from datetime import datetime
from time import perf_counter
from uuid import uuid4

import httpx
import numpy as np
from bs4 import BeautifulSoup
from pydub import AudioSegment
from readability import Document

from src.constants import OUTPUT_DIR
from src.model import text2audio_model
from src.schema import InputProcessedText, OutputProcessedText


class Text2AudioProcessor:
    async def execute(self, processed_text: OutputProcessedText) -> str:
        """
        Process the text or URL from the processed_text object and synthesize it to audio.
        This function should implement the actual text-to-audio synthesis logic.
        """
        start_time = perf_counter()
        filename = f"synthesized-{uuid4().hex}-{datetime.now().strftime('%Y%m%d%H%M%S')}.mp3"
        file_path = os.path.join(OUTPUT_DIR, filename)

        sample_rate = text2audio_model.get_sample_rate()

        full_audio = []
        for audio in text2audio_model.synthesize(text=processed_text.summary, voice=processed_text.voice):
            audio_data = audio.cpu().numpy().tolist()
            full_audio.append(audio_data)
            yield {
                "status": "processing",
                "time_taken": perf_counter() - start_time,
                "audio_data": audio_data,
                "sample_rate": sample_rate,
            }
            await asyncio.sleep(0)

        audio_array = np.concatenate(full_audio).squeeze()
        audio_int = np.int16(audio_array * 32767)
        audio_segment = AudioSegment(
            audio_int.tobytes(),
            frame_rate=sample_rate,
            sample_width=audio_int.dtype.itemsize,
            channels=2 if len(audio_int.shape) > 1 else 1,
        )
        audio_segment.export(file_path, format="mp3")

        yield {
            "status": "done",
            "time_taken": perf_counter() - start_time,
            "audio_data": text2audio_model.silence.cpu().numpy().tolist(),
            "sample_rate": sample_rate,
            "audio": f"/static/audio/{filename}",
        }
        await asyncio.sleep(0)


class TextProcessor:
    async def _load_url_text(self, url: str) -> str:
        """
        Load text from a URL.
        This function fetches the content from the provided URL and returns it as a string.
        """
        headers = {
            "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 13_5_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/13.1.1 Mobile/15E148 Safari/604.1"
        }
        with httpx.Client() as client:
            response = client.get(url, headers=headers)
            response.raise_for_status()
            return response.content.decode("utf-8")

    async def _parse_text(self, loaded_text: str) -> InputProcessedText:
        """
        Parse the loaded text and return a InputProcessedText object.
        This function can be extended to include more complex parsing logic if needed.
        """
        document = Document(loaded_text)
        content = document.summary(html_partial=True)
        content = BeautifulSoup(content, "html.parser").get_text(separator="\n").strip()
        if title := document.title().replace("[no-title]", "").strip():
            content = f"{title}: {content}"
        summary = re.sub(r'[()\[\]{}"\']+', "", content)
        summary = re.sub(r"[\'—;]+", ", ", summary)
        summary = re.sub(r"[%]+", " porcento", summary)
        summary = re.sub(r",+", ",", summary)
        content = re.sub(r"\.$", ".<br>", summary, flags=re.MULTILINE)
        return OutputProcessedText(summary=summary, content=content)

    async def execute(self, processed_text: InputProcessedText) -> OutputProcessedText:
        """
        Process the text or URL from the processed_text object and return a OutputProcessedText object.
        This function should implement the actual text processing logic.
        """
        raw_text = processed_text.raw_text.strip()
        if processed_text.url:
            raw_text = await self._load_url_text(processed_text.url)
        output_processed_text = await self._parse_text(raw_text)
        output_processed_text.voice = processed_text.voice or output_processed_text.voice
        return output_processed_text


text_processor = TextProcessor()
text2audio_processor = Text2AudioProcessor()
