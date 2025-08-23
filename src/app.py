import asyncio

import orjson
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, ORJSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.responses import StreamingResponse

from src.process import text2audio_processor, text_processor
from src.schema import InputProcessedText

templates = Jinja2Templates(directory="templates")

app = FastAPI(title="Text2Audio")
app.mount("/static", StaticFiles(directory="static"), name="static")
# Optional CORS (adjust origins as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/synthesize")
def synthesize_text(request: Request, processed_text: InputProcessedText) -> ORJSONResponse:
    """
    Endpoint to synthesize text to audio.
    The processed_text should contain the raw text or URL for the audio file.
    """

    async def generate_response(processed_text: InputProcessedText) -> StreamingResponse:
        yield "data: " + orjson.dumps({"status": "started"}).decode("utf-8") + "\n\n"
        await asyncio.sleep(0)

        output_processed_text = await text_processor.execute(processed_text)
        async for state in text2audio_processor.execute(output_processed_text):
            response_data = {
                "content": output_processed_text.content,
                **dict(state),
            }
            yield f"data: {orjson.dumps(response_data).decode('utf-8')}\n\n"
            await asyncio.sleep(0)

    return StreamingResponse(
        generate_response(processed_text),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # nginx: disable buffering
        },
    )
