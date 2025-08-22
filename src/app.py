from time import perf_counter

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, ORJSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

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
    start_time = perf_counter()
    output_processed_text = text_processor.execute(processed_text)
    audio_path = text2audio_processor.execute(output_processed_text)
    return ORJSONResponse(
        {
            "time_taken": perf_counter() - start_time,
            "content": output_processed_text.content,
            "audio_path": audio_path,
        }
    )
