# backend.py
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from datetime import datetime
import shutil, uuid

from trainer import start_training, view_logs, run_inference # Assuming trainer.py exists

app = FastAPI()

# Allow access from your local frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # you can restrict this later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Serve generated images (accessible at http://localhost:8000/outputs/<filename>.png)
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")


@app.post("/train")
async def train_model(zip_file: UploadFile, trigger_word: str = Form(...)):
    save_path = UPLOAD_DIR / zip_file.filename
    with open(save_path, "wb") as f:
        shutil.copyfileobj(zip_file.file, f)

    msg = start_training(save_path, trigger_word)
    return {"message": msg}


@app.get("/logs")
async def get_logs():
    logs = view_logs()
    return {"logs": logs}


@app.post("/generate")
async def generate_image(prompt: str = Form(...)):
    try:
        image = run_inference(prompt)  # returns PIL Image
        fname = f"generated_{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:8]}.png"
        out_path = OUTPUT_DIR / fname
        image.save(out_path)
        # Note: The frontend will prepend API_BASE (http://localhost:8000)
        return JSONResponse(status_code=200, content={"url": f"/outputs/{fname}"}) 
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# 🆕 New endpoint to get the list of generated images
@app.get("/images")
async def list_images():
    # List all files in the OUTPUT_DIR with a .png extension
    # We only return the URL path relative to the server root
    image_paths = [
        f"/outputs/{p.name}" 
        for p in OUTPUT_DIR.glob("*.png") 
        if p.is_file()
    ]
    # Sort by creation time (or simply by filename if creation time is encoded in it, like yours)
    image_paths.sort(reverse=True) 
    
    return {"images": image_paths}
