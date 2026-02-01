# Text-to-Image-generator(LoRA Based)

This project lets you **train your own AI image generator** using a small image dataset and then **generate realistic images using text prompts**.
You upload **50–70 images**, train the model for a few minutes, and then generate new images through a **simple web interface**.

🎥 **Demo Video:**
👉 *(Paste your video link here)*

---

## ✨ What This Project Does (In Simple Words)

* Upload a ZIP file containing images
* Enter a **trigger word** (unique name for your subject)
* Train a lightweight AI model using **LoRA**
* Generate images by writing normal English prompts
* View and download all generated images from the gallery

No command-line ML knowledge is required to use it — everything is done via a webpage.

---

## 📁 Project Structure

```
.
├── README.md                  # Project documentation
├── backend.py                 # FastAPI backend server
├── trainer.py                 # Model training + image generation logic
├── index.html                 # Frontend UI (open in browser)
├── requirements.txt           # Python dependencies
├── taylor_swift_135 (1).zip   # Example dataset (for testing)
├── uploads/                   # Uploaded training ZIP files
├── outputs/                   # Generated images (auto-created)
├── lora_model/                # Trained LoRA weights (auto-created)
└── train_log.txt              # Training logs
```

---

## 🧠 How It Works (Conceptually)

* Uses **Stable Diffusion v1.5** as the base model
* Trains **LoRA weights only** (very small & fast)
* Base model stays frozen
* Training takes **~5–10 minutes on a basic GPU**
* Generated images follow your custom subject using the trigger word

This approach is **fast, memory-efficient, and practical**.

---

## 🖥️ Frontend (How Users Interact)

The entire interaction happens through `index.html`.

### The webpage allows you to:

1. Upload a dataset ZIP
2. Enter a trigger word
3. Start training
4. Write a prompt
5. Generate images
6. View all generated images in a gallery
7. Download images

No API calls or technical knowledge required from the user side.

---

## 🚀 How to Run the Project

### 1️⃣ Create Python Environment (Recommended)

```bash
conda create -n image-gen python=3.10
conda activate image-gen
```

### 2️⃣ Install Requirements

```bash
pip install -r requirements.txt
```

> ⚠️ **CUDA GPU is required** (training & generation run on GPU)

---

### 3️⃣ Start Backend Server

```bash
uvicorn backend:app --reload
```

Backend runs at:

```
http://localhost:8000
```

---

### 4️⃣ Open Frontend

Simply open:

```
index.html
```

in your browser (Chrome / Edge recommended).

---

## Dataset Rules (Important)

* ZIP file should contain **only images**
* Supported formats: `.jpg`, `.jpeg`, `.png`
* Recommended: **50–70 images**
* No captions needed (they are auto-generated)

Example ZIP structure:

```
dataset.zip
├── img1.jpg
├── img2.png
├── img3.jpg
```

---

## Trigger Word

The trigger word is how the model recognizes your subject.

Example:

```
Trigger word: tswift
Prompt: a cinematic portrait of tswift in a red dress
```

Choose something **unique** and **not a common word**.

---

## Image Generation

After training:

* Write a normal English prompt
* The model uses your trained LoRA weights
* Generated images are saved automatically
* Images appear in the gallery section

All generated images are stored inside:

```
outputs/
```

---

## Training Logs

Training progress is written to:

```
train_log.txt
```

You can refresh logs from the UI to see:

* Epoch progress
* Loss values
* Errors (if any)

---

## Possible Improvements

* Multiple subject support
* Training progress bar
* Model versioning
* Online deployment
* Authentication

---

## Credits

* Stable Diffusion – RunwayML
* Hugging Face Diffusers
* PEFT (LoRA)
* FastAPI

