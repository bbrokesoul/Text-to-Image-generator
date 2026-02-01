# trainer.py (replace your current file with this)

# INSTALLS & IMPORTS

import os, random, zipfile, shutil, threading, time, traceback
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from diffusers import StableDiffusionPipeline
from transformers import CLIPTokenizer
from peft import get_peft_model, LoraConfig, PeftModel
from accelerate import Accelerator

# DATASET CLASS

class SimpleDataset(Dataset):
    def __init__(self, folder, tokenizer, size=512):
        self.paths = list(Path(folder).glob("*.jpg")) + list(Path(folder).glob("*.png"))
        self.tokenizer = tokenizer
        self.transform = transforms.Compose([
            transforms.Resize((size, size), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            # For RGB use 3 values
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path = self.paths[idx]
        txt_path = img_path.with_suffix(".txt")
        caption = txt_path.read_text().strip() if txt_path.exists() else "a photo"
        img = Image.open(img_path).convert("RGB")
        return {
            "pixel_values": self.transform(img),
            "input_ids": self.tokenizer(
                caption,
                padding="max_length",
                truncation=True,
                max_length=77,
                return_tensors="pt"
            ).input_ids[0]
        }


# TRAINING FUNCTION

def run_training(zip_path, trigger_word, output_dir="lora_model", epochs=3, lr=1e-4):
    log_file = "train_log.txt"
    with open(log_file, "w") as f:
        f.write(f"[{time.ctime()}] Starting training...\n")

    try:
        # prepare working dir
        work_dir = Path("training_data")
        if work_dir.exists():
            shutil.rmtree(work_dir)
        work_dir.mkdir(exist_ok=True)

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(work_dir)

        # auto captions
        for img_path in work_dir.glob("*"):
            if img_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                img_path.with_suffix(".txt").write_text(f"a photo of {trigger_word}")

        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        dataset = SimpleDataset(work_dir, tokenizer)
        dl = DataLoader(dataset, batch_size=2, shuffle=True)

        model_id = "runwayml/stable-diffusion-v1-5"
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id, torch_dtype=torch.float16, safety_checker=None
        ).to("cuda")

        lora_config = LoraConfig(
            r=4, lora_alpha=16,
            target_modules=["to_q", "to_v"],
            lora_dropout=0.05, bias="none"
        )
        pipe.unet = get_peft_model(pipe.unet, lora_config)
        pipe.vae.requires_grad_(False)
        pipe.text_encoder.requires_grad_(False)

        optimizer = torch.optim.AdamW(pipe.unet.parameters(), lr=lr)
        accelerator = Accelerator(mixed_precision="fp16")
        unet, optimizer, dl = accelerator.prepare(pipe.unet, optimizer, dl)

        import torch.nn.functional as F
        for epoch in range(epochs):
            unet.train()
            loss = None
            for batch in tqdm(dl, desc=f"Epoch {epoch+1}/{epochs}"):
                with accelerator.autocast():
                    images = batch["pixel_values"].to(accelerator.device, dtype=torch.float32)
                    input_ids = batch["input_ids"].to(accelerator.device)
                    # encode expects pixel values in [-1,1] — our normalize matches that
                    latents = pipe.vae.encode(images).latent_dist.sample().to(dtype=torch.float32) * 0.18215
                    latents = latents.to(dtype=torch.float16)
                    noise = torch.randn_like(latents)
                    timesteps = torch.randint(
                        0, pipe.scheduler.config.num_train_timesteps,
                        (latents.shape[0],), device=latents.device
                    ).long()
                    noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)
                    encoder_hidden_states = pipe.text_encoder(input_ids)[0]
                    noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states).sample
                    loss = F.mse_loss(noise_pred.float(), noise.float())
                optimizer.zero_grad()
                accelerator.backward(loss)
                optimizer.step()
            with open(log_file, "a") as f:
                f.write(f"[{time.ctime()}] Epoch {epoch+1}/{epochs} done - Loss: {loss.item():.4f}\n")

        os.makedirs(output_dir, exist_ok=True)
        # Save the LoRA weights (unet) that PEFT expects
        unet.save_pretrained(output_dir)
        with open(log_file, "a") as f:
            f.write(f"[{time.ctime()}] Training Complete!\n")
    except Exception as e:
        with open(log_file, "a") as f:
            f.write(f"[{time.ctime()}] EXCEPTION: {str(e)}\n")
            f.write(traceback.format_exc())
        raise


# THREAD WRAPPER

train_thread = None

def start_training(zip_file, trigger_word):
    global train_thread
    log_file = "train_log.txt"
    if os.path.exists(log_file):
        os.remove(log_file)

    def thread_target():
        try:
            run_training(zip_file, trigger_word)
        except Exception as e:
            # ensure exception is written to log
            with open(log_file, "a") as f:
                f.write(f"[{time.ctime()}] training thread exception: {e}\n")

    train_thread = threading.Thread(target=thread_target, daemon=True)
    train_thread.start()
    return "Training started in background. Click 'Refresh Logs' to follow progress."

def view_logs():
    log_file = "train_log.txt"
    if not os.path.exists(log_file):
        return "Logs will appear here shortly..."
    # return last part of the file
    text = open(log_file, "r").read()
    return text[-4000:]

# INFERENCE FUNCTION (with caching & clear errors)
# caching globals for pipeline and loaded LoRA
_PIPELINE = None
_LORA_ATTACHED = None

def _load_base_pipeline(model_id="runwayml/stable-diffusion-v1-5"):
    global _PIPELINE
    if _PIPELINE is None:
        _PIPELINE = StableDiffusionPipeline.from_pretrained(
            model_id, torch_dtype=torch.float16, safety_checker=None
        ).to("cuda")
    return _PIPELINE

def run_inference(prompt):
    """
    Returns a PIL.Image.Image or raises an exception with a helpful message.
    """
    model_dir = "lora_model"
    if not Path(model_dir).exists():
        raise FileNotFoundError(f"Expected LoRA weights in '{model_dir}' but directory not found. Train first.")

    try:
        pipe = _load_base_pipeline()
        # attach LoRA to the UNet only once (cache the PeftModel)
        global _LORA_ATTACHED
        if _LORA_ATTACHED is None:
            # This wraps/unpacks the unet with PeftModel weights from model_dir
            _LORA_ATTACHED = PeftModel.from_pretrained(pipe.unet, model_dir).to("cuda")
            pipe.unet = _LORA_ATTACHED

        # run generation (move prompt to device handled by pipeline)
        result = pipe(prompt, num_inference_steps=30, guidance_scale=7.5)
        image = result.images[0]
        if image is None:
            raise RuntimeError("Pipeline returned no image.")
        return image
    except Exception as e:
        # add helpful context
        raise RuntimeError(f"run_inference failed: {e}\n{traceback.format_exc()}")
