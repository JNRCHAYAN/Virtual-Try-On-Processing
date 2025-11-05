
# 🧥 Virtual Try-On — Inpainting Demo (FREE, Local)

A simple **Streamlit** web app that performs **virtual try-on** by *inpainting* a clothing item into a selected body region (Upper or Lower) using **Stable Diffusion Inpainting**.

## ✨ Features
- Upload a person image (JPG/PNG)
- Choose **Upper** or **Lower** region
- Enter a clothing **text prompt** (e.g., "a Hawaiian shirt")
- App builds a **mask** for the selected region (uses **MediaPipe Pose** if available, else a robust fallback rectangle)
- Runs **Stable Diffusion Inpainting** locally (free)

## ⚙️ Install (Python 3.10+ recommended)
> Torch is *not* pinned here because the correct build depends on your system. Install Torch first, then the rest.

### 1) Install PyTorch (CPU example)
Visit https://pytorch.org/get-started/locally/ or run (CPU-only):
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```
> If you have NVIDIA GPU/CUDA, choose the matching command from the PyTorch site.

### 2) Install the app deps
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note:** `mediapipe` is optional. If it fails to install, the app still works using a fallback mask.

## ▶️ Run
```bash
streamlit run app.py
```
Open the local URL it prints (usually http://localhost:8501).

## 📝 Tips
- Use clear, descriptive prompts for better results (e.g., "a stylish black leather jacket with zippers, slim fit").
- Higher **steps** and guidance may improve fidelity but are slower.
- If results look misaligned, try a different photo angle or tweak the prompt.

## 🧩 How it works
1. Your image is resized to 768 and padded to a square.
2. We compute a **region mask** (Upper torso or Lower body).
   - If **MediaPipe Pose** is installed, we use its landmarks to shape the mask.
   - Otherwise, we use a fallback rectangle with soft edges.
3. We run **Stable Diffusion Inpainting** (`runwayml/stable-diffusion-inpainting`) with your prompt and mask.
4. The output is resized back to the original image size.
5. You can **download** both the final image and the mask.

## 📦 Model
- Inpainting: `runwayml/stable-diffusion-inpainting` (downloaded automatically by `diffusers` on first run).

## ❓ Troubleshooting
- **Torch install fails:** Use the command from https://pytorch.org and pick CPU or GPU build matching your system.
- **MediaPipe install fails:** It's optional — the app will fallback to a rectangle mask.
- **Slow on CPU:** That’s expected. GPU with CUDA is much faster.
- **"xformers" missing warning:** Ignored — it's optional.

---

MIT License
