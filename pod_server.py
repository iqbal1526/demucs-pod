from fastapi import FastAPI, File, UploadFile
from demucs.pretrained import get_model
from demucs.apply import apply_model
import torch, torchaudio, uuid, os

app = FastAPI()
DEVICE = torch.device("cuda")

# Load Demucs once on startup
model = get_model("htdemucs_6s").to(DEVICE).eval()

@app.post("/separate")
async def separate(file: UploadFile = File(...)):
    job_id = uuid.uuid4().hex
    inp = f"/tmp/{job_id}_{file.filename}"
    outdir = f"/tmp/out_{job_id}"
    os.makedirs(outdir, exist_ok=True)

    # Save upload
    with open(inp, "wb") as f:
        f.write(await file.read())

    # Run Demucs separation
    waveform, sr = torchaudio.load(inp)
    waveform = waveform[:2]
    if sr != model.samplerate:
        waveform = torchaudio.functional.resample(waveform, sr, model.samplerate)
        sr = model.samplerate
    waveform = waveform.to(DEVICE)
    with torch.no_grad():
        sep = apply_model(model, waveform.unsqueeze(0), shifts=1)[0]

    order = model.sources
    vocals = sep[order.index("vocals")]
    accom  = sum(sep[i] for i,s in enumerate(order) if s!="vocals")

    # Save outputs
    vocals_path = os.path.join(outdir, "vocals.wav")
    accom_path  = os.path.join(outdir, "accompaniment.wav")
    torchaudio.save(vocals_path, vocals.cpu(), sr)
    torchaudio.save(accom_path, accom.cpu(), sr)

    # Return URLs (RunPod will serve /static → outdir)
    return {
      "vocals_url":         f"/static/{job_id}/vocals.wav",
      "accompaniment_url":  f"/static/{job_id}/accompaniment.wav"
    }
