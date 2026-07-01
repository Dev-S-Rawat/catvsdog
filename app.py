import io
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image


from src.model import get_model
from src.dataset import data_transforms

app = FastAPI(
    title="Cat vs Dog Image Classifier API",
    description="A production-ready deep learning API using MobileNetV3.",
    version="1.0.0"
)

# checks for gpu, default = cpu
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


MODEL_PATH = "weights/model.pth"
model = get_model(num_classes=2)

try:
    # Load the saved state dict adjusting weights, map to matching device
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()  # Set to evaluation mode (freezes dropout/batchnorm)
    print(f"📦 Model loaded successfully onto {device}!")
except Exception as e:
    print(f"❌ Failed to load the model artifact from {MODEL_PATH}: {str(e)}")
    raise e


CATEGORIES = {0: "Cat", 1: "Dog"}

@app.get("/")
def read_root():
    """Health check endpoint to ensure API container is responsive."""
    return {"status": "healthy", "model_loaded": True, "device": str(device)}

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    """Accepts an image upload file, processes it, and predicts Cat vs Dog."""
    # Safety Check: Verify file is actually an image
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be a valid image format.")

    try:
        # Read the raw byte data from incoming request
        image_bytes = await file.read()

        # Open and force standard 3-channel RGB format
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Apply the exact same preprocessing transforms we used during training
        tensor_image = data_transforms(image).unsqueeze(0).to(
            device)  # unsqueeze adds a batch dimension [1, 3, 224, 224]

        # Disable gradient calculations for blazing fast inference
        with torch.no_grad():
            outputs = model(tensor_image)
            # Run outputs through a Softmax function to translate raw logits into probabilities (0.0 - 1.0)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)

        return {
            "filename": file.filename,
            "prediction": CATEGORIES[predicted_idx.item()],
            "confidence": round(confidence.item(), 4)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference Engine Error: {str(e)}")