import io
import json
import os
from pathlib import Path

import requests
import torch
import torch.nn.functional as F
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from torchvision import models, transforms


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "food_model_best.pth"
CLASSES_PATH = BASE_DIR / "food_classes.json"
USDA_API_KEY = os.environ.get("USDA_API_KEY", "C6Fn6hDY3yjymEH1CvcC5bXxXO2DWA5gxi68W9A3")
USDA_SEARCH_URL = "https://api.nal.usda.gov/fdc/v1/foods/search"
OFF_API_URL = "https://world.openfoodfacts.org/api/v2/product"
OFF_USER_AGENT = "AaharAI/1.0 (realsamiradhikari2061@gmail.com)"


def load_classes(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as f:
        classes = json.load(f)
    if not isinstance(classes, list) or not classes:
        raise ValueError("food_classes.json must be a non-empty JSON list.")
    return classes


def build_model(num_classes: int, arch: str) -> torch.nn.Module:
    builder = {
        "efficientnet_b0": models.efficientnet_b0,
        "efficientnet_b1": models.efficientnet_b1,
        "efficientnet_b2": models.efficientnet_b2,
        "efficientnet_b3": models.efficientnet_b3,
        "efficientnet_b4": models.efficientnet_b4,
    }.get(arch)
    if builder is None:
        raise ValueError(f"Unsupported architecture: {arch}")

    model = builder(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(in_features, num_classes)
    return model


def load_model(model_path: Path, num_classes: int, device: torch.device) -> torch.nn.Module:
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, dict):
        raise ValueError("Unsupported checkpoint format. Expected a state_dict dictionary.")

    candidate_arches = [
        "efficientnet_b3",
        "efficientnet_b2",
        "efficientnet_b1",
        "efficientnet_b0",
        "efficientnet_b4",
    ]

    load_errors = []
    for arch in candidate_arches:
        model = build_model(num_classes, arch)
        try:
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()
            print(f"Loaded checkpoint with architecture: {arch}")
            return model
        except RuntimeError as exc:
            load_errors.append(f"{arch}: {exc}")

    raise RuntimeError(
        "Could not load checkpoint with EfficientNet variants. "
        "Tried architectures: "
        + ", ".join(candidate_arches)
        + "\n\nDetails:\n"
        + "\n\n".join(load_errors)
    )


def format_label(raw_label: str) -> str:
    return raw_label.replace("_", " ").title()


def _pick_nutrient(food_nutrients: list[dict], candidate_names: list[str], preferred_unit: str | None = None) -> dict | None:
    for nutrient_name in candidate_names:
        for item in food_nutrients:
            if item.get("nutrientName") != nutrient_name:
                continue
            unit = item.get("unitName")
            value = item.get("value")
            if value is None:
                continue
            if preferred_unit is None or unit == preferred_unit:
                return {
                    "name": nutrient_name,
                    "value": value,
                    "unit": unit,
                }
    return None


def fetch_nutrition_for_label(class_name: str) -> dict:
    query = class_name.replace("_", " ")
    params = {
        "api_key": USDA_API_KEY,
        "query": query,
        "pageSize": 1,
    }

    try:
        response = requests.get(USDA_SEARCH_URL, params=params, timeout=12)
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        return {
            "query": query,
            "found": False,
            "error": f"USDA API request failed: {exc}",
        }

    foods = payload.get("foods") or []
    if not foods:
        return {
            "query": query,
            "found": False,
            "error": "No USDA match found for this food.",
        }

    best_match = foods[0]
    nutrients = best_match.get("foodNutrients") or []
    serving_size = best_match.get("servingSize")
    serving_unit = best_match.get("servingSizeUnit")
    household_serving = best_match.get("householdServingFullText")
    data_type = best_match.get("dataType") or "Unknown"

    if serving_size is not None and serving_unit:
        basis = f"Per {serving_size} {serving_unit}"
    elif household_serving:
        basis = f"Per {household_serving}"
    else:
        basis = (
            "Basis not explicitly provided in this USDA record. "
            "Many USDA records are per 100 g, while branded foods are often per serving."
        )

    calories = _pick_nutrient(nutrients, ["Energy"], preferred_unit="KCAL") or _pick_nutrient(nutrients, ["Energy"])
    protein = _pick_nutrient(nutrients, ["Protein"])
    carbs = _pick_nutrient(nutrients, ["Carbohydrate, by difference"])
    fat = _pick_nutrient(nutrients, ["Total lipid (fat)"])
    fiber = _pick_nutrient(nutrients, ["Fiber, total dietary"])
    sugar = _pick_nutrient(nutrients, ["Sugars, total including NLEA", "Sugars, total"])
    sodium = _pick_nutrient(nutrients, ["Sodium, Na"])

    highlights = [
        {"label": "Calories", "data": calories},
        {"label": "Protein", "data": protein},
        {"label": "Carbohydrates", "data": carbs},
        {"label": "Fat", "data": fat},
        {"label": "Fiber", "data": fiber},
        {"label": "Sugar", "data": sugar},
        {"label": "Sodium", "data": sodium},
    ]

    formatted = []
    for item in highlights:
        if item["data"] is None:
            continue
        formatted.append(
            {
                "label": item["label"],
                "value": round(float(item["data"]["value"]), 2),
                "unit": item["data"].get("unit") or "",
            }
        )

    return {
        "query": query,
        "found": True,
        "matched_description": best_match.get("description") or query,
        "fdc_id": best_match.get("fdcId"),
        "display_basis": "Per 100 g (or per 100 ml for liquid products)",
        "basis": basis,
        "data_type": data_type,
        "serving_size": serving_size,
        "serving_unit": serving_unit,
        "household_serving": household_serving,
        "highlights": formatted,
    }


def predict_image(model: torch.nn.Module, preprocess: transforms.Compose, image: Image.Image, class_names: list[str], device: torch.device) -> dict:
    tensor = preprocess(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.inference_mode():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)

    top_probs, top_indices = torch.topk(probs[0], k=3)
    top_results = []
    for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
        top_results.append(
            {
                "class_index": idx,
                "class_name": class_names[idx],
                "display_name": format_label(class_names[idx]),
                "confidence": round(prob * 100, 2),
            }
        )

    return {"prediction": top_results[0], "top_3": top_results}


app = FastAPI(title="AaharAI Model API", version="2.0.0")

# ── Recommendation router ──────────────────────────────────────────────────
from recommendation.recommendation_main import router as recommend_router, load_dataset
load_dataset()
app.include_router(recommend_router, prefix="/recommend")
# ──────────────────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_names = load_classes(CLASSES_PATH)
model = load_model(MODEL_PATH, len(class_names), device)

preprocess = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


@app.get("/")
def root():
    return {
        "service": "food-cv-fastapi",
        "status": "ok",
        "message": "Use POST /predict with multipart form field name 'image'.",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": True,
        "num_classes": len(class_names),
        "device": str(device),
    }


@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    if image is None or not image.filename:
        return JSONResponse(
            status_code=400,
            content={"error": "Please upload an image using the 'image' form field."},
        )

    try:
        image_bytes = await image.read()
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        result = predict_image(
            model=model,
            preprocess=preprocess,
            image=pil_image,
            class_names=class_names,
            device=device,
        )

        nutrition = fetch_nutrition_for_label(result["prediction"]["class_name"])
        # If not found, frontend should show nutrients not available.
        result["nutrition"] = nutrition
        return result
    except Exception as exc:
        return JSONResponse(status_code=500, content={"error": f"Prediction failed: {exc}"})


@app.post("/barcode-lookup")
async def barcode_lookup(payload: dict):
    barcode = str(payload.get("barcode", "")).strip()
    if not barcode.isdigit():
        return JSONResponse(
            status_code=400,
            content={"found": False, "message": "Product not found in the database"},
        )

    try:
        response = requests.get(
            f"{OFF_API_URL}/{barcode}.json",
            headers={"User-Agent": OFF_USER_AGENT},
            timeout=12,
        )
        response.raise_for_status()
        data = response.json()
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={"found": False, "message": f"Barcode lookup failed: {exc}"},
        )

    if data.get("status") != 1:
        return {"found": False, "message": "Product not found in the database"}

    product = data.get("product", {})
    nutriments = product.get("nutriments", {})

    cleaned = {
        "found": True,
        "barcode": barcode,
        "product_name": product.get("product_name") or "Unknown",
        "brands": product.get("brands") or "Unknown",
        "ingredients_text": product.get("ingredients_text") or "Not available",
        "nutriscore_grade": (product.get("nutriscore_grade") or "N/A").upper(),
        "nutrition_per_100g": {
            "energy": nutriments.get("energy-kcal_100g", nutriments.get("energy_100g")),
            "fat": nutriments.get("fat_100g"),
            "saturated_fat": nutriments.get("saturated-fat_100g"),
            "carbohydrates": nutriments.get("carbohydrates_100g"),
            "sugars": nutriments.get("sugars_100g"),
            "fiber": nutriments.get("fiber_100g"),
            "proteins": nutriments.get("proteins_100g"),
            "salt": nutriments.get("salt_100g"),
        },
    }
    return cleaned


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5050"))
    uvicorn.run(app, host="0.0.0.0", port=port, reload=False)
