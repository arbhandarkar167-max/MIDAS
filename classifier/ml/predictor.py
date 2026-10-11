import json
import os
import threading

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from django.conf import settings

from .model import OptimizedDermaFusion


MODEL_PATH = getattr(
    settings,
    "MIDAS_MODEL_PATH",
    os.path.join(
        settings.BASE_DIR,
        "ml_models",
        "optimized_fusion_best.pth",
    ),
)

LABEL_MAP_PATH = getattr(
    settings,
    "MIDAS_LABEL_MAP_PATH",
    os.path.join(
        settings.BASE_DIR,
        "ml_models",
        "label_map.json",
    ),
)


# =========================================================
# Encodings
# =========================================================

GENDER_MAP = {
    "male": 0,
    "female": 1,
}

FITZPATRICK_MAP = {
    "i pale white skin, blue/green eyes, blond/red hair": 1,
    "ii fair skin, blue eyes": 2,
    "iii darker white skin": 3,
    "iv light brown skin": 4,
    "v brown skin": 5,
    "vi dark brown or black skin": 6,
}

LOCATION_MAP = {
    "head_neck": 0,
    "lower_extremity": 1,
    "other": 2,
    "trunk": 3,
    "upper_extremity": 4,
}

DISTANCE_MAP = {
    "6in": 0,
    "1ft": 1,
    "dscope": 2,
    "n/a - virtual": 3,
}

RACE_MAP = {
    "asian": 0,
    "black or african american": 1,
    "other": 2,
    "unknown": 3,
    "white": 4,
}


# =========================================================
# Training normalization
# =========================================================

META_MEAN = np.array(
    [
        55.2356687898,
        0.3656050955,
        2.2140127389,
        2.2216560510,
        1.0853503185,
        0.1299363057,
        0.3031847134,
        6.7363057325,
        5.3713375796,
        46.7589171975,
        3.5184713376,
        0.5681528662,
        1.2985816663,
        0.6305732484,
        0.3528662420,
        0.8229299363,
        0.6407643312,
        0.3057324841,
    ],
    dtype=np.float32,
)

META_STD = np.array(
    [
        16.5588545766,
        0.4819064732,
        0.6701030418,
        1.2502203108,
        0.8987530539,
        0.3364477110,
        0.4599273986,
        3.6346924060,
        3.6520376433,
        87.7838703426,
        1.1316313432,
        0.4956492105,
        0.3263968753,
        0.4829573042,
        0.4781662076,
        0.3819713586,
        0.4800822918,
        0.4610107073,
    ],
    dtype=np.float32,
)

META_STD[META_STD == 0] = 1.0


# =========================================================
# Image preprocessing
# =========================================================

IMAGE_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[
                0.485,
                0.456,
                0.406,
            ],
            std=[
                0.229,
                0.224,
                0.225,
            ],
        ),
    ]
)


# =========================================================
# Model
# =========================================================

_model = None
_model_lock = threading.Lock()


def _load_label_map():

    if os.path.exists(LABEL_MAP_PATH):

        with open(
            LABEL_MAP_PATH,
            "r",
            encoding="utf-8",
        ) as file:

            data = json.load(file)

        return {
            int(index): label
            for index, label
            in data["idx2label"].items()
        }

    return {
        0: "melanocytic_nevus",
        1: "melanoma",
    }


def _load_model():

    global _model

    if _model is not None:
        return _model

    with _model_lock:

        if _model is not None:
            return _model

        if not os.path.exists(MODEL_PATH):

            raise FileNotFoundError(
                f"DermFusion checkpoint not found: "
                f"{MODEL_PATH}"
            )

        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        model = OptimizedDermaFusion(
            num_meta=18,
        )

        checkpoint = torch.load(
            MODEL_PATH,
            map_location=device,
            weights_only=False,
        )

        if (
            not isinstance(checkpoint, dict)
            or "model_state" not in checkpoint
        ):
            raise RuntimeError(
                "Invalid DermFusion checkpoint format."
            )

        model.load_state_dict(
            checkpoint["model_state"],
            strict=True,
        )

        model.to(device)
        model.eval()

        _model = model

    return _model


# =========================================================
# Metadata
# =========================================================

def _build_metadata(data):

    age = float(
        data["age"]
    )

    gender_enc = GENDER_MAP[
        data["gender"]
    ]

    fitzpatrick = FITZPATRICK_MAP[
        data["fitzpatrick"]
    ]

    location_enc = LOCATION_MAP[
        data["location"]
    ]

    distance_enc = DISTANCE_MAP[
        data["distance"]
    ]

    is_control = (
        1.0
        if data.get("is_control")
        else 0.0
    )

    melanoma = (
        1.0
        if data["melanoma_history"] == "yes"
        else 0.0
    )

    lesion_length = float(
        data["length"]
    )

    lesion_width = float(
        data["width"]
    )

    lesion_area = (
        lesion_length *
        lesion_width
    )

    race_enc = RACE_MAP[
        data["race"]
    ]

    # Engineered features
    diameter_gt_6mm = (
        1.0
        if lesion_length >= 6.0
        else 0.0
    )

    aspect_ratio = (
        lesion_length /
        (lesion_width + 1e-3)
    )

    age_gt_50 = (
        1.0
        if age >= 50
        else 0.0
    )

    age_gt_65 = (
        1.0
        if age >= 65
        else 0.0
    )

    fair_skin = (
        1.0
        if fitzpatrick in [1, 2]
        else 0.0
    )

    high_risk_site = (
        1.0
        if location_enc in [0, 3]
        else 0.0
    )

    elderly_male = (
        1.0
        if age >= 60
        and gender_enc == 0
        else 0.0
    )

    raw_values = np.array(
        [
            age,
            gender_enc,
            fitzpatrick,
            location_enc,
            distance_enc,
            is_control,
            melanoma,
            lesion_length,
            lesion_width,
            lesion_area,
            race_enc,
            diameter_gt_6mm,
            aspect_ratio,
            age_gt_50,
            age_gt_65,
            fair_skin,
            high_risk_site,
            elderly_male,
        ],
        dtype=np.float32,
    )

    normalized = (
        raw_values - META_MEAN
    ) / META_STD

    return torch.tensor(
        normalized,
        dtype=torch.float32,
    ).unsqueeze(0)


# =========================================================
# Grad-CAM heatmap
# =========================================================

def _create_gradcam(
    original_image,
    image_tensor,
    metadata_tensor,
    model,
    target_class,
):

    activations = {}

    # -----------------------------------------------------
    # Target layer
    #
    # Final convolutional block of EfficientNet-B0.
    # -----------------------------------------------------

    target_layer = model.features[-1]

    def forward_hook(module, inputs, output):

        activations["value"] = output

        # Keep gradient on the activation tensor.
        output.retain_grad()

    hook_handle = target_layer.register_forward_hook(
        forward_hook
    )

    try:

        model.zero_grad(set_to_none=True)

        logits = model(
            image_tensor,
            metadata_tensor,
        )

        target_logit = logits[
            0,
            target_class,
        ]

        target_logit.backward()

        activation = activations["value"]
        gradient = activation.grad

        if gradient is None:
            raise RuntimeError(
                "Could not obtain Grad-CAM gradients."
            )

        activation = activation[0]
        gradient = gradient[0]

        # Global-average-pool gradients.
        weights = gradient.mean(
            dim=(1, 2),
        )

        # Weighted sum of feature maps.
        cam = (
            weights[:, None, None]
            * activation
        ).sum(dim=0)

        # Keep only positive contributions.
        cam = F.relu(cam)

        # Normalize.
        cam_min = cam.min()
        cam_max = cam.max()

        cam = (
            cam - cam_min
        ) / (
            cam_max - cam_min + 1e-8
        )

        # Upsample to original image size.
        cam = F.interpolate(
            cam.unsqueeze(0).unsqueeze(0),
            size=(
                original_image.height,
                original_image.width,
            ),
            mode="bilinear",
            align_corners=False,
        )

        cam = cam[
            0,
            0,
        ].detach().cpu().numpy()

        cam = np.clip(
            cam,
            0.0,
            1.0,
        )

        # -------------------------------------------------
        # Create a heatmap without extra dependencies.
        # -------------------------------------------------

        red = np.clip(
            2.0 * cam,
            0.0,
            1.0,
        )

        green = np.clip(
            2.0 * (
                1.0 - np.abs(cam - 0.5)
            ),
            0.0,
            1.0,
        )

        blue = np.clip(
            2.0 * (
                1.0 - cam
            ),
            0.0,
            1.0,
        )

        heatmap_array = np.stack(
            [
                red,
                green,
                blue,
            ],
            axis=-1,
        )

        heatmap_array = (
            heatmap_array * 255
        ).astype(
            np.uint8
        )

        heatmap = Image.fromarray(
            heatmap_array,
            mode="RGB",
        )

        # -------------------------------------------------
        # Overlay
        # -------------------------------------------------

        original = original_image.convert(
            "RGB"
        )

        overlay = Image.blend(
            original,
            heatmap,
            alpha=0.45,
        )

        return overlay

    finally:

        hook_handle.remove()

        model.zero_grad(
            set_to_none=True
        )


# =========================================================
# Main prediction
# =========================================================

def predict(
    image_file,
    metadata,
):

    model = _load_model()
    label_map = _load_label_map()

    device = next(
        model.parameters()
    ).device

    # -----------------------------------------------------
    # Read original image
    # -----------------------------------------------------

    if hasattr(
        image_file,
        "seek",
    ):
        image_file.seek(0)

    original_image = Image.open(
        image_file
    ).convert("RGB")

    # -----------------------------------------------------
    # Image tensor
    # -----------------------------------------------------

    image_tensor = IMAGE_TRANSFORM(
        original_image
    ).unsqueeze(0)

    # -----------------------------------------------------
    # Metadata tensor
    # -----------------------------------------------------

    metadata_tensor = _build_metadata(
        metadata
    )

    image_tensor = image_tensor.to(
        device
    )

    metadata_tensor = metadata_tensor.to(
        device
    )

    # -----------------------------------------------------
    # Classification
    # -----------------------------------------------------

    model.eval()

    with torch.enable_grad():

        logits = model(
            image_tensor,
            metadata_tensor,
        )

        probabilities = torch.softmax(
            logits.detach(),
            dim=1,
        )[0]

        predicted_index = int(
            torch.argmax(
                probabilities
            ).item()
        )

    confidence = float(
        probabilities[
            predicted_index
        ].item()
    )

    predicted_class = label_map[
        predicted_index
    ]

    probability_dict = {
        label_map[index]: float(
            probabilities[index].item()
        )
        for index in range(
            len(label_map)
        )
    }

    # -----------------------------------------------------
    # Grad-CAM
    # -----------------------------------------------------

    gradcam_overlay = _create_gradcam(
        original_image=original_image,
        image_tensor=image_tensor,
        metadata_tensor=metadata_tensor,
        model=model,
        target_class=predicted_index,
    )

    return {
        "class_index": predicted_index,
        "predicted_class": predicted_class,
        "confidence": confidence,
        "probabilities": probability_dict,
        "gradcam_overlay": gradcam_overlay,
    }