import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image
import io

# Monkeypatch torch.load for PyTorch >=2.6 so that weights_only=False by default
import torch  # noqa: E402
if hasattr(torch, "load"):
    _torch_load_orig = torch.load  # type: ignore

    def _torch_load_patched(*args, **kwargs):  # type: ignore
        kwargs.setdefault("weights_only", False)
        return _torch_load_orig(*args, **kwargs)

    torch.load = _torch_load_patched  # type: ignore

# FastSAM imports
aTryFastSAM = True
try:
    from fastsam import FastSAM
except ModuleNotFoundError:
    aTryFastSAM = False
    st.error(
        "FastSAM package not found. Make sure you ran `pip install fastsam` and restart the app."
    )

MODEL_PATH_DEFAULT = "FastSAM-s.pt"  # change if you downloaded a different ckpt

def load_model(model_path: str | Path = MODEL_PATH_DEFAULT):
    """Cache and return a FastSAM model."""
    if not aTryFastSAM:
        return None

    @st.cache_resource(show_spinner=False)
    def _inner(path_str: str):
        device = "cuda" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu"
        # Workaround for Torch >=2.6 weights_only default
        try:
            from ultralytics.nn.tasks import SegmentationModel  # type: ignore
            import torch.serialization as _ts
            _ts.add_safe_globals([SegmentationModel])
        except Exception:
            pass  # if torch <2.6 or import fails, just proceed
        model = FastSAM(path_str)
        return model, device

    return _inner(str(model_path))


def segment_car(model, device: str, bgr_img: np.ndarray) -> np.ndarray:
    """Return a binary (0/255) mask of the largest object (assumed car)."""
    # Run Fast-SAM inference (returns list with dict per image)
    results = model(bgr_img, device=device, retina_masks=True)

    # Depending on FastSAM version, masks may be in different attrs
    masks = None
    if isinstance(results, list):
        item = results[0]
        masks = item.get("masks") if isinstance(item, dict) else getattr(item, "masks", None)
    
    if masks is None:
        raise RuntimeError("Fast-SAM did not return masks")

    # If masks is a list/tuple with variable shapes, pick largest manually
    if isinstance(masks, (list, tuple)):
        best = None
        max_pix = 0
        for m in masks:
            arr = np.array(m)
            pix = arr.sum()
            if pix > max_pix:
                max_pix = pix
                best = arr
        if best is None:
            raise RuntimeError("Masks list empty")
        return best.astype(np.uint8) * 255

    # Handle ultralytics.yolo.engine.results.Masks object
    if hasattr(masks, "data"):
        # Get the tensor data from the Masks object
        data = masks.data
        if hasattr(data, "cpu") and hasattr(data, "numpy"):
            # Convert torch tensor to numpy
            masks_array = data.cpu().numpy()
        else:
            # Already numpy array
            masks_array = np.array(data)
        
        # Handle the masks array
        if masks_array.ndim == 2:  # single mask HxW
            return masks_array.astype(np.uint8) * 255
        elif masks_array.ndim == 3:  # multiple masks (num_masks, H, W)
            pixel_counts = masks_array.reshape(masks_array.shape[0], -1).sum(axis=1)
            idx = int(np.argmax(pixel_counts))
            return masks_array[idx].astype(np.uint8) * 255
        else:
            raise RuntimeError(f"Unexpected masks array shape: {masks_array.shape}")

    # If masks is a 'Masks' object (from ultralytics), get its numpy array
    # Some versions have .numpy(), some have .data (torch.Tensor or np.ndarray)
    if hasattr(masks, "numpy") and callable(masks.numpy):
        masks = masks.numpy()
    # Now masks should be a numpy array
    if isinstance(masks, np.ndarray):
        if masks.ndim == 2:  # single mask HxW
            return masks.astype(np.uint8) * 255
        pixel_counts = masks.reshape(masks.shape[0], -1).sum(axis=1)
        idx = int(np.argmax(pixel_counts))
        return masks[idx].astype(np.uint8) * 255

    raise RuntimeError(f"Unrecognized mask format returned by Fast-SAM. Type: {type(masks)}")


def overlay_sticker(
    car_bgr: np.ndarray, car_mask: np.ndarray, sticker_bgra: np.ndarray
) -> np.ndarray:
    """Resize sticker to car mask bounding-box and composite it over the car."""
    ys, xs = np.where(car_mask == 255)
    if len(xs) == 0 or len(ys) == 0:
        raise RuntimeError("Car mask appears empty.")

    x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()
    w, h = x1 - x0 + 1, y1 - y0 + 1

    # Resize sticker
    sticker_resized = cv2.resize(sticker_bgra, (w, h), interpolation=cv2.INTER_AREA)

    # Split channels
    if sticker_resized.shape[2] == 4:
        sticker_bgr = sticker_resized[:, :, :3]
        alpha = sticker_resized[:, :, 3].astype(np.float32) / 255.0
    else:
        sticker_bgr = sticker_resized
        alpha = np.ones((h, w), dtype=np.float32)

    # Get the car region
    car_sub = car_mask[y0 : y1 + 1, x0 : x1 + 1].astype(np.float32) / 255.0
    roi = car_bgr[y0 : y1 + 1, x0 : x1 + 1].astype(np.float32)

    # Create the final alpha mask (sticker alpha * car mask)
    # Expand alpha to 3D to match BGR channels: (h, w) -> (h, w, 3)
    final_alpha = (alpha * car_sub)[:, :, np.newaxis]

    # Blend the sticker over the car using alpha compositing
    # result = background * (1 - alpha) + foreground * alpha
    blended = roi * (1 - final_alpha) + sticker_bgr.astype(np.float32) * final_alpha

    # Copy back to the result
    result = car_bgr.copy()
    result[y0 : y1 + 1, x0 : x1 + 1] = blended.astype(np.uint8)
    return result


def np_from_uploaded(uploaded) -> np.ndarray | None:
    """Convert a Streamlit UploadedFile to a BGRA OpenCV image."""
    if uploaded is None:
        return None
    # Read bytes and open with PIL from an in-memory buffer
    data = uploaded.read()
    try:
        pil_img = Image.open(io.BytesIO(data)).convert("RGBA")
    except Exception:
        return None
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGBA2BGRA)


st.set_page_config(page_title="Virtual Car Wrap", page_icon="🚗")
st.title("🚗 Virtual Car Wrap")

st.markdown(
    "Upload a **car photo** and a **sticker/decal image (PNG recommended)**.\n"
    "The app segments the car with Fast-SAM and overlays the sticker so you can preview a virtual wrap."
)

with st.sidebar:
    st.header("Model checkpoint")
    model_path = st.text_input("Path to FastSAM checkpoint (*.pt)", value=MODEL_PATH_DEFAULT)

car_file = st.file_uploader("Car image", type=["jpg", "jpeg", "png"], key="car")
sticker_file = st.file_uploader("Sticker / decal image", type=["png", "jpg", "jpeg"], key="sticker")

run_btn = st.button("Wrap it!")

if run_btn:
    if car_file is None or sticker_file is None:
        st.warning("Please upload both a car image and a sticker image.")
        st.stop()

    car_img = np_from_uploaded(car_file)  # BGRA
    if car_img is None:
        st.error("Could not read car image.")
        st.stop()

    if car_img.shape[2] == 4:
        car_img = car_img[:, :, :3]  # discard alpha if any

    sticker_img = np_from_uploaded(sticker_file)
    if sticker_img is None:
        st.error("Could not read sticker image.")
        st.stop()

    try:
        model_tup = load_model(model_path)
        if model_tup is None:
            st.stop()
        model, device = model_tup

        with st.spinner("Segmenting car…"):
            car_mask = segment_car(model, device, car_img)

        with st.spinner("Overlaying sticker…"):
            wrapped = overlay_sticker(car_img, car_mask, sticker_img)

        st.subheader("Result")
        st.image(cv2.cvtColor(wrapped, cv2.COLOR_BGR2RGB), use_column_width=True)

        # Save to temp file and offer download
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        cv2.imwrite(tmp.name, wrapped)
        with open(tmp.name, "rb") as f:
            st.download_button(
                label="Download wrapped image",
                data=f.read(),
                file_name="wrapped_car.png",
                mime="image/png",
            )
    except Exception as e:
        st.exception(e)
