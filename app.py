import tempfile
from pathlib import Path
import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter
import io
import random

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


def improve_car_mask(car_mask: np.ndarray, car_img: np.ndarray) -> np.ndarray:
    """Improve the car mask by removing noise and focusing on the car body."""
    # Convert to binary
    mask = (car_mask > 127).astype(np.uint8)
    
    # Morphological operations to clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Find contours and keep only the largest one (the car)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        mask = np.zeros_like(mask)
        cv2.fillPoly(mask, [largest_contour], (255,))
    
    # Apply some erosion to focus on the car body (avoid windows, etc.)
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.erode(mask, kernel_small, iterations=1)
    
    return mask


def create_professional_background(width: int, height: int, style: str = "studio") -> np.ndarray:
    """Create professional backgrounds for car photography."""
    if style == "studio":
        # Clean studio background with gradient
        background = np.zeros((height, width, 3), dtype=np.uint8)
        for i in range(height):
            intensity = int(255 * (1 - i / height * 0.3))
            background[i, :] = [intensity, intensity, intensity]
    
    elif style == "outdoor":
        # Outdoor scene with sky gradient
        background = np.zeros((height, width, 3), dtype=np.uint8)
        for i in range(height):
            # Sky gradient from blue to light blue
            blue_intensity = int(200 - (i / height) * 100)
            green_intensity = int(220 - (i / height) * 80)
            red_intensity = int(240 - (i / height) * 60)
            background[i, :] = [blue_intensity, green_intensity, red_intensity]
    
    elif style == "garage":
        # Garage/industrial background
        background = np.ones((height, width, 3), dtype=np.uint8) * 50
        # Add some texture
        for i in range(0, height, 20):
            background[i:i+2, :] = [30, 30, 30]
    
    else:  # neutral
        background = np.ones((height, width, 3), dtype=np.uint8) * 128
    
    return background


def enhance_lighting_and_shadows(car_img: np.ndarray, car_mask: np.ndarray) -> np.ndarray:
    """Enhance lighting and add realistic shadows to the car."""
    result = car_img.copy()
    
    # Create a shadow mask (dilated car mask)
    shadow_mask = cv2.dilate(car_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    shadow_mask = cv2.GaussianBlur(shadow_mask, (21, 21), 0)
    
    # Add shadow to the background
    shadow_intensity = 0.3
    shadow_region = (shadow_mask > 0) & (car_mask == 0)
    result[shadow_region] = (result[shadow_region] * (1 - shadow_intensity)).astype(np.uint8)
    
    # Enhance car lighting
    car_region = (car_mask > 127)
    if np.any(car_region):
        # Increase contrast and brightness slightly
        car_pixels = result[car_region].astype(np.float32)
        car_pixels = car_pixels * 1.1 + 10  # Brighten and add contrast
        result[car_region] = np.clip(car_pixels, 0, 255).astype(np.uint8)
    
    return result


def add_car_paint_texture(car_img: np.ndarray, car_mask: np.ndarray, texture_type: str = "metallic") -> np.ndarray:
    """Add realistic car paint textures and reflections."""
    result = car_img.copy()
    car_region = (car_mask > 127)
    
    if not np.any(car_region):
        return result
    
    height, width = car_img.shape[:2]
    
    if texture_type == "metallic":
        # Create metallic reflection pattern
        reflection = np.zeros((height, width), dtype=np.float32)
        for i in range(height):
            for j in range(width):
                reflection[i, j] = 0.5 + 0.3 * np.sin(i * 0.02) * np.cos(j * 0.03)
        
        # Apply reflection to car
        for c in range(3):
            result[car_region, c] = np.clip(
                result[car_region, c].astype(np.float32) * (1 + reflection[car_region] * 0.2),
                0, 255
            ).astype(np.uint8)
    
    elif texture_type == "pearl":
        # Create pearl effect
        pearl = np.random.rand(height, width).astype(np.float32) * 0.1
        pearl = cv2.GaussianBlur(pearl, (5, 5), 0)
        
        for c in range(3):
            result[car_region, c] = np.clip(
                result[car_region, c].astype(np.float32) * (1 + pearl[car_region]),
                0, 255
            ).astype(np.uint8)
    
    elif texture_type == "matte":
        # Create matte finish
        noise = np.random.rand(height, width).astype(np.float32) * 0.05
        noise = cv2.GaussianBlur(noise, (3, 3), 0)
        
        for c in range(3):
            result[car_region, c] = np.clip(
                result[car_region, c].astype(np.float32) * (1 - noise[car_region] * 0.3),
                0, 255
            ).astype(np.uint8)
    
    return result


def apply_professional_filters(car_img: np.ndarray, filter_type: str = "none") -> np.ndarray:
    """Apply professional photography filters to enhance the image."""
    if filter_type == "none":
        return car_img
    
    # Convert to PIL for easier filter application
    pil_img = Image.fromarray(cv2.cvtColor(car_img, cv2.COLOR_BGR2RGB))
    
    if filter_type == "warm":
        # Warm filter
        enhancer = ImageEnhance.Color(pil_img)
        pil_img = enhancer.enhance(1.2)
        enhancer = ImageEnhance.Brightness(pil_img)
        pil_img = enhancer.enhance(1.1)
    
    elif filter_type == "cool":
        # Cool filter
        enhancer = ImageEnhance.Color(pil_img)
        pil_img = enhancer.enhance(0.8)
        enhancer = ImageEnhance.Contrast(pil_img)
        pil_img = enhancer.enhance(1.2)
    
    elif filter_type == "dramatic":
        # Dramatic filter
        enhancer = ImageEnhance.Contrast(pil_img)
        pil_img = enhancer.enhance(1.4)
        enhancer = ImageEnhance.Brightness(pil_img)
        pil_img = enhancer.enhance(0.9)
    
    elif filter_type == "vintage":
        # Vintage filter
        enhancer = ImageEnhance.Color(pil_img)
        pil_img = enhancer.enhance(0.7)
        enhancer = ImageEnhance.Brightness(pil_img)
        pil_img = enhancer.enhance(1.2)
    
    # Convert back to OpenCV format
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def change_car_color(car_img: np.ndarray, car_mask: np.ndarray, target_color: tuple) -> np.ndarray:
    """Change the color of the car using AI-powered color transfer."""
    # Convert to HSV for better color manipulation
    hsv = cv2.cvtColor(car_img, cv2.COLOR_BGR2HSV)
    
    # Create a copy for modification
    result = car_img.copy()
    
    # Get the car region
    car_region = (car_mask > 127)
    
    # Calculate the current average color of the car
    car_pixels = car_img[car_region]
    if len(car_pixels) > 0:
        current_avg = np.mean(car_pixels, axis=0)
        
        # Convert target color to BGR if it's RGB
        if len(target_color) == 3:
            target_bgr = target_color[::-1]  # RGB to BGR
        else:
            target_bgr = target_color
        
        # Calculate color difference
        color_diff = np.array(target_bgr, dtype=np.float32) - current_avg
        
        # Apply color change with smooth blending
        for i in range(3):  # BGR channels
            result[car_region, i] = np.clip(
                result[car_region, i].astype(np.float32) + color_diff[i] * 0.7, 
                0, 255
            ).astype(np.uint8)
    
    return result


def get_largest_contour(mask: np.ndarray):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def get_car_side_quad(contour):
    # Approximate the contour to a quadrilateral (side view)
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    if len(approx) == 4:
        return approx.reshape(4, 2)
    # If not 4, use bounding rect corners
    x, y, w, h = cv2.boundingRect(contour)
    return np.array([[x, y], [x+w, y], [x+w, y+h], [x, y+h]], dtype=np.float32)


def refine_body_mask(car_img: np.ndarray, car_mask: np.ndarray) -> np.ndarray:
    """Refine the car mask to focus on the painted body (exclude windows, wheels, background)."""
    mask = (car_mask > 127).astype(np.uint8)
    # Remove small objects (wheels, mirrors)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    # Exclude dark regions (windows, tires)
    hsv = cv2.cvtColor(car_img, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    dark = (v < 60).astype(np.uint8)
    mask[dark == 1] = 0
    # Exclude very bright regions (reflections, glass)
    bright = (v > 230).astype(np.uint8)
    mask[bright == 1] = 0
    # Fill holes and keep largest area (main body)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        mask = np.zeros_like(mask)
        cv2.drawContours(mask, [largest], -1, (255,), -1)
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)))
    return mask


def warp_and_blend_wrap(
    car_img: np.ndarray,
    car_mask: np.ndarray,
    wrap_img: np.ndarray,
    position_x: float,
    position_y: float,
    scale: float,
    rotation: float
) -> np.ndarray:
    """Warp the wrap image to the car's side and blend it realistically, using the refined mask."""
    result = car_img.copy()
    mask = (car_mask > 127).astype(np.uint8)
    contour = get_largest_contour(mask)
    if contour is None:
        return result
    quad = get_car_side_quad(contour)
    quad = order_points(quad)
    h, w = wrap_img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), rotation, scale)
    wrap_transformed = cv2.warpAffine(wrap_img, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=(0,0,0,0))
    # Position offset (now both x and y)
    dx = int((position_x-0.5) * w * 0.5)
    dy = int((position_y-0.5) * h * 0.5)
    src = np.array([[0+dx,0+dy],[w+dx,0+dy],[w+dx,h+dy],[0+dx,h+dy]], dtype=np.float32)
    dst = quad.astype(np.float32)
    matrix = cv2.getPerspectiveTransform(src, dst)
    wrap_warped = cv2.warpPerspective(wrap_transformed, matrix, (car_img.shape[1], car_img.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)
    # Use refined mask for blending
    refined_mask = refine_body_mask(car_img, car_mask)
    if wrap_warped.shape[2] == 4:
        alpha = (wrap_warped[:,:,3].astype(np.float32)/255.0) * (refined_mask.astype(np.float32)/255.0)
        for c in range(3):
            result[:,:,c] = (wrap_warped[:,:,c]*alpha + result[:,:,c]*(1-alpha)).astype(np.uint8)
    else:
        mask = ((wrap_warped.sum(axis=2)>0) & (refined_mask>0)).astype(np.uint8)
        for c in range(3):
            result[:,:,c] = np.where(mask, wrap_warped[:,:,c], result[:,:,c])
    return result


def order_points(pts):
    # Order points: top-left, top-right, bottom-right, bottom-left
    rect = np.zeros((4, 2), dtype = "float32")
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


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


def is_likely_logo_or_invalid_car(img: np.ndarray) -> str | None:
    """Heuristic checks to warn if the uploaded car image is not a real car photo."""
    h, w = img.shape[:2]
    if h < 200 or w < 200:
        return "The uploaded car image is very small. Please use a larger car photo."
    # If image has alpha channel, check transparency
    if img.shape[2] == 4:
        alpha = img[:, :, 3]
        non_transparent = np.count_nonzero(alpha > 32)
        if non_transparent < 0.5 * h * w:
            return "The uploaded car image is mostly transparent. Please upload a real car photo, not a logo or sticker."
    # Check color variance (logos are often flat color)
    if np.std(img[:, :, :3]) < 15:
        return "The uploaded car image has very low color variance. Please upload a real car photo, not a logo or sticker."
    return None


def apply_body_mask_to_sticker(sticker_img: np.ndarray, body_mask: np.ndarray) -> np.ndarray:
    # Resize mask to sticker size if needed
    mask_resized = cv2.resize(body_mask, (sticker_img.shape[1], sticker_img.shape[0]), interpolation=cv2.INTER_NEAREST)
    # Ensure mask is binary
    mask_bin = (mask_resized > 127).astype(np.uint8)
    # If sticker has no alpha, add one
    if sticker_img.shape[2] == 3:
        sticker_img = cv2.cvtColor(sticker_img, cv2.COLOR_BGR2BGRA)
    # Apply mask to alpha channel
    sticker_img = sticker_img.copy()
    sticker_img[:, :, 3] = sticker_img[:, :, 3] * mask_bin
    return sticker_img


st.set_page_config(page_title="Virtual Car Wrap", page_icon="🚗")
st.title("🚗 Virtual Car Wrap - Professional Edition")

st.markdown(
    "Upload a **car photo** and a **sticker/decal image (PNG recommended)**.\n"
    "The app uses AI to create professional-quality virtual car wraps with advanced styling options."
)

with st.sidebar:
    st.header("Model checkpoint")
    model_path = st.text_input("Path to FastSAM checkpoint (*.pt)", value=MODEL_PATH_DEFAULT)
    
    st.header("🎨 Car Color Options")
    st.markdown("Change the car color before applying the wrap:")
    
    # Color picker for car color change
    car_color = st.color_picker("New Car Color", "#FF0000", help="Choose a new color for the car")
    
    # Convert hex to RGB
    hex_color = car_color.lstrip('#')
    rgb_color = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    st.header("🎭 Paint Texture")
    paint_texture = st.selectbox(
        "Car Paint Finish",
        ["none", "metallic", "pearl", "matte"],
        help="Add realistic paint textures and reflections"
    )
    
    st.header("🌅 Background Style")
    background_style = st.selectbox(
        "Professional Background",
        ["none", "studio", "outdoor", "garage"],
        help="Replace background with professional photography settings"
    )
    
    st.header("💡 Lighting Enhancement")
    enhance_lighting = st.checkbox("Enhance lighting & shadows", value=True, help="Add realistic lighting and shadows")
    
    st.header("📸 Professional Filters")
    photo_filter = st.selectbox(
        "Photography Filter",
        ["none", "warm", "cool", "dramatic", "vintage"],
        help="Apply professional photography filters"
    )
    
    st.header("⚙️ Processing Options")
    improve_mask = st.checkbox("Improve car mask", value=True, help="Clean up the car segmentation for better results")

    # --- Wrap controls ---
    st.sidebar.header("🛠️ Wrap Placement Controls")
    wrap_position_x = st.sidebar.slider("Horizontal Position", 0.0, 1.0, 0.5, 0.01, help="Move the wrap left/right")
    wrap_position_y = st.sidebar.slider("Vertical Position", 0.0, 1.0, 0.5, 0.01, help="Move the wrap up/down")
    wrap_scale = st.sidebar.slider("Scale", 0.2, 2.0, 1.0, 0.01, help="Resize the wrap")
    wrap_rotation = st.sidebar.slider("Rotation", -45.0, 45.0, 0.0, 0.5, help="Rotate the wrap (degrees)")

car_file = st.file_uploader("Car image", type=["jpg", "jpeg", "png"], key="car")
sticker_file = st.file_uploader("Sticker / decal image", type=["png", "jpg", "jpeg"], key="sticker")

run_btn = st.button("🚀 Create Professional Wrap!")

if run_btn:
    if car_file is None or sticker_file is None:
        st.warning("Please upload both a car image and a sticker image.")
        st.stop()

    car_img = np_from_uploaded(car_file)  # BGRA
    if car_img is None:
        st.error("Could not read car image.")
        st.stop()

    # Input validation for car image
    logo_warn = is_likely_logo_or_invalid_car(car_img)
    if logo_warn:
        st.warning(logo_warn)
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

        with st.spinner("🤖 AI-powered car segmentation…"):
            car_mask = segment_car(model, device, car_img)
            
            if improve_mask:
                car_mask = improve_car_mask(car_mask, car_img)

        # Show original image and mask
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Car")
            st.image(cv2.cvtColor(car_img, cv2.COLOR_BGR2RGB), use_container_width=True)
        
        with col2:
            st.subheader("AI Car Mask")
            st.image(car_mask, use_container_width=True)

        # Apply color change if requested
        if car_color != "#FF0000":  # Default color
            with st.spinner("🎨 Changing car color…"):
                car_img_colored = change_car_color(car_img, car_mask, rgb_color)
        else:
            car_img_colored = car_img

        # Apply paint texture
        if paint_texture != "none":
            with st.spinner("✨ Adding paint texture…"):
                car_img_colored = add_car_paint_texture(car_img_colored, car_mask, paint_texture)

        # Apply background replacement
        if background_style != "none":
            with st.spinner("🌅 Creating professional background…"):
                height, width = car_img_colored.shape[:2]
                background = create_professional_background(width, height, background_style)
                
                # Composite car over background
                car_region = (car_mask > 127)
                result = background.copy()
                result[car_region] = car_img_colored[car_region]
                car_img_colored = result

        # Apply lighting enhancement
        if enhance_lighting:
            with st.spinner("💡 Enhancing lighting and shadows…"):
                car_img_colored = enhance_lighting_and_shadows(car_img_colored, car_mask)

        # Apply professional filters
        if photo_filter != "none":
            with st.spinner("📸 Applying professional filter…"):
                car_img_colored = apply_professional_filters(car_img_colored, photo_filter)

        with st.spinner("🎯 Applying sticker overlay…"):
            # Refine the body mask for sticker masking
            refined_body_mask = refine_body_mask(car_img_colored, car_mask)
            sticker_img_masked = apply_body_mask_to_sticker(sticker_img, refined_body_mask)
            wrapped = warp_and_blend_wrap(car_img_colored, car_mask, sticker_img_masked, wrap_position_x, wrap_position_y, wrap_scale, wrap_rotation)

        st.subheader("🎉 Professional Result")
        st.image(cv2.cvtColor(wrapped, cv2.COLOR_BGR2RGB), use_container_width=True)

        # Save to temp file and offer download
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        cv2.imwrite(tmp.name, wrapped)
        with open(tmp.name, "rb") as f:
            st.download_button(
                label="📥 Download Professional Wrap",
                data=f.read(),
                file_name="professional_car_wrap.png",
                mime="image/png",
            )
    except Exception as e:
        st.exception(e)
