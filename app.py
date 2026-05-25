import os
import requests
import random
import colorsys
import textwrap
import qrcode
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sklearn.cluster import KMeans
from io import BytesIO
import streamlit as st

# ── API KEY ───────────────────────────────────────────────────────────────────
api_key = st.secrets["GOOGLE_FONTS_API_KEY"]

# ── FONT LIST ─────────────────────────────────────────────────────────────────
font_list = [
    "Inter", "Montserrat", "Oswald", "Playfair Display", "Bodoni Moda",
    "EB Garamond", "Lora", "Pacifico", "Dancing Script", "Sacramento",
    "Pinyon Script", "Bebas Neue", "Abril Fatface"
]

# ── DOWNLOAD FONTS (cached — runs once per session, not every reload) ─────────
@st.cache_resource
def load_all_fonts():
    for font_name in font_list:
        filename = f"{font_name}.ttf"
        if os.path.exists(filename):
            continue
        try:
            encoded_name = font_name.replace(" ", "+")
            url = f"https://www.googleapis.com/webfonts/v1/webfonts?key={api_key}&family={encoded_name}"
            response = requests.get(url, timeout=10)
            data = response.json()
            if 'items' not in data or not data['items']:
                continue
            files = data['items'][0]['files']
            font_url = files.get('regular') or files.get('400') or list(files.values())[0]
            font_url = font_url.replace("http://", "https://")
            font_response = requests.get(font_url, timeout=10)
            with open(filename, 'wb') as f:
                f.write(font_response.content)
        except Exception as e:
            st.warning(f"Could not load font {font_name}: {e}")

load_all_fonts()

# ── COLOR HELPERS ─────────────────────────────────────────────────────────────
def is_dark(r, g, b):
    return (r + g + b) / 3 < 127.5

def get_warmth(r, g, b):
    return "warm" if (r - b) > 0 else "cool"

def get_luminance(r, g, b):
    def linearize(c):
        c = c / 255
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    return 0.2126 * linearize(r) + 0.7152 * linearize(g) + 0.0722 * linearize(b)

def get_contrast_ratio(l1, l2):
    lighter = max(l1, l2)
    darker = min(l1, l2)
    return (lighter + 0.05) / (darker + 0.05)

def get_dominant_color(image_path, k=5):
    img = Image.open(image_path).convert('RGB')
    img.thumbnail((200, 200))
    pixels = np.array(img).reshape(-1, 3)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(pixels)
    counts = np.bincount(kmeans.labels_)
    dominant_color = kmeans.cluster_centers_[np.argmax(counts)].astype(int)
    return tuple(int(x) for x in dominant_color)

def suggest_rgb(r, g, b):
    bg_lum = get_luminance(r, g, b)
    white_contrast = get_contrast_ratio(bg_lum, 1.0)
    black_contrast = get_contrast_ratio(bg_lum, 0.0)

    h, s, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
    h = (h + 0.5) % 1
    v = max(v, 0.8)
    r2, g2, b2 = colorsys.hsv_to_rgb(h, s, v)
    comp_lum = get_luminance(int(r2 * 255), int(g2 * 255), int(b2 * 255))
    comp_contrast = get_contrast_ratio(bg_lum, comp_lum)

    if comp_contrast >= 4.5:
        return (int(r2 * 255), int(g2 * 255), int(b2 * 255))
    elif white_contrast > black_contrast:
        return (255, 255, 255)
    else:
        return (0, 0, 0)

def pick_text_color(bg):
    bg_lum = get_luminance(bg[0], bg[1], bg[2])
    white_contrast = get_contrast_ratio(bg_lum, 1.0)
    black_contrast = get_contrast_ratio(bg_lum, 0.0)
    return (255, 255, 255) if white_contrast > black_contrast else (0, 0, 0)

def rgb_to_hex(r, g, b):
    return f"#{r:02x}{g:02x}{b:02x}".upper()

def rgb_to_cmyk(r, g, b):
    r, g, b = r / 255, g / 255, b / 255
    k = 1 - max(r, g, b)
    if k == 1:
        return (0, 0, 0, 100)
    c = (1 - r - k) / (1 - k)
    m = (1 - g - k) / (1 - k)
    y = (1 - b - k) / (1 - k)
    return (round(c * 100), round(m * 100), round(y * 100), round(k * 100))

# ── FACE DETECTION ────────────────────────────────────────────────────────────
def find_face(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    return faces

# ── TEXT EMOTION ──────────────────────────────────────────────────────────────
def analyze_text_emotion(text):
    text = text.lower()
    if any(w in text for w in ["sale", "now", "limited", "off", "free", "mega", "hurry", "today"]):
        return "urgent"
    elif any(w in text for w in ["unlock", "potential", "power", "achieve", "strength", "win", "champion"]):
        return "confident"
    elif any(w in text for w in ["love", "handcraft", "natural", "care", "gentle", "beauty", "since"]):
        return "warm"
    else:
        return "neutral"

# ── FONT STYLE ────────────────────────────────────────────────────────────────
def suggest_font_style(r, g, b, emotion="neutral"):
    dark = is_dark(r, g, b)
    warmth = get_warmth(r, g, b)
    if emotion == "urgent":
        return random.sample(["Bebas Neue", "Oswald", "Montserrat"], 3)
    elif emotion == "confident":
        return random.sample(["Inter", "Montserrat", "EB Garamond"], 3)
    elif emotion == "warm":
        return random.sample(["Pacifico", "Dancing Script", "Lora"], 3)
    else:
        if dark and warmth == "warm":
            return random.sample(["Bebas Neue", "Abril Fatface", "Montserrat"], 3)
        elif dark and warmth == "cool":
            return random.sample(["Inter", "Oswald", "EB Garamond"], 3)
        elif not dark and warmth == "warm":
            return random.sample(["Pacifico", "Dancing Script", "Sacramento", "Pinyon Script"], 3)
        else:
            return random.sample(["Playfair Display", "Bodoni Moda", "Lora"], 3)

# ── POSITION ──────────────────────────────────────────────────────────────────
def suggest_position(image_path, emotion="neutral"):
    img = Image.open(image_path).convert('RGB')
    width, height = img.size
    faces = find_face(image_path)
    best_section = None

    if len(faces) > 0:
        x, y, w, h = faces[0]
        face_center_x = x + w // 2
        face_center_y = y + h // 2
        if face_center_x < width // 2 and face_center_y < height // 2:
            face_section = "Top-Left"
        elif face_center_x >= width // 2 and face_center_y < height // 2:
            face_section = "Top-Right"
        elif face_center_x < width // 2 and face_center_y >= height // 2:
            face_section = "Bottom-Left"
        else:
            face_section = "Bottom-Right"
        if abs(face_center_x - width // 2) < width * 0.2 and \
           abs(face_center_y - height // 2) < height * 0.2:
            best_section = "Top-Left" if emotion != "warm" else "Bottom-Right"
        else:
            opposite = {
                'Top-Left': 'Bottom-Right', 'Top-Right': 'Bottom-Left',
                'Bottom-Left': 'Top-Right', 'Bottom-Right': 'Top-Left'
            }
            best_section = opposite[face_section]
    else:
        if emotion == "urgent":
            best_section = "Top-Left"
        elif emotion == "confident":
            best_section = "Top-Right"
        elif emotion == "warm":
            best_section = "Bottom-Left"
        else:
            sections = {
                'Top-Left':     img.crop((0, 0, width // 2, height // 2)),
                'Top-Right':    img.crop((width // 2, 0, width, height // 2)),
                'Bottom-Left':  img.crop((0, height // 2, width // 2, height)),
                'Bottom-Right': img.crop((width // 2, height // 2, width, height)),
            }
            lowest_std = 999999
            for section_name, section_img in sections.items():
                pixels = np.array(section_img).reshape(-1, 3)
                std = np.std(pixels)
                if std < lowest_std:
                    lowest_std = std
                    best_section = section_name
            if best_section is None:
                best_section = "Top-Left"

    if emotion == "urgent" and best_section.startswith("Bottom"):
        best_section = best_section.replace("Bottom", "Top")
    elif emotion == "warm" and best_section.startswith("Top"):
        best_section = best_section.replace("Top", "Bottom")
    return best_section

# ── GRID HELPER ───────────────────────────────────────────────────────────────
def grid_to_coords(grid_input, width, height):
    col = width // 3
    row = height // 3
    col_map = {"A": 0, "B": 1, "C": 2}
    row_map = {"1": 0, "2": 1, "3": 2}
    col_letter = grid_input[0].upper()
    row_number = grid_input[1]
    x = (col_map[col_letter] * col) + 50
    y = (row_map[row_number] * row) + 50
    return (x, y)

# ── RENDER TEXT ───────────────────────────────────────────────────────────────
def render_text_on_image(image_path, text, font_path=None, manual_pos=None):
    img = Image.open(image_path).convert('RGB')
    width, height = img.size
    bg = get_dominant_color(image_path)
    emotion = analyze_text_emotion(text)

    r, g, b = suggest_rgb(bg[0], bg[1], bg[2])
    text_fill_color = (r, g, b)

    font_size = max(20, width // 20)
    fonts = suggest_font_style(bg[0], bg[1], bg[2], emotion)
    selected_font = fonts[0]
    actual_font_path = f"{selected_font}.ttf"

    try:
        font = ImageFont.truetype(actual_font_path, font_size)
    except Exception as e:
        font = ImageFont.load_default()

    if manual_pos:
        x, y = grid_to_coords(manual_pos, width, height)
    else:
        position_name = suggest_position(image_path, emotion)
        pos_map = {
            'Top-Left':     (50, 50),
            'Top-Right':    (width // 2, 50),
            'Bottom-Left':  (50, height - (font_size * 4)),
            'Bottom-Right': (width // 2, height - (font_size * 4)),
        }
        x, y = pos_map[position_name]

    wrap_width = max(10, (width // font_size) - 2)
    wrapped_text = textwrap.fill(text, width=wrap_width)
    shadow_color = (255, 255, 255) if text_fill_color == (0, 0, 0) else (0, 0, 0)

    draw = ImageDraw.Draw(img)
    draw.text((x + 2, y + 2), wrapped_text, fill=shadow_color, font=font)
    draw.text((x, y), wrapped_text, fill=text_fill_color, font=font)

    output_path = "output_temp.jpg"
    img.save(output_path)
    return img, output_path, text_fill_color, actual_font_path, selected_font, emotion

# ── RENDER CTA ────────────────────────────────────────────────────────────────
def render_cta(img, cta_text, contact, text_fill_color, bg, selected_font, font_path=None, manual_pos=None):
    width, height = img.size
    draw = ImageDraw.Draw(img)
    cta_size = max(16, width // 18)
    ph_size = max(14, width // 22)

    if selected_font in ["Pacifico", "Dancing Script", "Sacramento", "Pinyon Script",
                         "Playfair Display", "Bodoni Moda", "EB Garamond", "Lora"]:
        cta_font_path = "Oswald.ttf"
    else:
        cta_font_path = "Playfair Display.ttf"

    try:
        cta_font = ImageFont.truetype(cta_font_path, cta_size)
    except:
        cta_font = ImageFont.load_default()
    try:
        ph_font = ImageFont.truetype(font_path, ph_size) if font_path else ImageFont.load_default()
    except:
        ph_font = ImageFont.load_default()

    # FIXED: use luminance-based contrast, not the old 6-arg version
    cta_color = pick_text_color(bg)
    shadow_color = (0, 0, 0) if cta_color == (255, 255, 255) else (255, 255, 255)

    if manual_pos:
        cta_x, cta_y = grid_to_coords(manual_pos, width, height)
    else:
        cta_x = width - (width // 3)
        cta_y = height - (cta_size * 3)

    ph_x = cta_x
    ph_y = cta_y + cta_size + 20

    draw.text((cta_x + 2, cta_y + 2), cta_text, fill=shadow_color, font=cta_font)
    draw.text((cta_x, cta_y), cta_text, fill=cta_color, font=cta_font)

    if contact and contact.startswith("http"):
        qr = qrcode.make(contact)
        qr = qr.resize((ph_size * 2, ph_size * 2))
        img.paste(qr, (ph_x, ph_y))
    elif contact:
        draw.text((ph_x, ph_y), contact, fill=cta_color, font=ph_font)

    return img

# ── LIVE REPOSITION HELPER ────────────────────────────────────────────────────
def render_live(mx, my, cx, cy, bg, text_fill_color, selected_font, font_path, text, cta_text, contact, image_bytes=None):
    if image_bytes:
        img2 = Image.open(BytesIO(image_bytes)).convert('RGB')
    else:
        img2 = Image.open("temp_image.jpg").convert('RGB')
    width2, height2 = img2.size

    new_main_x = int(width2 * mx / 100)
    new_main_y = int(height2 * my / 100)
    new_cta_x  = int(width2 * cx / 100)
    new_cta_y  = int(height2 * cy / 100)

    font_size2 = max(20, width2 // 20)
    cta_size   = max(16, width2 // 18)
    ph_size    = max(14, width2 // 22)

    try:
        font2 = ImageFont.truetype(f"{selected_font}.ttf", font_size2)
    except:
        font2 = ImageFont.load_default()

    if selected_font in ["Pacifico", "Dancing Script", "Sacramento", "Pinyon Script",
                         "Playfair Display", "Bodoni Moda", "EB Garamond", "Lora"]:
        cta_font_path = "Oswald.ttf"
    else:
        cta_font_path = "Playfair Display.ttf"

    try:
        cta_font = ImageFont.truetype(cta_font_path, cta_size)
    except:
        cta_font = ImageFont.load_default()
    try:
        ph_font = ImageFont.truetype(font_path, ph_size) if font_path else ImageFont.load_default()
    except:
        ph_font = ImageFont.load_default()

    wrap_width2 = max(10, (width2 // font_size2) - 2)
    wrapped2 = textwrap.fill(text, width=wrap_width2)
    shadow2 = (255, 255, 255) if text_fill_color == (0, 0, 0) else (0, 0, 0)

    draw = ImageDraw.Draw(img2)
    draw.text((new_main_x + 2, new_main_y + 2), wrapped2, fill=shadow2, font=font2)
    draw.text((new_main_x, new_main_y), wrapped2, fill=text_fill_color, font=font2)

    # FIXED: use luminance-based contrast, not the old 6-arg version
    cta_color = pick_text_color(bg)
    cta_shadow = (0, 0, 0) if cta_color == (255, 255, 255) else (255, 255, 255)

    draw.text((new_cta_x + 2, new_cta_y + 2), cta_text, fill=cta_shadow, font=cta_font)
    draw.text((new_cta_x, new_cta_y), cta_text, fill=cta_color, font=cta_font)

    ph_x = new_cta_x
    ph_y = new_cta_y + cta_size + 20
    if contact and contact.startswith("http"):
        qr = qrcode.make(contact)
        qr = qr.resize((ph_size * 2, ph_size * 2))
        img2.paste(qr, (ph_x, ph_y))
    elif contact:
        draw.text((ph_x, ph_y), contact, fill=cta_color, font=ph_font)

    return img2

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("Ad Maker")
st.write("Upload your photo and get science-backed design suggestions.")

uploaded_file = st.file_uploader("Upload your image", type=["jpg", "jpeg", "png", "webp"], key="upload")
text     = st.text_input("Enter main text:", key="main_text")
cta_text = st.text_input("Enter CTA text (e.g. Visit Us Today):", key="cta_text")
contact  = st.text_input("Enter phone number or website link:", key="contact")

if st.button("Generate Poster", key="generate"):
    if uploaded_file and text:
        image_bytes = uploaded_file.getbuffer().tobytes()
        with open("temp_image.jpg", "wb") as f:
            f.write(image_bytes)

        st.session_state["image_bytes"] = image_bytes

        img, output_path, text_fill_color, font_path, selected_font, emotion = render_text_on_image(
            "temp_image.jpg", text
        )
        bg = get_dominant_color("temp_image.jpg")
        fonts = suggest_font_style(bg[0], bg[1], bg[2], emotion)
        position = suggest_position("temp_image.jpg", emotion)

        if cta_text or contact:
            img = render_cta(img, cta_text, contact, text_fill_color, bg, selected_font, font_path)

        img.save("base_render.jpg")

        st.session_state["generated"]       = True
        st.session_state["bg"]              = bg
        st.session_state["emotion"]         = emotion
        st.session_state["fonts"]           = fonts
        st.session_state["selected_font"]   = selected_font
        st.session_state["position"]        = position
        st.session_state["text_fill_color"] = text_fill_color
        st.session_state["font_path"]       = font_path
    else:
        st.warning("Please upload an image and enter your main text.")

# ── RESULTS ───────────────────────────────────────────────────────────────────
if st.session_state.get("generated") and st.session_state.get("image_bytes"):
    bg            = st.session_state["bg"]
    emotion       = st.session_state["emotion"]
    fonts         = st.session_state["fonts"]
    selected_font = st.session_state["selected_font"]
    position      = st.session_state["position"]
    text_fill_color = st.session_state["text_fill_color"]
    font_path     = st.session_state["font_path"]
    image_bytes   = st.session_state["image_bytes"]
    text          = st.session_state.get("main_text", text)
    cta_text      = st.session_state.get("cta_text", cta_text)
    contact       = st.session_state.get("contact", contact)

    st.subheader("Adjust Positions")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Main Text Position**")
        main_x = st.slider("Main text — Left/Right", 0, 100, 10, key="main_x")
        main_y = st.slider("Main text — Up/Down",    0, 100, 10, key="main_y")
    with col2:
        st.write("**CTA Position**")
        cta_x_pct = st.slider("CTA — Left/Right", 0, 100, 70, key="cta_x")
        cta_y_pct = st.slider("CTA — Up/Down",    0, 100, 80, key="cta_y")

    live_img = render_live(
        main_x, main_y, cta_x_pct, cta_y_pct,
        bg, text_fill_color, selected_font, font_path,
        text, cta_text, contact, image_bytes=image_bytes
    )
    st.image(live_img, caption="Live Preview")

    hex_color = rgb_to_hex(text_fill_color[0], text_fill_color[1], text_fill_color[2])
    cmyk      = rgb_to_cmyk(text_fill_color[0], text_fill_color[1], text_fill_color[2])
    bg_hex    = rgb_to_hex(bg[0], bg[1], bg[2])
    bg_cmyk   = rgb_to_cmyk(bg[0], bg[1], bg[2])

    # write image to disk briefly for face detection
    with open("temp_image.jpg", "wb") as f:
        f.write(image_bytes)

    st.subheader("Why this design")
    st.info(f"Font — {selected_font} was chosen because your image is {'dark' if is_dark(bg[0], bg[1], bg[2]) else 'light'} and {get_warmth(bg[0], bg[1], bg[2])}, and your text emotion is {emotion}.")
    st.info(f"Color — font color {hex_color} was chosen using WCAG contrast math. It meets the AA standard for readability.")
    st.info(f"Position — text placed {position} based on {'face detection avoiding the subject' if len(find_face('temp_image.jpg')) > 0 else 'the flattest area of the image (least visual noise)'}.")

    st.subheader("Design Analysis")
    st.write(f"**Emotion detected:** {emotion}")
    st.write(f"**Text position:** {position}")
    st.write(f"**Font used:** {selected_font}")
    st.write(f"**Top 3 suggested fonts:** {', '.join(fonts)}")
    st.write(f"**Font color RGB:** {text_fill_color}  |  HEX: {hex_color}  |  CMYK: C={cmyk[0]} M={cmyk[1]} Y={cmyk[2]} K={cmyk[3]}")
    st.write(f"**Background RGB:** {bg}  |  HEX: {bg_hex}  |  CMYK: C={bg_cmyk[0]} M={bg_cmyk[1]} Y={bg_cmyk[2]} K={bg_cmyk[3]}")

    live_img.save("output_repositioned.jpg")
    with open("output_repositioned.jpg", "rb") as f:
        st.download_button("Download Poster", f, "poster.jpg", "image/jpeg", key="dl")
