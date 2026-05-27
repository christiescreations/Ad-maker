import os
import requests
import random
import colorsys
import textwrap
import qrcode
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
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

# ── DOWNLOAD FONTS (cached — runs once per session) ───────────────────────────
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
    darker  = min(l1, l2)
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
    bg_lum         = get_luminance(r, g, b)
    white_contrast = get_contrast_ratio(bg_lum, 1.0)
    black_contrast = get_contrast_ratio(bg_lum, 0.0)

    h, s, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
    h = (h + 0.5) % 1
    v = max(v, 0.8)
    r2, g2, b2   = colorsys.hsv_to_rgb(h, s, v)
    comp_lum     = get_luminance(int(r2 * 255), int(g2 * 255), int(b2 * 255))
    comp_contrast = get_contrast_ratio(bg_lum, comp_lum)

    if comp_contrast >= 4.5:
        return (int(r2 * 255), int(g2 * 255), int(b2 * 255))
    elif white_contrast > black_contrast:
        return (255, 255, 255)
    else:
        return (0, 0, 0)

def pick_text_color(bg):
    bg_lum         = get_luminance(bg[0], bg[1], bg[2])
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
    t = text.lower()
    if any(w in t for w in ["sale","now","limited","off","free","mega","hurry","today"]):
        return "urgent"
    elif any(w in t for w in ["unlock","potential","power","achieve","strength","win","champion"]):
        return "confident"
    elif any(w in t for w in ["love","handcraft","natural","care","gentle","beauty","since"]):
        return "warm"
    return "neutral"

# ── FONT STYLE ────────────────────────────────────────────────────────────────
def suggest_font_style(r, g, b, emotion="neutral"):
    dark   = is_dark(r, g, b)
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
        fcx = x + w // 2
        fcy = y + h // 2
        if   fcx < width//2  and fcy < height//2:  face_section = "Top-Left"
        elif fcx >= width//2 and fcy < height//2:  face_section = "Top-Right"
        elif fcx < width//2  and fcy >= height//2: face_section = "Bottom-Left"
        else:                                       face_section = "Bottom-Right"
        if abs(fcx - width//2) < width*0.2 and abs(fcy - height//2) < height*0.2:
            best_section = "Top-Left" if emotion != "warm" else "Bottom-Right"
        else:
            opposite = {'Top-Left':'Bottom-Right','Top-Right':'Bottom-Left',
                        'Bottom-Left':'Top-Right','Bottom-Right':'Top-Left'}
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
                'Top-Left':     img.crop((0, 0, width//2, height//2)),
                'Top-Right':    img.crop((width//2, 0, width, height//2)),
                'Bottom-Left':  img.crop((0, height//2, width//2, height)),
                'Bottom-Right': img.crop((width//2, height//2, width, height)),
            }
            lowest_std = 999999
            for name, section_img in sections.items():
                std = np.std(np.array(section_img).reshape(-1, 3))
                if std < lowest_std:
                    lowest_std = std
                    best_section = name
            if best_section is None:
                best_section = "Top-Left"

    if emotion == "urgent" and best_section.startswith("Bottom"):
        best_section = best_section.replace("Bottom", "Top")
    elif emotion == "warm" and best_section.startswith("Top"):
        best_section = best_section.replace("Top", "Bottom")
    return best_section

# ── LOGO PASTE ────────────────────────────────────────────────────────────────
def paste_logo(img, logo_bytes, corner, opacity):
    width, height = img.size
    logo = Image.open(BytesIO(logo_bytes)).convert("RGBA")

    # resize to 1/6 of image width keeping aspect ratio
    logo_w = width // 6
    logo_h = int(logo.height * (logo_w / logo.width))
    logo   = logo.resize((logo_w, logo_h), Image.LANCZOS)

    # apply opacity to alpha channel
    if opacity < 100:
        r_ch, g_ch, b_ch, a_ch = logo.split()
        a_ch = a_ch.point(lambda p: int(p * opacity / 100))
        logo = Image.merge("RGBA", (r_ch, g_ch, b_ch, a_ch))

    margin = 20
    positions = {
        "Top-Left":     (margin, margin),
        "Top-Right":    (width - logo_w - margin, margin),
        "Bottom-Left":  (margin, height - logo_h - margin),
        "Bottom-Right": (width - logo_w - margin, height - logo_h - margin),
    }
    pos = positions[corner]
    img = img.convert("RGBA")
    img.paste(logo, pos, logo)
    return img.convert("RGB")

# ── CORE RENDER — one complete poster with a specific font ────────────────────
def render_poster(image_bytes, text, cta_text, contact,
                  font_name, logo_bytes=None, logo_corner="Bottom-Right", logo_opacity=100):
    img    = Image.open(BytesIO(image_bytes)).convert('RGB')
    width, height = img.size
    bg      = get_dominant_color(BytesIO(image_bytes) if False else "temp_image.jpg")
    emotion = analyze_text_emotion(text)

    text_fill_color = suggest_rgb(bg[0], bg[1], bg[2])
    font_size       = max(20, width // 20)
    font_path       = f"{font_name}.ttf"

    try:
        font = ImageFont.truetype(font_path, font_size)
    except:
        font = ImageFont.load_default()

    position_name = suggest_position("temp_image.jpg", emotion)
    pos_map = {
        'Top-Left':     (50, 50),
        'Top-Right':    (width // 2, 50),
        'Bottom-Left':  (50, height - (font_size * 4)),
        'Bottom-Right': (width // 2, height - (font_size * 4)),
    }
    x, y = pos_map[position_name]

    wrap_width   = max(10, (width // font_size) - 2)
    wrapped      = textwrap.fill(text, width=wrap_width)
    shadow_color = (255, 255, 255) if text_fill_color == (0, 0, 0) else (0, 0, 0)

    draw = ImageDraw.Draw(img)
    draw.text((x + 2, y + 2), wrapped, fill=shadow_color, font=font)
    draw.text((x, y),         wrapped, fill=text_fill_color, font=font)

    # ── CTA ──
    if cta_text or contact:
        cta_size = max(16, width // 18)
        ph_size  = max(14, width // 22)
        if font_name in ["Pacifico","Dancing Script","Sacramento","Pinyon Script",
                         "Playfair Display","Bodoni Moda","EB Garamond","Lora"]:
            cta_font_path = "Oswald.ttf"
        else:
            cta_font_path = "Playfair Display.ttf"
        try:
            cta_font = ImageFont.truetype(cta_font_path, cta_size)
        except:
            cta_font = ImageFont.load_default()
        try:
            ph_font = ImageFont.truetype(font_path, ph_size)
        except:
            ph_font = ImageFont.load_default()

        cta_color  = pick_text_color(bg)
        cta_shadow = (0, 0, 0) if cta_color == (255, 255, 255) else (255, 255, 255)

        cta_x = width - (width // 3)
        cta_y = height - (cta_size * 3)
        ph_x  = cta_x
        ph_y  = cta_y + cta_size + 20

        draw.text((cta_x + 2, cta_y + 2), cta_text, fill=cta_shadow, font=cta_font)
        draw.text((cta_x, cta_y),          cta_text, fill=cta_color,  font=cta_font)

        if contact and contact.startswith("http"):
            qr = qrcode.make(contact).resize((ph_size * 2, ph_size * 2))
            img.paste(qr, (ph_x, ph_y))
        elif contact:
            draw.text((ph_x, ph_y), contact, fill=cta_color, font=ph_font)

    # ── LOGO ──
    if logo_bytes:
        img = paste_logo(img, logo_bytes, logo_corner, logo_opacity)

    return img, text_fill_color, font_path, position_name, emotion, bg

# ── LIVE REPOSITION ───────────────────────────────────────────────────────────
def render_live(mx, my, cx, cy, bg, text_fill_color, selected_font,
                font_path, text, cta_text, contact,
                image_bytes, logo_bytes=None, logo_corner="Bottom-Right", logo_opacity=100):
    img    = Image.open(BytesIO(image_bytes)).convert('RGB')
    width, height = img.size

    new_main_x = int(width  * mx / 100)
    new_main_y = int(height * my / 100)
    new_cta_x  = int(width  * cx / 100)
    new_cta_y  = int(height * cy / 100)

    font_size = max(20, width // 20)
    cta_size  = max(16, width // 18)
    ph_size   = max(14, width // 22)

    try:
        font = ImageFont.truetype(f"{selected_font}.ttf", font_size)
    except:
        font = ImageFont.load_default()

    if selected_font in ["Pacifico","Dancing Script","Sacramento","Pinyon Script",
                         "Playfair Display","Bodoni Moda","EB Garamond","Lora"]:
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

    wrap_width   = max(10, (width // font_size) - 2)
    wrapped      = textwrap.fill(text, width=wrap_width)
    shadow_color = (255, 255, 255) if text_fill_color == (0, 0, 0) else (0, 0, 0)

    draw = ImageDraw.Draw(img)
    draw.text((new_main_x + 2, new_main_y + 2), wrapped, fill=shadow_color, font=font)
    draw.text((new_main_x, new_main_y),           wrapped, fill=text_fill_color, font=font)

    cta_color  = pick_text_color(bg)
    cta_shadow = (0, 0, 0) if cta_color == (255, 255, 255) else (255, 255, 255)

    draw.text((new_cta_x + 2, new_cta_y + 2), cta_text, fill=cta_shadow, font=cta_font)
    draw.text((new_cta_x, new_cta_y),           cta_text, fill=cta_color,  font=cta_font)

    ph_x = new_cta_x
    ph_y = new_cta_y + cta_size + 20
    if contact and contact.startswith("http"):
        qr = qrcode.make(contact).resize((ph_size * 2, ph_size * 2))
        img.paste(qr, (ph_x, ph_y))
    elif contact:
        draw.text((ph_x, ph_y), contact, fill=cta_color, font=ph_font)

    if logo_bytes:
        img = paste_logo(img, logo_bytes, logo_corner, logo_opacity)

    return img

# ═══════════════════════════════════════════════════════════════════════════════
# ── UI ────────────────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════
st.title("Ad Maker")
st.write("Upload your photo and get science-backed design suggestions.")

# ── INPUTS ────────────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader("Upload your image", type=["jpg","jpeg","png","webp"])
text     = st.text_input("Enter main text:")
cta_text = st.text_input("Enter CTA text (e.g. Visit Us Today):")
contact  = st.text_input("Enter phone number or website link:")

st.markdown("---")
st.subheader("Logo (optional)")
logo_file    = st.file_uploader("Upload your logo", type=["jpg","jpeg","png","webp"], key="logo")
logo_corner  = st.radio("Logo corner", ["Top-Left","Top-Right","Bottom-Left","Bottom-Right"],
                         index=3, horizontal=True)
logo_opacity = st.slider("Logo opacity", 10, 100, 90)
st.markdown("---")

# ── GENERATE ──────────────────────────────────────────────────────────────────
if st.button("Generate — see 3 font options", type="primary"):
    if uploaded_file and text:
        image_bytes = uploaded_file.getbuffer().tobytes()
        logo_bytes  = logo_file.getbuffer().tobytes() if logo_file else None

        # save temp file — needed for face detection and dominant color (cv2/PIL reads path)
        with open("temp_image.jpg", "wb") as f:
            f.write(image_bytes)

        bg      = get_dominant_color("temp_image.jpg")
        emotion = analyze_text_emotion(text)
        fonts   = suggest_font_style(bg[0], bg[1], bg[2], emotion)

        # render three posters — one per font
        previews = []
        shared_meta = None
        for font_name in fonts:
            img, text_fill_color, font_path, position, emo, bg_out = render_poster(
                image_bytes, text, cta_text, contact,
                font_name, logo_bytes, logo_corner, logo_opacity
            )
            previews.append((font_name, img))
            if shared_meta is None:
                shared_meta = (text_fill_color, font_path, position, emo, bg_out)

        st.session_state["previews"]        = previews
        st.session_state["fonts"]           = fonts
        st.session_state["image_bytes"]     = image_bytes
        st.session_state["logo_bytes"]      = logo_bytes
        st.session_state["logo_corner"]     = logo_corner
        st.session_state["logo_opacity"]    = logo_opacity
        st.session_state["bg"]              = shared_meta[4]
        st.session_state["emotion"]         = shared_meta[3]
        st.session_state["position"]        = shared_meta[2]
        st.session_state["text_fill_color"] = shared_meta[0]
        st.session_state["font_path"]       = shared_meta[1]
        st.session_state["chosen_font"]     = None
        st.session_state["show_adjust"]     = False

    else:
        st.warning("Please upload an image and enter your main text.")

# ── THREE FONT PREVIEWS ───────────────────────────────────────────────────────
if st.session_state.get("previews") and not st.session_state.get("show_adjust"):
    st.markdown("---")
    st.subheader("Choose your font style")
    st.caption("These three fonts were picked by science — based on your image mood and text emotion. Choose the one that feels right.")

    col1, col2, col3 = st.columns(3)
    for col, (font_name, img) in zip([col1, col2, col3], st.session_state["previews"]):
        with col:
            st.image(img, caption=font_name, use_container_width=True)
            if st.button(f"Use {font_name}", key=f"pick_{font_name}"):
                st.session_state["chosen_font"]     = font_name
                st.session_state["font_path"]       = f"{font_name}.ttf"
                st.session_state["text_fill_color"] = suggest_rgb(
                    st.session_state["bg"][0],
                    st.session_state["bg"][1],
                    st.session_state["bg"][2]
                )
                st.session_state["show_adjust"] = True
                st.rerun()

# ── ADJUSTMENT PANEL ─────────────────────────────────────────────────────────
if st.session_state.get("show_adjust"):
    bg              = st.session_state["bg"]
    emotion         = st.session_state["emotion"]
    fonts           = st.session_state["fonts"]
    selected_font   = st.session_state["chosen_font"]
    position        = st.session_state["position"]
    text_fill_color = st.session_state["text_fill_color"]
    font_path       = st.session_state["font_path"]
    image_bytes     = st.session_state["image_bytes"]
    logo_bytes      = st.session_state["logo_bytes"]
    logo_corner     = st.session_state["logo_corner"]
    logo_opacity    = st.session_state["logo_opacity"]

    st.markdown("---")
    st.subheader(f"Adjusting — {selected_font}")

    col1, col2 = st.columns(2)
    with col1:
        st.write("**Main Text**")
        main_x = st.slider("Left / Right", 0, 100, 10, key="main_x")
        main_y = st.slider("Up / Down",    0, 100, 10, key="main_y")
    with col2:
        st.write("**CTA**")
        cta_x_pct = st.slider("Left / Right", 0, 100, 70, key="cta_x")
        cta_y_pct = st.slider("Up / Down",    0, 100, 80, key="cta_y")

    live_img = render_live(
        main_x, main_y, cta_x_pct, cta_y_pct,
        bg, text_fill_color, selected_font, font_path,
        text, cta_text, contact,
        image_bytes, logo_bytes, logo_corner, logo_opacity
    )
    st.image(live_img, caption="Live Preview", use_container_width=True)

    # ── WHY THIS DESIGN ──
    hex_color = rgb_to_hex(text_fill_color[0], text_fill_color[1], text_fill_color[2])
    cmyk      = rgb_to_cmyk(text_fill_color[0], text_fill_color[1], text_fill_color[2])
    bg_hex    = rgb_to_hex(bg[0], bg[1], bg[2])
    bg_cmyk   = rgb_to_cmyk(bg[0], bg[1], bg[2])

    with open("temp_image.jpg", "wb") as f:
        f.write(image_bytes)

    st.markdown("---")
    st.subheader("Why this design")
    st.info(f"**Font** — {selected_font} chosen because your image is "
            f"{'dark' if is_dark(bg[0],bg[1],bg[2]) else 'light'} and "
            f"{get_warmth(bg[0],bg[1],bg[2])}, and your text emotion is {emotion}.")
    st.info(f"**Color** — {hex_color} chosen using WCAG contrast math. Meets AA readability standard.")
    st.info(f"**Position** — text placed {position} based on "
            f"{'face detection avoiding the subject' if len(find_face('temp_image.jpg')) > 0 else 'the flattest area of the image (least visual noise)'}.")

    st.subheader("Design Analysis")
    st.write(f"**Emotion:** {emotion}  |  **Position:** {position}  |  **Font:** {selected_font}")
    st.write(f"**Top 3 fonts:** {', '.join(fonts)}")
    st.write(f"**Font color** RGB {text_fill_color} | HEX {hex_color} | CMYK C={cmyk[0]} M={cmyk[1]} Y={cmyk[2]} K={cmyk[3]}")
    st.write(f"**Background** RGB {bg} | HEX {bg_hex} | CMYK C={bg_cmyk[0]} M={bg_cmyk[1]} Y={bg_cmyk[2]} K={bg_cmyk[3]}")

    # save and download
    buf = BytesIO()
    live_img.save(buf, format="JPEG")
    st.download_button("Download Poster", buf.getvalue(), "poster.jpg", "image/jpeg", key="dl")
