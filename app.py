import os
import requests
import random
import colorsys
import textwrap
import qrcode
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from sklearn.cluster import KMeans
from io import BytesIO
import streamlit as st

# ── API KEY ──────────────────────────────────────────────────────────────────
api_key = st.secrets["GOOGLE_FONTS_API_KEY"]

# ── FONT LIST ─────────────────────────────────────────────────────────────────
font_list = [
    "Inter", "Montserrat", "Oswald", "Playfair Display", "Bodoni Moda",
    "EB Garamond", "Lora", "Pacifico", "Dancing Script", "Sacramento",
    "Pinyon Script", "Bebas Neue", "Abril Fatface"
]

# ── DOWNLOAD FONTS ────────────────────────────────────────────────────────────
def download_font(font_name, api_key):
    filename = f"{font_name}.ttf"
    if os.path.exists(filename):
        return
    encoded_name = font_name.replace(" ", "+")
    url = f"https://www.googleapis.com/webfonts/v1/webfonts?key={api_key}&family={encoded_name}"
    response = requests.get(url)
    data = response.json()
    if 'items' not in data or not data['items']:
        print(f"Error: Could not find font data for {font_name}.")
        return
    files = data['items'][0]['files']
    font_url = files.get('regular') or files.get('400') or list(files.values())[0]
    font_url = font_url.replace("http://", "https://")
    font_response = requests.get(font_url)
    with open(filename, 'wb') as f:
        f.write(font_response.content)

for font in font_list:
    download_font(font, api_key)

# ── COLOR HELPERS ─────────────────────────────────────────────────────────────
def is_dark(r, g, b):
    return (r + g + b) / 3 < 127.5

def get_warmth(r, g, b):
    return "warm" if (r - b) > 0 else "cool"

def get_dominant_color(image_path, k=1):
    img = Image.open(image_path).convert('RGB')
    img.thumbnail((200, 200))
    pixels = np.array(img).reshape(-1, 3)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(pixels)
    dominant_color = kmeans.cluster_centers_[0].astype(int)
    return tuple(int(x) for x in dominant_color)

def get_luminance(r, g, b):
    r, g, b = r/255, g/255, b/255
    return 0.2126 * r + 0.7152 * g + 0.0722 * b

def get_contrast_ratio(r1, g1, b1, r2, g2, b2):
    L1 = get_luminance(r1, g1, b1)
    L2 = get_luminance(r2, g2, b2)
    lighter = max(L1, L2)
    darker = min(L1, L2)
    return (lighter + 0.05) / (darker + 0.05)

def suggest_rgb(r, g, b):
    brightness = (r + g + b) / 3
    if brightness < 30:
        return (255, 255, 255)
    elif brightness > 220:
        return (0, 0, 0)
    else:
        h, s, v = colorsys.rgb_to_hsv(r/255, g/255, b/255)
        h = (h + 0.5) % 1
        v = max(v, 0.8)
        r2, g2, b2 = colorsys.hsv_to_rgb(h, s, v)
        contrast = get_contrast_ratio(r, g, b, int(r2*255), int(g2*255), int(b2*255))
        if contrast < 4.5:
            white_contrast = get_contrast_ratio(r, g, b, 255, 255, 255)
            black_contrast = get_contrast_ratio(r, g, b, 0, 0, 0)
            return (255, 255, 255) if white_contrast > black_contrast else (0, 0, 0)
        return (int(r2 * 255), int(g2 * 255), int(b2 * 255))

def rgb_to_hex(r, g, b):
    return f"#{r:02x}{g:02x}{b:02x}".upper()

def rgb_to_cmyk(r, g, b):
    r, g, b = r/255, g/255, b/255
    k = 1 - max(r, g, b)
    if k == 1:
        return (0, 0, 0, 100)
    c = (1 - r - k) / (1 - k)
    m = (1 - g - k) / (1 - k)
    y = (1 - b - k) / (1 - k)
    return (round(c*100), round(m*100), round(y*100), round(k*100))

# ── FACE DETECTION ────────────────────────────────────────────────────────────
def find_face(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    return faces

# ── TEXT EMOTION ──────────────────────────────────────────────────────────────
def analyze_text_emotion(text):
    text = text.lower()
    if any(word in text for word in ["sale", "now", "limited", "off", "free", "mega", "hurry", "today"]):
        return "urgent"
    elif any(word in text for word in ["unlock", "potential", "power", "achieve", "strength", "win", "champion"]):
        return "confident"
    elif any(word in text for word in ["love", "handcraft", "natural", "care", "gentle", "beauty", "since"]):
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
                'Top-Left':     img.crop((0, 0, width//2, height//2)),
                'Top-Right':    img.crop((width//2, 0, width, height//2)),
                'Bottom-Left':  img.crop((0, height//2, width//2, height)),
                'Bottom-Right': img.crop((width//2, height//2, width, height)),
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

# ── GRID HELPERS ──────────────────────────────────────────────────────────────
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
    r, g, b = suggest_rgb(bg[0], bg[1], bg[2])
    text_fill_color = (r, g, b)
    emotion = analyze_text_emotion(text)
    font_size = max(20, width // 20)
    fonts = suggest_font_style(bg[0], bg[1], bg[2], emotion)
    selected_font = fonts[0]
    actual_font_path = f"{selected_font}.ttf"
    try:
        font = ImageFont.truetype(actual_font_path, font_size)
    except Exception as e:
        print(f"Font failed: {e}")
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
    draw.text((x+2, y+2), wrapped_text, fill=shadow_color, font=font)
    draw.text((x, y), wrapped_text, fill=text_fill_color, font=font)
    output_path = "output_temp.jpg"
    img.save(output_path)
    return img, output_path, text_fill_color, actual_font_path

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
        ph_font = ImageFont.truetype(font_path, ph_size)
    except:
        ph_font = ImageFont.load_default()
    white_contrast = get_contrast_ratio(bg[0], bg[1], bg[2], 255, 255, 255)
    black_contrast = get_contrast_ratio(bg[0], bg[1], bg[2], 0, 0, 0)
    if white_contrast > black_contrast:
        cta_color = (255, 255, 255)
        shadow_color = (0, 0, 0)
    else:
        cta_color = (0, 0, 0)
        shadow_color = (255, 255, 255)
    if manual_pos:
        cta_x, cta_y = grid_to_coords(manual_pos, width, height)
    else:
        cta_x = width - (width // 3)
        cta_y = height - (cta_size * 3)
    ph_x = cta_x
    ph_y = cta_y + cta_size + 20
    draw.text((cta_x+2, cta_y+2), cta_text, fill=shadow_color, font=cta_font)
    draw.text((cta_x, cta_y), cta_text, fill=cta_color, font=cta_font)
    if contact.startswith("http"):
        qr = qrcode.make(contact)
        qr = qr.resize((ph_size * 2, ph_size * 2))
        img.paste(qr, (ph_x, ph_y))
    else:
        draw.text((ph_x, ph_y), contact, fill=cta_color, font=ph_font)
    return img

# ── UI — ALWAYS LAST ──────────────────────────────────────────────────────────
st.title("Ad Maker")
st.write("Welcome to Ad Maker!")
st.write("Upload your photo and get science-backed design suggestions")

uploaded_file = st.file_uploader("Upload your image", type=["jpg", "jpeg", "png"])
text = st.text_input("Enter main text:")
cta_text = st.text_input("Enter CTA text (e.g. Visit Us Today):")
contact = st.text_input("Enter phone number or website link:")

if st.button("Generate Poster"):
    if uploaded_file and text:
        with open("temp_image.jpg", "wb") as f:
            f.write(uploaded_file.getbuffer())

        img, output_path, text_fill_color, font_path = render_text_on_image(
            "temp_image.jpg", text
        )

        bg = get_dominant_color("temp_image.jpg")
        fonts = suggest_font_style(bg[0], bg[1], bg[2], analyze_text_emotion(text))
        selected_font = fonts[0]

        img = render_cta(
            img, cta_text, contact, text_fill_color, bg, selected_font, font_path
        )

        st.image(img, caption="Your Generated Poster")
    else:
        st.warning("Please upload an image and enter your main text.")
