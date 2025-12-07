import os
import base64
from io import BytesIO

from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont
import textwrap

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def generate_ai_image(prompt: str, size: str = "1024x1024") -> bytes:
    """
    Call OpenAI's image API and return raw PNG bytes.
    Falls back to a nicer placeholder if anything fails.
    """
    try:
        response = client.images.generate(
            model="dall-e-3",          # or gpt-image-1 if/when you have it
            prompt=prompt,
            n=1,
            size=size,                 # "1024x1024", "1024x1792", etc.
            response_format="b64_json"
        )

        b64 = response.data[0].b64_json
        return base64.b64decode(b64)

    except Exception as e:
        print("⚠️ OpenAI image generation failed, using placeholder:", repr(e))
        return generate_placeholder_image(prompt, size)


def generate_placeholder_image(prompt: str, size: str = "1024x1024") -> bytes:
    """
    Better-looking placeholder card that matches your dark UI,
    with wrapped prompt text instead of a random block.
    """
    w, h = (int(x) for x in size.split("x"))
    bg_color = (6, 26, 48)           # dark navy like your UI
    fg_color = (220, 235, 255)

    img = Image.new("RGB", (w, h), bg_color)
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", size=int(w * 0.035))
    except Exception:
        font = ImageFont.load_default()

    title = "AI IMAGE PLACEHOLDER"
    body = f'"{prompt}"'

    # title at top
    tw, th = draw.textsize(title, font=font)
    draw.text(((w - tw) / 2, h * 0.1), title, font=font, fill=fg_color)

    # wrapped prompt text in the middle
    wrapped = textwrap.fill(body, width=40)
    _, _, _, body_h = draw.multiline_textbbox((0, 0), wrapped, font=font)

    draw.multiline_text(
        (w * 0.08, (h - body_h) / 2),
        wrapped,
        font=font,
        fill=fg_color,
        spacing=4,
    )

    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


