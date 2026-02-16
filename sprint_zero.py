"""Sprint Zero: validate API + image encoding with a single grid image."""

import base64
import io
import os

import anthropic
from PIL import Image, ImageDraw


def make_grid(rows: int = 5, cols: int = 6, size: int = 512) -> Image.Image:
    img = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(img)
    row_h = size / rows
    col_w = size / cols
    for r in range(rows + 1):
        y = int(r * row_h)
        draw.line([(0, y), (size, y)], fill="black", width=2)
    for c in range(cols + 1):
        x = int(c * col_w)
        draw.line([(x, 0), (x, size)], fill="black", width=2)
    return img


def image_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.standard_b64encode(buf.getvalue()).decode("utf-8")


def main():
    model = os.environ.get("VLM_MODEL", "claude-haiku-4-5-20251001")
    img = make_grid(5, 6)
    b64 = image_to_base64(img)

    client = anthropic.Anthropic()
    response = client.messages.create(
        model=model,
        max_tokens=256,
        temperature=0.0,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": b64,
                        },
                    },
                    {
                        "type": "text",
                        "text": "Count the rows and columns in this grid. "
                        "Reply in the format: rows=N columns=M",
                    },
                ],
            }
        ],
    )

    text = response.content[0].text
    print(f"Model: {model}")
    print(f"Ground truth: rows=5 columns=6")
    print(f"Response: {text}")
    print(f"Tokens: {response.usage.input_tokens} in, {response.usage.output_tokens} out")


if __name__ == "__main__":
    main()
