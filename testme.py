from openai import OpenAI
import base64

client = OpenAI()

prompt = "A bright, friendly illustration of a rocket ship launching, 2D, flat design"

try:
    img = client.images.generate(
        model="gpt-image-1",
        prompt=prompt,
        n=1,
        size="1024x1024",
        output_format="png",
    )
    print("OK, got image, first 80 chars of b64:", img.data[0].b64_json[:80])
except Exception as e:
    print("ERROR:", repr(e))
