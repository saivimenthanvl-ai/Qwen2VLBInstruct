#!pip -q install "transformers==4.51.3" "accelerate>=0.31.0" pillow timm

import torch
import requests
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
from io import BytesIO
import json
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype  = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "microsoft/Florence-2-base-ft"

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    torch_dtype=dtype,
    attn_implementation="eager",     # <- key line
).to(device)

if not hasattr(model, "_supports_sdpa"):
    model._supports_sdpa = False

processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG"
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

prompt = "<DETAILED_CAPTION>"
inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, dtype)

with torch.inference_mode():
    gen_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=96,
        num_beams=3,
        do_sample=False,
    )

gen_text = processor.batch_decode(gen_ids, skip_special_tokens=False)[0]
caption = processor.post_process_generation(
    gen_text, task="<DETAILED_CAPTION>", image_size=(image.width, image.height)
)["<DETAILED_CAPTION>"]
print("Caption:", caption)

od_inputs = processor(text="<OD>", images=image, return_tensors="pt").to(device, dtype)
od_text = processor.batch_decode(od_ids, skip_special_tokens=False)[0]
od = processor.post_process_generation(
    od_text, task="<OD>", image_size=(image.width, image.height)
)["<OD>"]
print("Detected labels (first 10):", od["labels"][:10])

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype  = torch.float16 if torch.cuda.is_available() else torch.float32

MODEL_ID = "microsoft/Florence-2-base-ft"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, trust_remote_code=True, torch_dtype=dtype, attn_implementation="eager"
).to(device)
# Patch: some transformer builds expect this private flag
if not hasattr(model, "_supports_sdpa"):
    model._supports_sdpa = False

processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

def load_image(url: str) -> Image.Image:
    return Image.open(requests.get(url, stream=True).raw).convert("RGB")

def florence_generate(img: Image.Image, task_token: str, max_new_tokens=128, beams=3):
    inputs = processor(text=task_token, images=img, return_tensors="pt").to(device, dtype)
    with torch.inference_mode():
        ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=max_new_tokens,
            num_beams=beams,
            do_sample=False,
        )
    raw = processor.batch_decode(ids, skip_special_tokens=False)[0]
    out = processor.post_process_generation(
        raw, task=task_token, image_size=(img.width, img.height)
    )[task_token]
    return out

img1 = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG")

od1 = florence_generate(img1, "<OD>", max_new_tokens=256)
labels1 = {l.lower() for l in od1.get("labels", [])}

cap1 = florence_generate(img1, "<DETAILED_CAPTION>", max_new_tokens=128)

animals_vocab = {
    "cat","dog","bear","rabbit","cow","horse","tiger","lion","panda",
    "fox","penguin","koala","monkey","elephant","giraffe","zebra","sheep","goat","deer","duck","bird","mouse"
}

def guess_animal(labels: set, caption: str) -> str:
    # Prefer OD labels
    for a in animals_vocab:
        if a in labels:
            return a
    cap = caption.lower()
    for a in animals_vocab:
        if a in cap:
            return a
    return "unknown"

print("Output for image 1:")
print("OD labels (sample):", list(labels1)[:10])
print("Caption:", cap1)
print("Animal guess:", guess_animal(labels1, cap1))

horse_candidates = [
    # Pexels (public CDN)
    "https://images.pexels.com/photos/1996333/pexels-photo-1996333.jpeg",
    "https://images.pexels.com/photos/1996334/pexels-photo-1996334.jpeg",
    # Wikimedia (public)
    "https://upload.wikimedia.org/wikipedia/commons/9/9a/Andalusian_horse_2009.jpg",
]

img2 = load_first_working_image(horse_candidates)

desc2 = florence_generate(img2, "<DETAILED_CAPTION>", max_new_tokens=128)
print("\nOutput for image 2:")
print(desc2)

from huggingface_hub import whoami, login
login(token=os.environ["HF_TOKEN"])
print(whoami())  # should print your user + orgs

os.environ["HF_TOKEN"] = "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"

headers = {}
if os.getenv("HF_BILL_TO_ORG"):
    headers["X-HF-Bill-To"] = os.environ["HF_BILL_TO_ORG"]

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ["HF_TOKEN"],
    default_headers=headers or None,
)

ping = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",  # text-only to isolate permissions
    messages=[{"role": "user", "content": "Say 'pong' if auth works."}],
    max_tokens=8,
)
print("Ping:", ping.choices[0].message.content)

resp = client.chat.completions.create(
    model="Qwen/Qwen2.5-VL-7B-Instruct",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "What animal is on the candy?"},
            {"type": "image_url", "image_url": {
                "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG"
            }},
        ],
    }],
    max_tokens=64,
)
print(resp.choices[0].message.content)

