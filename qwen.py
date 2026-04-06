"""
Qwen3.5-4B 统一视觉-语言模型推理示例：纯文本 + 图文多模态。

依赖: transformers, torch, pillow
默认模型: Qwen/Qwen3.5-4B（HF pipeline: image-text-to-text）
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor


DEFAULT_MODEL = "Qwen/Qwen3.5-4B"


def load_model_and_processor(
    model_name: str,
    device: torch.device,
    dtype: torch.dtype | None = None,
) -> tuple[AutoModelForImageTextToText, AutoProcessor]:
    if dtype is None:
        dtype = torch.float16 if device.type == "cuda" else torch.float32
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=dtype,
    ).to(device)
    model.eval()
    return model, processor


def generate_from_messages(
    model: AutoModelForImageTextToText,
    processor: AutoProcessor,
    messages: list[dict],
    device: torch.device,
    max_new_tokens: int = 256,
) -> str:
    try:
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    images: list[Image.Image] = []
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") != "image":
                continue
            img = item.get("image")
            if isinstance(img, (str, Path)):
                img = Image.open(img).convert("RGB")
            elif isinstance(img, Image.Image):
                img = img.convert("RGB")
            else:
                raise TypeError(f"Unsupported image value: {type(img)}")
            images.append(img)

    proc_kw: dict = {"text": [text], "return_tensors": "pt", "padding": True}
    if images:
        proc_kw["images"] = images

    inputs = processor(**proc_kw).to(device)

    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=max_new_tokens)

    in_len = inputs["input_ids"].shape[1]
    new_tokens = generated[:, in_len:]
    return processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()


def run_text_only_demo(
    model: AutoModelForImageTextToText,
    processor: AutoProcessor,
    device: torch.device,
    max_new_tokens: int,
) -> None:
    messages = [
        {"role": "user", "content": "用一两句话解释什么是梯度下降。"},
    ]
    out = generate_from_messages(model, processor, messages, device, max_new_tokens=max_new_tokens)
    print("\n=== 纯文本 ===")
    print("Q:", messages[0]["content"])
    print("A:", out)


def run_vision_demo(
    model: AutoModelForImageTextToText,
    processor: AutoProcessor,
    device: torch.device,
    image_path: str | None,
    max_new_tokens: int,
) -> None:
    if image_path:
        img_source: str | Path = image_path
    else:
        # 无文件时生成简单彩条图，便于直接跑通流程
        img = Image.new("RGB", (256, 256), color=(80, 120, 200))
        for x in range(256):
            for y in range(256):
                img.putpixel((x, y), (x % 256, y % 256, (x + y) % 256))
        img_source = img

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img_source},
                {"type": "text", "text": "请简要描述这张图片中的主要颜色和图案。"},
            ],
        }
    ]
    out = generate_from_messages(model, processor, messages, device, max_new_tokens=max_new_tokens)
    print("\n=== 图文 ===")
    print("Q: [图像] + 请简要描述这张图片中的主要颜色和图案。")
    print("A:", out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen3.5-4B 纯文本 / 图文推理测试")
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--image", default=None, help="图文测试用的图片路径；省略则使用内置测试图")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--text-only", action="store_true", help="只跑纯文本")
    parser.add_argument("--vision-only", action="store_true", help="只跑图文")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}, model: {args.model_name}")

    model, processor = load_model_and_processor(args.model_name, device)
    # print(model)
    # exit()
    run_text = not args.vision_only
    run_vision = not args.text_only

    if run_text:
        run_text_only_demo(model, processor, device, args.max_new_tokens)
    if run_vision:
        run_vision_demo(
            model,
            processor,
            device,
            image_path=args.image,
            max_new_tokens=args.max_new_tokens,
        )


if __name__ == "__main__":
    main()
