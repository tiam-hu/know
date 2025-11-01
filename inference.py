import os
import json
import torch
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor


def infer(model, processor, record, device):
    image_path = record["images"][0]
    question_text = record["input"].replace("<image>", "").strip()


    system_instruction = (
        "You are a helpful and concise visual question answering assistant. "
        "Answer as briefly as possible. "
        "If you cannot confidently answer based on the image and question, "
        "respond exactly with: I'm sorry, but this question is beyond my current knowledge."
    )

    messages = [
        {"role": "system", "content": system_instruction},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": question_text},
            ]
        }
    ]

    # 构造模型输入
    text_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=[text_prompt], images=[image_path], return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=128)

    output_text = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    # 清理冗余前缀
    if "assistant\n" in output_text:
        output_text = output_text.split("assistant\n", 1)[1].strip()
    elif "assistant:" in output_text:
        output_text = output_text.split("assistant:", 1)[1].strip()

    return output_text


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="/mnt/yunpan/hym/COCO/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--test_file", default="/mnt/yunpan/hym/COCO/test_data_2.json")
    parser.add_argument("--output_file", default="predictions_new.jsonl")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    print(f"加载模型：{args.model_path}")
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(args.model_path, use_fast=False)


    with open(args.test_file, "r") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} samples from {args.test_file}")

    # 检查已有结果，避免重复推理
    existing_results = set()
    if os.path.exists(args.output_file):
        with open(args.output_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line)
                    existing_results.add(item["question"])
                except:
                    continue
        print(f"检测到已存在 {len(existing_results)} 条结果，将自动跳过这些样本。")

    # 开始推理
    with open(args.output_file, "a", encoding="utf-8") as f:
        for rec in tqdm(data, desc="推理中"):
            if rec["input"] in existing_results:
                continue  # 跳过已完成
            try:
                pred = infer(model, processor, rec, args.device)
            except Exception as e:
                pred = f"[Error: {str(e)}]"

            result = {
                "question": rec["input"],
                "ground_truth_answer": rec.get("ground_truth_answer", ""),
                "predicted_answer": pred
            }

            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()  # 实时写入

    print(f"✅ 推理完成，结果已保存至 {args.output_file}")


if __name__ == "__main__":
    main()
