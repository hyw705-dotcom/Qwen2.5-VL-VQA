import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

# 1. 替换为正确的模型路径
model_path = r"E:\Qwen\models\Qwen2.5-VL-7B-Instruct"

# 2. 加载正确的 Processor 和 Model 类
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16, # 推荐用 bfloat16，如果显卡较老（如10系）可换成 float16
    device_map="auto",
    trust_remote_code=True
)

model.eval()

def run(image_path):
    # 打开图片
    image = Image.open(image_path).convert("RGB")

    # 3. 官方推荐的 Messages 格式（图片传入路径或PIL对象，但格式需符合标准）
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "请简要描述这张图片。"}
            ]
        }
    ]

    # 4. 准备输入数据（注意：新版需要用 text 传参生成文本提示词）
    text = processor.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    # 5. 同时处理文本和图片，转换成模型需要的 Tensor
    # Qwen2.5-VL 的 inputs 是一个自定义字典，不能直接 .to(device)
    inputs = processor(
        text=[text], 
        images=[image], 
        padding=True, 
        return_tensors="pt"
    )
    
    # 正确的数据搬运到 GPU 的方式
    inputs = inputs.to(model.device)

    # 6. 推理生成
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        
        # 过滤掉输入部分，只保留模型新生成的回答
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
    # 7. 解码输出
    output_text = processor.batch_decode(
        generated_ids_trimmed, 
        skip_special_tokens=True, 
        clean_up_tokenization_spaces=False
    )
    
    print("\n模型回复：")
    print(output_text[0])

if __name__ == "__main__":
    run("cat.jpg")