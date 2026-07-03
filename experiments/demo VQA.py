import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

# 1. 模型路径
model_path = r"E:\Qwen\models\Qwen2.5-VL-7B-Instruct"

# 2. 加载 Processor 和 Model
print("正在加载模型，请稍候...")
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16, 
    device_map="auto",
    trust_remote_code=True
)
model.eval()
print("模型加载完成！\n" + "="*40)

# 修改 run 函数，使其支持传入自定义的 prompt 提示词
def run(image_path, prompt_text):
    # 打开图片
    image = Image.open(image_path).convert("RGB")

    # 将传入的 prompt_text 放到消息结构中
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt_text}  # 动态传入提示词
            ]
        }
    ]

    # 准备输入数据
    text = processor.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    inputs = processor(
        text=[text], 
        images=[image], 
        padding=True, 
        return_tensors="pt"
    )
    
    inputs = inputs.to(model.device)

    # 推理生成
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=150) # 稍微放大token，防止详细描述被截断
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
    # 解码输出
    output_text = processor.batch_decode(
        generated_ids_trimmed, 
        skip_special_tokens=True, 
        clean_up_tokenization_spaces=False
    )
    
    # 打印结果
    print(f"【Prompt】: {prompt_text}")
    print(f"【模型回复】:\n{output_text[0]}")
    print("-" * 50)

if __name__ == "__main__":
    # 准备你的测试图片（你可以换成 traffic.jpg 绘图）
    test_image = "traffic.jpg" 
    
    # 定义你要对比的 4 个不同的 Prompt
    prompts = [
        "Describe this image.",
        "Describe the image in detail.",
        "Please focus on the objects.",
        "图片中有：\n汽车\n红绿灯\n行人\n……\n请仿照上面的格式，列举出图片中的物体。" # 对应你目标中的格式要求
    ]
    
    # 循环运行，对比结果
    for i, p in enumerate(prompts, 1):
        print(f"\n正在测试第 {i} 个实验...")
        try:
            run(test_image, p)
        except FileNotFoundError:
            # 如果没有 traffic.jpg，先用你刚才的 cat.jpg 顶替测试
            print(f"未找到 {test_image}，自动切换为 cat.jpg 进行测试...")
            test_image = "cat.jpg"
            run(test_image, p)