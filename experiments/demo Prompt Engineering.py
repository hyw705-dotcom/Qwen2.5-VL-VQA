import os
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

# 1. 模型初始化
model_path = r"E:\Qwen\models\Qwen2.5-VL-7B-Instruct"

print("正在加载 Qwen2.5-VL 模型...")
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16, 
    device_map="auto",
    trust_remote_code=True
)
model.eval()
print("模型加载成功！开始 Prompt Engineering 评测...\n" + "="*50)


# 2. 统一推理函数
def ask_with_prompt(image_path, prompt_text):
    image = Image.open(image_path).convert("RGB")
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt_text}
            ]
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt").to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=200) # 给足Token让Role-play和Few-shot充分发挥
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        
    output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return output_text[0].strip()


# 3. 评测主程序
if __name__ == "__main__":
    
    # 🌟 建议使用第三/四阶段的交通图片进行对比，最容易看出“交通专家”的效果
    test_image = "traffic.jpg" 
    
    if not os.path.exists(test_image):
        print(f"⚠️ 未找到 {test_image}，自动切换为 cat.jpg 进行测试...")
        test_image = "cat.jpg"

    # 🌟 核心：设计三种不同的 Prompt 架构
    prompt_experiments = {
        
        # 1. Zero-shot (零样本提示：直接下达命令，不做任何示范)
        "Zero-shot Prompt": (
            "请分析并描述这张图片中的交通状况。"
        ),
        
        # 2. Few-shot (少样本提示：给大模型吃“定心丸”，先教它规范，再让它模仿)
        "Few-shot Prompt": (
            "根据输入的图片，请模仿以下示例的结构来回答当前图片的问题。\n\n"
            "【示例】\n"
            "Q: 请分析并描述这张图片中的交通状况。\n"
            "A: [主要对象]: 街道与车辆。\n"
            "[细节发现]: 画面中央有两辆私家车，左侧有行人。\n"
            "[环境感知]: 天气晴朗，路面干燥。\n\n"
            "【正式任务】\n"
            "Q: 请分析并描述这张图片中的交通状况。\n"
            "A:"
        ),
        
        # 3. Role-play Prompt (角色扮演提示：注入专业人设，激发隐性领域知识)
        "Role-play Prompt": (
            "你是一位拥有20年经验的资深交通规划专家与交警总指挥。请从专业的交通安全、"
            "车流密度、潜在违章风险、道路合规性等深度视角，严谨地分析并描述这张图片中的交通状况。"
        )
    }

    # 实验结果保存
    report_file = "prompt_engineering_report.md"
    
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("# Qwen2.5-VL 提示词工程 (Prompt Engineering) 评测报告\n\n")
        f.write(f"**测试图片：** `{test_image}`\n\n")
        f.write("---\n\n")
        
        for name, prompt in prompt_experiments.items():
            print(f"正在测试: {name}...")
            
            # 执行推理
            response = ask_with_prompt(test_image, prompt)
            
            # 打印到控制台
            print(f"【{name} 结果】:\n{response}\n" + "-"*30)
            
            # 写入 Markdown 报告
            f.write(f"## 实验组：{name}\n\n")
            f.write(f"**输入的注入 Prompt：**\n```text\n{prompt}\n```\n\n")
            f.write(f"**模型的实际输出 (Output)：**\n\n{response}\n\n")
            f.write("---\n\n")
            
    print(f"\n🎉 评测完成！结果已导出至: {os.path.abspath(report_file)}")
    print("💡 请打开该文件，对比这三种输出在【格式规整度】和【内容专业度】上的巨大区别！")