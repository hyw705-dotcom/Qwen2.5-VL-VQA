import os
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

# 1. 模型初始化配置
model_path = r"E:\Qwen\models\Qwen2.5-VL-7B-Instruct"

print("正在加载 Qwen2.5-VL 视觉大模型...")
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16, 
    device_map="auto",
    trust_remote_code=True
)
model.eval()
print("模型加载成功！开始执行 VQA 评测任务...\n" + "="*50)


# 2. 核心 VQA 推理函数
def ask_qwen_vqa(image_path, question):
    """
    输入图片和单个问题，返回模型的文本回答
    """
    try:
        image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        return f"错误：未找到图片文件 {image_path}"

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question}
            ]
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt").to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=100) # VQA 通常是短回答，100 足够
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        
    output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return output_text[0].strip()


# 3. 自动化测试主程序
if __name__ == "__main__":
    
    # 🌟 在这里配置你的 20 张图片和各自对应的 5 个问题
    # 格式为: "图片文件名": ["问题1", "问题2", "问题3", "问题4", "问题5"]
    vqa_dataset = {
        # 场景 1：人物类
        "people.jpg": [
            "图片中一共有多少人？",
            "中间的人穿着什么颜色的衣服？",
            "他们正在干什么？",
            "能看出这是在室内还是室外吗？",
            "照片里的天气或者光线怎么样？"
        ],
        # 场景 2：交通类
        "traffic.jpg": [
            "现在的红绿灯是红灯还是绿灯？",
            "马路上有没有汽车？",
            "画面里有行人或自行车吗？",
            "天气如何？是晴天还是阴天？",
            "能看到路标或地面的指示线吗？"
        ],
        # 场景 3：OCR 文本文字类
        "ocr_shop.jpg": [
            "招牌上写的是什么中文字？",
            "能看到菜单或海报上的价格是多少吗？",
            "这家店的名字叫什么？",
            "有什么显眼的英文字母吗？",
            "招牌是什么颜色的？"
        ],
        # 场景 4：室内/桌面类
        "indoor.jpg": [
            "桌子上放了什么东西？",
            "房间里有几台电脑或显示器？",
            "有没有植物或者水杯？",
            "椅子的颜色是什么？",
            "房间的整体照明是明亮还是昏暗？"
        ],
        
        # 💡 提示：你可以在下方继续按照这个格式添加剩下的 16 张图，直到凑满 20 张
    }

    # 实验报告保存路径
    report_file = "vqa_experiment_report.md"
    
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("# Qwen2.5-VL 视觉问答 (VQA) 实验报告\n\n")
        
        # 遍历数据集开始测试
        for img_name, questions in vqa_dataset.items():
            print(f"\n正在处理图片: {img_name}")
            
            # 检查图片是否存在，不存在则跳过，防止程序崩溃
            if not os.path.exists(img_name):
                print(f"⚠️ 警告: 当前目录下未找到 {img_name}，已跳过。请确保图片放在 E:\\Qwen\\ 目录下。")
                continue
                
            f.write(f"## 图片: {img_name}\n")
            f.write(f"![{img_name}]({img_name})\n\n")
            
            # 循环问 5 个问题
            for i, q in enumerate(questions, 1):
                print(f"  正在回答问题 {i}: {q}")
                answer = ask_qwen_vqa(img_name, q)
                
                # 实时打印到控制台看效果
                print(f"  AI回复: {answer}")
                
                # 写入 Markdown 报告
                f.write(f"**问 {i}：** {q}\n\n")
                f.write(f"**答：** {answer}\n\n")
                f.write("---\n\n")
                
    print(f"\n🎉 恭喜！第四阶段全部 VQA 测试完成！实验报告已自动保存至: {os.path.abspath(report_file)}")