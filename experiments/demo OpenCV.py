import os
import cv2
import torch
import numpy as np
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
print("模型加载成功！开始 OpenCV + VLM 融合实验...\n" + "="*50)


# 2. OpenCV 图像增强算法库
def enhance_image(image_path, method="raw"):
    """
    使用 OpenCV 处理图片，并转回 PIL Image 格式供大模型读取
    """
    # 使用 OpenCV 读取图片 (BGR 格式)
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"未找到图片：{image_path}")
        
    if method == "raw":
        # 原图，仅做通道转换
        enhanced_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
    elif method == "clahe":
        # CLAHE (限制对比度自适应直方图均衡化) —— 专门对付暗光、光线不均匀
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl_gray = clahe.apply(gray)
        # 将灰度恢复为彩色亮度均衡（利用 YCrCb 颜色空间）
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        ycrcb[:, :, 0] = clahe.apply(ycrcb[:, :, 0])
        enhanced_img = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
        
    elif method == "gamma":
        # Gamma 矫正 —— 整体提亮暗部
        gamma = 1.5 # 大于1提亮，小于1变暗
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        gamma_img = cv2.LUT(img, table)
        enhanced_img = cv2.cvtColor(gamma_img, cv2.COLOR_BGR2RGB)
        
    else:
        enhanced_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 保存处理后的临时图片，方便你在文件夹里查看效果
    output_temp_path = f"temp_{method}.jpg"
    cv2.imwrite(output_temp_path, cv2.cvtColor(enhanced_img, cv2.COLOR_RGB2BGR))
    
    # 返回 PIL Image 对象供 Transformers 使用
    return Image.fromarray(enhanced_img)


# 3. VLM 推理函数
def ask_qwen(pil_image, prompt):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_image},
                {"type": "text", "text": prompt}
            ]
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[pil_image], padding=True, return_tensors="pt").to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=150)
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        
    return processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()


# 4. 主程序
if __name__ == "__main__":
    # 🌟 找一张故意拍得很暗、或者夜间的交通/车辆图片，命名为 dark_scene.jpg
    test_image = "dark_scene.jpg" 
    
    if not os.path.exists(test_image):
        print(f"⚠️ 未找到 {test_image}，请准备一张暗光环境的图片以体现 CV 算法的优势！")
        # 如果没有，先临时用 traffic.jpg 代替
        test_image = "traffic.jpg"
        if not os.path.exists(test_image):
            test_image = "cat.jpg"

    # 测试问题：要求模型识别暗光环境下的细节
    prompt = "请仔细辨认并列出图片中包含的所有交通工具、行人和障碍物。如果看不清，请直接说明。"

    methods = ["raw", "clahe", "gamma"]
    results = {}

    for m in methods:
        print(f"正在执行【{m.upper()} 增强】并进行大模型推理...")
        try:
            # 1. 调用 OpenCV 增强
            pil_img = enhance_image(test_image, method=m)
            # 2. 送入 Qwen 运行
            response = ask_qwen(pil_img, prompt)
            results[m] = response
            print(f"【{m.upper()} 结果】:\n{response}\n" + "-"*40)
        except Exception as e:
            print(f"处理方法 {m} 失败: {e}")

    # 5. 生成实验报告表格
    report_file = "cv_enhancement_vlm_report.md"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("# OpenCV 图像增强对 VLM 推理质量影响报告\n\n")
        f.write(f"**核心测试提示词 (Prompt):** `{prompt}`\n\n")
        
        # 写入对比表格
        f.write("| 图像处理方法 | 生成的临时图片 | 大模型回答质量与观察结果 |\n")
        f.write("| :--- | :--- | :--- |\n")
        f.write(f"| **原图 (Raw)** | `temp_raw.jpg` | {results.get('raw', '测试失败')} |\n")
        f.write(f"| **CLAHE (局部均衡)** | `temp_clahe.jpg` | {results.get('clahe', '测试失败')} |\n")
        f.write(f"| **Gamma 矫正 (提亮)** | `temp_gamma.jpg` | {results.get('gamma', '测试失败')} |\n\n")
        
        f.write("\n## 实验总结与分析建议\n")
        f.write("1. **原图组**：由于环境昏暗，模型通常会回答‘画面较暗，无法分辨细节’或遗漏暗部的目标。\n")
        f.write("2. **CLAHE 组**：局部对比度拉开后，原本隐藏在暗影里的汽车轮廓或边缘被放大，模型能数出更多的物体。\n")
        f.write("3. **Gamma 组**：整体亮度提升，有助于模型认清暗处的颜色和文字。\n")

    print(f"\n🎉 传统CV + 现代VLM 融合实验完成！成果已导出至表格报告: {os.path.abspath(report_file)}")