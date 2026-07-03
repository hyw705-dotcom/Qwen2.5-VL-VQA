import gradio as gr
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

# 1. 模型初始化
model_path = r"E:\Qwen\models\Qwen2.5-VL-7B-Instruct"

print("正在加载 Qwen2.5-VL 视觉大模型...")
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

# 🌟 显存深度优化配置
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.float16,             # 如果老显卡不支持 bfloat16，强制用 float16 更稳
    device_map="auto",                     # 自动分配设备
    low_cpu_mem_usage=True,                # 优化CPU内存占用
    trust_remote_code=True
)
model.eval()
print("模型加载成功！Gradio 界面正在启动...\n" + "="*50)


# 2. 核心多模态多轮对话引擎
def model_chat(history, image, message):
    if not message and not image:
        return history, ""

    messages = []
    
    # 还原历史记录
    for msg in history:
        role = msg.get("role")
        content = msg.get("content")
        
        if isinstance(content, dict) and "path" in content:
            try:
                hist_img = Image.open(content["path"]).convert("RGB")
                messages.append({"role": role, "content": [{"type": "image", "image": hist_img}]})
            except Exception:
                continue
        elif isinstance(content, str):
            messages.append({"role": role, "content": [{"type": "text", "text": content}]})

    # 构建当前轮的输入
    current_content = []
    
    # 只有当这是新对话的第一轮，且用户传了图时，才把图塞给模型
    if image is not None and len(history) == 0:
        pil_img = Image.open(image).convert("RGB")
        current_content.append({"type": "image", "image": pil_img})
    else:
        pil_img = None

    current_content.append({"type": "text", "text": message})
    messages.append({"role": "user", "content": current_content})

    # 应用聊天模板
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    if image is not None:
        active_img = Image.open(image).convert("RGB")
        inputs = processor(text=[text], images=[active_img], padding=True, return_tensors="pt")
    else:
        inputs = processor(text=[text], padding=True, return_tensors="pt")
        
    inputs = inputs.to(model.device)

    # 推理生成
    with torch.no_grad():
        # 🌟 优化点：限制 max_new_tokens=128，减少显存压力，生成速度会大幅提升
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        
    bot_response = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()

    if history is None:
        history = []
        
    if image is not None and len(history) == 0:
        history.append({"role": "user", "content": {"path": image}})
    
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": bot_response})

    return history, ""


def quick_task(image, task_type):
    if image is None:
        return [{"role": "assistant", "content": "⚠️ 系统提示：请先在左侧上传一张图片！"}], ""
        
    prompt_map = {
        "Caption": "请简要描述一下这张图片中所展现的物体和主要内容。",  # 改为简要，加速推理
        "OCR": "请识别并精准提取出图片中包含的所有文字或招牌。直接输出文本。",
        "VQA": "我准备对这张图进行视觉问答，请问你准备好了吗？"
    }
    
    empty_history = []
    new_history, _ = model_chat(empty_history, image, prompt_map[task_type])
    return new_history, ""


# 3. Gradio 界面布局设计
with gr.Blocks(title="Qwen2.5-VL 多模态智能工作台") as demo:
    gr.Markdown("# 🤖 Qwen2.5-VL 多模态视觉大模型交互工作台")
    gr.Markdown("支持多轮对话、全景描述(Caption)、文字提取(OCR)与图像视觉问答(VQA)。")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📸 1. 上传测试图片")
            input_image = gr.Image(type="filepath", label="支持拖拽或点击上传")
            
            gr.Markdown("### ⚡ 快捷算法任务")
            with gr.Row():
                btn_caption = gr.Button("🖼️ Image Caption", variant="secondary")
                btn_ocr = gr.Button("🔍 OCR 文本提取", variant="secondary")
                btn_vqa = gr.Button("❓ 触发 VQA 问答", variant="secondary")

        with gr.Column(scale=2):
            gr.Markdown("### 💬 2. 视觉多轮对话")
            chatbot = gr.Chatbot(label="Qwen2.5-VL 对话历史", height=450)
            
            with gr.Row():
                input_text = gr.Textbox(
                    show_label=False, 
                    placeholder="在这输入你关于图片的问题，按回车或点击发送...", 
                    scale=4
                )
                btn_submit = gr.Button("发送", variant="primary", scale=1)
                btn_clear = gr.Button("清空上下文", variant="stop", scale=1)

    # 绑定事件
    btn_submit.click(model_chat, [chatbot, input_image, input_text], [chatbot, input_text])
    input_text.submit(model_chat, [chatbot, input_image, input_text], [chatbot, input_text])
    
    btn_caption.click(quick_task, [input_image, gr.State("Caption")], [chatbot, input_text])
    btn_ocr.click(quick_task, [input_image, gr.State("OCR")], [chatbot, input_text])
    btn_vqa.click(quick_task, [input_image, gr.State("VQA")], [chatbot, input_text])
    
    btn_clear.click(lambda: ([], "", None), None, [chatbot, input_text, input_image])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False, theme=gr.themes.Soft())