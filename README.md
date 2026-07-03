##  Qwen2.5-VL 多模态视觉理解与全栈问答系统

本项目基于阿里巴巴开源的最新一代多模态大模型 **Qwen2.5-VL-7B-Instruct** 构建（模型托管于 Hugging Face 官方路径 `Qwen/Qwen2.5-VL-7B-Instruct`）。项目成功打通了从底层离线脚本推理、前置传统计算机视觉（CV）图像增强优化、中层高级提示词工程（Prompt Engineering）系统评测，到上层交互式 Web 端到端交付的全栈闭环流。本工程深度探索了多模态大模型（VLM）在端侧落地时的感知边界与控制策略，非常适合作为计算机视觉、大语言模型应用及多模态方向的深度实践案例。

本系统所依赖的多模态视觉基座模型托管于 Hugging Face 官方社区，具体的模型权重及配置文件下载路径为：
* **官方路径**：`[Qwen/Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)`
* **项目引用**：在本地部署时，系统通过 `transformers` 的从本地或云端直接索引该路径，完成 `AutoProcessor` 与 `Qwen2_5_VLForConditionalGeneration` 核心组件的加载。
## 技术栈 Python   PyTorch   Transformers  Qwen2.5-VL  OpenCV  Gradio  PIL

## 技术栈与核心依赖 

系统底层基于 PyTorch 深度学习框架，采用 `bfloat16` 与 `float16` 混合精度推理模式，并在加载时通过 `low_cpu_mem_usage=True` 与 `device_map="auto"` 实现异构硬件下的显存动态切分与极致优化。大模型管线依托 Transformers 库的 `AutoProcessor` 模块，完美对齐自适应 Vision-Language 聊天模板。前置图像处理管线引入 OpenCV-Python，利用空间算子与非线性灰度变换实现实时流式图像增强。上层全栈交互前端采用最新的 Gradio 6.0+ 响应式框架，基于全新的异步消息状态机实现多模态数据的流式渲染。

---

##  系统核心能力与四阶段实验模块

1、基准多模态推理与模型部署
基于 HuggingFace Transformers 完成 Qwen2.5-VL 的本地部署，构建统一的视觉-文本输入 pipeline，实现图像编码 + 文本生成的端到端推理流程，支持 PIL 图像输入与 chat template 推理模式。

2、 高级提示词工程 (Prompt Engineering) 鲁棒性评测
为深度分析不同提示词策略对视觉问答（VQA）任务的控制力，本模块设计并自动化对比了多种高级 Prompt 架构。其中，Zero-shot Prompt 用于测试无先验状态下模型的泛化感知本能；Few-shot Prompt 通过构建基于少样本上下文约束的结构化模板，将模型的输出格式对齐率提升至接近 100%；Role-play Prompt 则通过注入特定领域人设（如“20年经验交通规划专家”），有效激活了模型的隐性深层领域知识，显著降低了视觉幻觉率（Hallucination Rate）与拒答率。

3、传统 CV 图像增强与 VLM 协同设计 
针对真实工业场景下暗光、低对比度、恶劣光照导致的视觉语言模型感知退化问题，项目创新性地在 VLM 输入前置流中集成了 OpenCV 增强算子。系统通过部署 CLAHE（限制对比度自适应直方图均衡化） 算法拉开图像暗部的局部对比度，使模型能够清晰辨识隐藏在阴影中的目标轮廓；同时引入 Gamma 矫正（非线性亮度校正） 动态平滑极限光照，显著提升了模型在夜间密集 OCR 文本提取与复杂物体计数任务中的推理稳定性。

4、 交互式端到端闭环系统 (Gradio UI)
基于最新版 Gradio 响应式前端框架构建了端到端的多模态交互式智能工作台。系统全面适配了 Gradio 全新字典态消息数据流（Role-Content Dict Flow），在前端完美支持了能够维持图像上下文的高带宽多轮图文深度对话。此外，前端界面还无缝集成了全景图像描述（Image Caption）、高精度文字提取（OCR）与开放式视觉问答（VQA）的三路快捷分流触发器，具备极强的工业级全栈交付与可视化演示价值。

---

##  实验结论摘要

* **提示词敏感度**：结构化约束和角色注入对控制 VLM 行为至关重要。Role-play 能让模型在垂直场景下的词汇专业度大幅跃升，而 Few-shot 是保证输出格式不跑偏的最优解。
* **前置 CV 的必要性**：实验结果表明，前置图像增强（尤其是 CLAHE 算子）能有效拓宽视觉大模型的感知边界，将暗部细小目标的幻觉率明显降低，验证了“传统 CV + 现代大模型”协同优化的商业落地潜力。
* **任务稳定性评估**：Qwen2.5-VL 在处理复杂商业招牌、多语种密集文本的自然场景 OCR 提取，以及开放式多轮对话中表现极其鲁棒，具备极高的实际端侧应用价值。

---

## 🚀 快速开始与复现 (Quick Start)

### 1. 环境准备与依赖安装
确保您的机器拥有 NVIDIA 显卡并已配置好 CUDA 环境，激活您的虚拟环境后执行以下命令安装核心依赖：
pip install -r requirements.txt
python app.py
```bash
pip install torch transformers accelerate opencv-python gradio>=6.0 pillow
