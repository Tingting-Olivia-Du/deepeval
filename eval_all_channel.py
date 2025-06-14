from dotenv import load_dotenv
load_dotenv()

import os
import json
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, HallucinationMetric

# ✅ 自适应截断：确保字符长度不超过限制
def adaptive_truncate(text: str, max_tokens: int = 2000, max_char_len: int = 8000) -> str:
    while max_tokens > 200:
        truncated = text[:max_tokens * 4]
        if len(truncated) <= max_char_len:
            return truncated
        max_tokens = int(max_tokens * 0.8)
    return text[:max_char_len]

# ✅ 项目路径设置
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
output_root = os.path.join(ROOT_DIR, "output")
retrieved_root = os.path.join(ROOT_DIR, "extracted_info")
save_root = os.path.join(ROOT_DIR, "eval_results")
os.makedirs(save_root, exist_ok=True)

# ✅ 初始化评估指标
metrics = {
    "answer_relevancy": AnswerRelevancyMetric(),
    "faithfulness": FaithfulnessMetric(),
    "hallucination": HallucinationMetric()
}

# ✅ 已评估频道列表
existing_results = {
    fname.replace("_all_metrics.json", "") for fname in os.listdir(save_root)
    if fname.endswith("_all_metrics.json")
}

# ✅ 格式化 gen_data 中的值
def format_value(v):
    if isinstance(v, str):
        return v
    elif isinstance(v, list) and all(isinstance(x, str) for x in v):
        return "; ".join(v)
    else:
        return None  # 忽略其他类型

# ✅ 遍历所有频道
for video_id in os.listdir(output_root):
    if video_id in existing_results:
        print(f"⏭️ 已存在评估结果，跳过: {video_id}")
        continue

    video_path = os.path.join(output_root, video_id)
    if not os.path.isdir(video_path):
        continue

    print(f"\n📺 正在评估频道: {video_id}")
    retrieved_file = os.path.join(retrieved_root, f"{video_id}_extract.json")
    if not os.path.exists(retrieved_file):
        print(f"❌ 缺少 retrieved_context 文件: {retrieved_file}")
        continue

    # === 加载上下文并截断 ===
    with open(retrieved_file, "r") as f:
        retrieved_data = json.load(f)
    all_desc = [v.get("description", "") for v in retrieved_data.values() if isinstance(v, dict)]
    retrieved_context = adaptive_truncate("\n".join(all_desc).strip(), max_tokens=1500, max_char_len=6000)
    print(f"🧠 retrieved_context 长度: {len(retrieved_context)} 字符")

    # === 遍历模型输出 ===
    channel_results = {}
    for fname in os.listdir(video_path):
        if not fname.endswith(".json") or "_metrics.json" in fname:
            continue

        model_key = fname.replace(f"{video_id}_", "").replace("_free_analysis.json", "")
        generated_file = os.path.join(video_path, fname)

        with open(generated_file, "r") as f:
            gen_data = json.load(f)

        # ✅ 处理 str / List[str]
        generated_output = "\n".join([
            f"{k}: {formatted}" for k, v in gen_data.items()
            if (formatted := format_value(v)) is not None
        ]).strip()

        # ✅ 截断生成内容并输出长度
        generated_output = adaptive_truncate(generated_output, max_tokens=2000, max_char_len=8000)
        print(f"📝 generated_output 长度: {len(generated_output)} 字符")

        if not generated_output:
            print(f"⚠️ 生成内容为空，跳过: {fname}")
            continue

        test_case = LLMTestCase(
            input=retrieved_context,
            actual_output=generated_output,
            retrieval_context=[retrieved_context],
            context=[retrieved_context]
        )

        result = {}
        for name, metric in metrics.items():
            try:
                score = metric.measure(test_case)
                print(f"   📊 {model_key} | {name:<20} → Score: {score:.4f}")
                result[name] = float(score)
            except Exception as e:
                print(f"   ❌ {model_key} | {name:<20} failed: {e}")
                result[name] = None

        channel_results[model_key] = result

    # ✅ 仅在存在有效模型时保存结果
    if channel_results:
        save_path = os.path.join(save_root, f"{video_id}_all_metrics.json")
        with open(save_path, "w") as f:
            json.dump(channel_results, f, indent=2)
        print(f"✅ 已保存评估结果: {save_path}")
    else:
        print(f"⚠️ 所有模型都被跳过，未保存评估文件: {video_id}")
