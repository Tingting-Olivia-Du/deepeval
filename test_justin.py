from dotenv import load_dotenv
load_dotenv()

import os
import json
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.metrics import FaithfulnessMetric
from deepeval.metrics import HallucinationMetric

# ✅ 可选：截断 context 长度
def truncate_context(context: str, max_tokens: int = 2000) -> str:
    return context[:max_tokens * 4]

# ✅ 设置 project 根目录路径（tingting_intern）
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# === ✅ 初始化所有 metric ===
metrics = {
    "answer_relevancy": AnswerRelevancyMetric(),
    "faithfulness": FaithfulnessMetric(),
    "hallucination": HallucinationMetric()
}

# === ✅ 路径和模型名设置 ===
video_id = "justinbieber"
model_key = "justinbieber_mistralai_mistral-small-3.1-24b-instruct_free_analysis.json"

retrieved_file = os.path.join(ROOT_DIR, "extracted_info", f"{video_id}_extract.json")
generated_file = os.path.join(ROOT_DIR, "output", video_id, model_key)
output_path = os.path.join(ROOT_DIR, "eval_results", f"{video_id}_mistral_all_metrics.json")
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# === ✅ 加载 retrieved context
with open(retrieved_file, "r") as f:
    retrieved_data = json.load(f)
retrieved_context = truncate_context(
    "\n".join([v.get("description", "") for v in retrieved_data.values() if isinstance(v, dict)])
).strip()

# === ✅ 加载 generated output
with open(generated_file, "r") as f:
    gen_data = json.load(f)
generated_output = "\n".join([f"{k}: {v}" for k, v in gen_data.items() if isinstance(v, str)]).strip()


# === ✅ 构造 LLMTestCase，包括 retrieval_context 和 context 字段（都要是 list[str]）
test_case = LLMTestCase(
    input=retrieved_context,
    actual_output=generated_output,
    retrieval_context=[retrieved_context],  # ✅ 修正为 list[str]
    context=[retrieved_context]             # ✅ 修正为 list[str]
)


# === ✅ 执行所有指标评估
results = {}
for name, metric in metrics.items():
    try:
        score = metric.measure(test_case)
        print(f"📊 {name:<20} → Score: {score:.4f}")
        results[name] = float(score)
    except Exception as e:
        print(f"❌ {name:<20} failed: {e}")
        results[name] = None

# === ✅ 保存评估结果
with open(output_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\n✅ 所有指标结果已保存到: {output_path}")
