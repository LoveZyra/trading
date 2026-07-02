"""LLM 因子推理 prompt 生态(优化方案 v3 §2.12 / §2.14 / §2.15 / §2.16)。

设计原则(roadmap 原则 5):skill 侧只有 prompt 模板 + 输入组装 + 输出解析的
**纯代码**;LLM 调用永远由 agent(Claude)侧执行。本模块不 import 任何网络 /
模型依赖,离线可测。

v2/v3 边界(§2.15 核验注):
  - FactorMAD 原文是多 Agent 辩论式**因子挖掘**框架,本模块只借其**评审辩论**
    一环——单次会话内由 agent 扮演多视角做一轮语义综合(成本 ≈ 一次推理);
  - v2(2026-06-25)否决的是"N× API 成本的自动挖掘循环",本模块不做、也不为
    其提供任何循环钩子;
  - 推理 / 辩论输出只作附注,纯代码指标(§2.3 拥挤、§2.4 衰减、§2.16 四维)
    永远是主判定。

agent 使用范式(三步,所有 build_* 相同):
  1. prompt = build_xxx_prompt(...)                       # skill 纯代码组装
  2. text   = <agent(Claude)对 prompt 做一次推理>          # LLM 调用在 agent 侧
  3. out    = parse_llm_json(text, required_keys=[...], clamp01=[...])  # skill 解析
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

# scripts/research/llm_factor_reasoning.py -> skill 根目录 / prompts
PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"

_MAX_NEWS_ITEMS = 15     # 新闻条数上限:再多只是烧 token,不增加判定信息
_MAX_NEWS_CHARS = 280    # 单条新闻截断:标题+导语足够做失效判断


# ---------------------------------------------------------------------------
# 模板装载与填充
# ---------------------------------------------------------------------------

def _load_template(name: str) -> str:
    path = PROMPTS_DIR / name
    if not path.is_file():
        raise FileNotFoundError(f"prompt template missing: {path}")
    return path.read_text(encoding="utf-8")


def _fill(template: str, mapping: dict) -> str:
    """只替换已知的 {key} 占位符。不用 str.format:模板里的 JSON 输出示例含大量
    字面大括号,format 会把它们当占位符炸掉;逐 token replace 最诚实。"""
    out = template
    for k, v in mapping.items():
        token = "{" + k + "}"
        if token not in out:
            raise ValueError(f"template placeholder {token} not found in template")
        out = out.replace(token, str(v))
    return out


def _clean_line(s, limit: int) -> str:
    """自由文本消毒:压掉换行 / 控制字符、去大括号(防伪造占位符)、截断超长——
    prompt 的结构不能被输入内容撑破。"""
    s = re.sub(r"[\x00-\x1f\x7f]+", " ", str(s))
    s = s.replace("{", "(").replace("}", ")").strip()
    if len(s) > limit:
        s = s[: limit - 1].rstrip() + "…"
    return s


def _truncate(s, limit: int) -> str:
    """多行文本(因子逻辑 / 表达式 / 假设)只截断,不压换行。"""
    s = str(s).strip()
    return s if len(s) <= limit else s[: limit - 1].rstrip() + "…"


def _fmt_val(v) -> str:
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        if math.isnan(v):
            return "NaN"
        if math.isinf(v):
            return "inf" if v > 0 else "-inf"
        return f"{v:.4f}"
    if isinstance(v, dict):
        return "{" + ", ".join(f"{k}: {_fmt_val(x)}" for k, x in v.items()) + "}"
    if isinstance(v, (list, tuple)):
        return "[" + "; ".join(_fmt_val(x) for x in v) + "]"
    return _clean_line(v, 400)


def _fmt_kv_block(d: dict | None, empty: str = "(无数据)") -> str:
    """把纯代码指标 dict 排成 '- key: value' 行;缺数据如实标注而不是留空——
    LLM 面对空白会脑补,面对"(无数据)"才会按证据不足处理。"""
    if not d:
        return empty
    return "\n".join(f"- {k}: {_fmt_val(v)}" for k, v in d.items())


# ---------------------------------------------------------------------------
# build_* —— prompt 组装(纯代码)
# ---------------------------------------------------------------------------

def build_factor_reasoning_prompt(factor_logic: str, market_context: dict,
                                  news_items: list) -> str:
    """§2.12 Alpha-R1 式因子相关性推理 prompt:因子逻辑 + 市场环境 + 新闻 →
    LLM 判断经济逻辑是否仍成立 / 是否被新闻触发失效。

    news_items 每条截断到 280 字符、最多 15 条(超出如实标注截断条数),
    并清洗控制字符与大括号,防止输入内容破坏 prompt 结构。

    agent 使用范式:
      prompt = build_factor_reasoning_prompt(logic, {"vol_regime": "high"}, news)
      text   = <agent(Claude)对 prompt 推理一次>
      out    = parse_llm_json(text, ["relevance_score", "activation", "reasoning",
                                     "regime_dependence"], clamp01=["relevance_score"])
    """
    items = [_clean_line(n, _MAX_NEWS_CHARS)
             for n in list(news_items or [])[:_MAX_NEWS_ITEMS]]
    dropped = max(0, len(news_items or []) - _MAX_NEWS_ITEMS)
    news_block = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(items)) \
        if items else "(无新闻输入)"
    if dropped:
        news_block += f"\n(另有 {dropped} 条超出上限,已截断)"
    return _fill(_load_template("factor_reasoning.txt"), {
        "factor_logic": _truncate(factor_logic, 2000),
        "market_context": _fmt_kv_block(dict(market_context or {}), "(无环境摘要)"),
        "news_items": news_block,
    })


def build_debate_prompt(factor_name, stats: dict, crowding: dict,
                        decay: dict, scorecard: dict) -> str:
    """§2.15 FactorMAD 式四视角辩论 prompt:把 alpha_eval / crowding_score /
    decay_warning / factor_scorecard 的**原始输出 dict** 直接喂进来,本函数负责
    格式化为四块辩论输入(嵌套 dict / NaN / inf 都能如实呈现)。

    stats 里的 interpretability_prompt(alpha_eval 附带的长文本)会被剔除——
    它是给另一个 LLM 环节用的,混进辩论输入只会污染证据块。

    agent 使用范式:
      prompt = build_debate_prompt(name, alpha_eval_out, crowding_out, decay_out, card)
      text   = <agent(Claude)对 prompt 推理一次>
      out    = parse_llm_json(text, ["verdicts", "consensus_score", "dissent",
                                     "recommendation"], clamp01=["consensus_score"])
    """
    stats = {k: v for k, v in (stats or {}).items() if k != "interpretability_prompt"}
    return _fill(_load_template("factor_debate.txt"), {
        "factor_name": _clean_line(factor_name, 120),
        "stats_summary": _fmt_kv_block(stats),
        "crowding_summary": _fmt_kv_block(crowding),
        "decay_summary": _fmt_kv_block(decay),
        "scorecard_summary": _fmt_kv_block(scorecard),
    })


def build_alignment_prompt(expr, hypothesis) -> str:
    """§2.14 hypothesis alignment prompt:因子表达式 vs 研究假设的语义一致性。

    agent 使用范式:
      prompt = build_alignment_prompt("-ts_rank(volume, 10)", "缩量意味着抛压衰竭")
      text   = <agent(Claude)对 prompt 推理一次>
      out    = parse_llm_json(text, ["alignment_score", "mismatch"],
                              clamp01=["alignment_score"])
    """
    return _fill(_load_template("hypothesis_alignment.txt"), {
        "factor_expr": _truncate(expr, 2000),
        "hypothesis": _truncate(hypothesis, 2000),
    })


def build_interpretability_prompt(expr_or_logic) -> str:
    """§2.16 可解释性 prompt:因子表达式 / 逻辑描述 → 经济学故事 + 红旗清单。
    与 factor_lab.alpha_eval 附带的 interpretability_prompt 同一评判口径,
    这里是独立可复用版本(输出为严格 JSON,便于 parse_llm_json 回收)。

    agent 使用范式:
      prompt = build_interpretability_prompt("close/delay(close,231) 的 5 日均值")
      text   = <agent(Claude)对 prompt 推理一次>
      out    = parse_llm_json(text, ["clarity_score", "economic_story", "red_flags"],
                              clamp01=["clarity_score"])
    """
    return _fill(_load_template("factor_interpretability.txt"), {
        "factor_description": _truncate(expr_or_logic, 2000),
    })


# ---------------------------------------------------------------------------
# 输出解析(纯代码)
# ---------------------------------------------------------------------------

def _first_balanced_json(text: str):
    """从文本里挖出第一个能被 json.loads 接受的平衡 JSON 对象。
    括号配对时尊重字符串字面量(引号内的大括号不算),配平但 loads 失败的
    候选跳过、继续找下一个 '{'。"""
    start = text.find("{")
    while start != -1:
        depth, in_str, esc, end = 0, False, False, -1
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    break
        if end != -1:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                pass
        start = text.find("{", start + 1)
    raise ValueError("no balanced JSON object found in LLM response")


def parse_llm_json(text: str, required_keys: list, *, clamp01: list = ()) -> dict:
    """鲁棒解析 agent 拿回的 LLM 回复为 dict。

    处理链:剥掉 markdown 代码围栏(```json / ```)→ 找第一个平衡且可解析的
    JSON 对象(前后废话自动忽略)→ 缺键抛 ValueError 并列出缺了什么 →
    clamp01 里的键做数值类型校验并截断到 [0,1](bool / 字符串不算数值——
    LLM 把分数写成 "high" 这种事必须当场报错,不能静默吞掉)。
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("empty LLM response — nothing to parse")
    cleaned = re.sub(r"```[a-zA-Z0-9_-]*", "", text)
    obj = _first_balanced_json(cleaned)
    if not isinstance(obj, dict):
        raise ValueError(f"expected a JSON object, got {type(obj).__name__}")
    missing = [k for k in required_keys if k not in obj]
    if missing:
        raise ValueError(f"LLM JSON missing required keys: {missing} "
                         f"(got keys: {sorted(obj.keys())})")
    for k in clamp01:
        if k not in obj:
            continue
        v = obj[k]
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError(f"key {k!r} must be numeric for clamp01, "
                             f"got {type(v).__name__}: {v!r}")
        obj[k] = float(min(1.0, max(0.0, float(v))))
    return obj


# ---------------------------------------------------------------------------
# 批量激活门(纯代码,消费 agent 拿回的解析结果)
# ---------------------------------------------------------------------------

def context_aware_factor_gate(factors: dict, parsed_responses: dict, *,
                              min_relevance: float = 0.4) -> dict:
    """§2.12 批量激活/停用决策(纯代码,不含任何 LLM 调用)。

    factors           : {name: logic_str} 因子池;
    parsed_responses  : {name: parse_llm_json 的结果}(即 agent 对每个因子跑完
                        build_factor_reasoning_prompt → 推理 → 解析后的 dict);
    停用条件(任一):LLM 判 activation=false;或 relevance_score < min_relevance。
    没有对应 response 的因子默认保留 active——LLM 推理只是附注,无判定时
    不应凭空停用(纯代码指标才是主判定,见模块 docstring 的 v2/v3 边界)。

    返回 {"active": [names], "deactivated": [(name, reason)]}。
    """
    active: list = []
    deactivated: list = []
    for name in factors:
        resp = (parsed_responses or {}).get(name)
        if not resp:
            active.append(name)
            continue
        try:
            rel = float(resp.get("relevance_score"))
        except (TypeError, ValueError):
            rel = float("nan")
        if not bool(resp.get("activation", True)):
            reason = str(resp.get("reasoning") or "").strip() \
                or "LLM 判定 activation=false"
            deactivated.append((name, reason))
        elif math.isfinite(rel) and rel < float(min_relevance):
            reason = f"relevance_score {rel:.2f} < min_relevance {min_relevance:g}"
            extra = str(resp.get("reasoning") or "").strip()
            if extra:
                reason += f" — {extra}"
            deactivated.append((name, reason))
        else:
            active.append(name)
    return {"active": active, "deactivated": deactivated}
