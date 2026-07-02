"""Round 11-H — LLM 因子推理 prompt 生态(§2.12/2.14/2.15/2.16)。

只测纯代码侧:模板完备性、prompt 组装、LLM 回复解析、批量激活门,
以及辩论 prompt 能直接消化 crowding_score / decay_warning / alpha_eval 的真实输出。
"""
import re

import numpy as np
import pandas as pd
import pytest

from scripts.research import llm_factor_reasoning as lfr

# 只匹配 {snake_case} 形式的占位符;模板中的 JSON 示例都是 {"key": ...} 形式,不会误伤
PLACEHOLDER_RE = re.compile(r"\{[a-z_]+\}")

TEMPLATES = {
    "factor_reasoning.txt": ["{factor_logic}", "{market_context}", "{news_items}"],
    "factor_debate.txt": ["{factor_name}", "{stats_summary}", "{crowding_summary}",
                          "{decay_summary}", "{scorecard_summary}"],
    "hypothesis_alignment.txt": ["{factor_expr}", "{hypothesis}"],
    "factor_interpretability.txt": ["{factor_description}"],
}


def test_templates_exist_with_all_placeholders():
    for name, phs in TEMPLATES.items():
        path = lfr.PROMPTS_DIR / name
        assert path.is_file(), f"missing template {path}"
        txt = path.read_text(encoding="utf-8")
        for ph in phs:
            assert ph in txt, f"{name} lacks {ph}"
        # 模板中除声明的占位符外,不得有别的 {snake_case} token
        assert set(PLACEHOLDER_RE.findall(txt)) == set(phs)


def test_build_factor_reasoning_prompt_assembly_and_truncation():
    long_news = "监管新规冲击高频交易 " * 100          # 远超单条上限
    news = [long_news] + [f"新闻{i}" for i in range(20)]  # 共 21 条,超条数上限
    p = lfr.build_factor_reasoning_prompt(
        "12-1 动量:过去 12 个月剔除最近 1 个月的收益率排序",
        {"vol_regime": "high", "trend": "bear", "vix": 31.5},
        news)
    assert not PLACEHOLDER_RE.search(p)              # 无残留占位符
    assert "12-1 动量" in p and "vol_regime" in p and "31.5" in p
    assert "监管新规冲击高频交易" in p and "…" in p    # 超长条被截断
    assert "已截断" in p                              # 超出条数如实标注
    assert "relevance_score" in p and "regime_dependence" in p


def test_build_alignment_and_interpretability_prompts():
    a = lfr.build_alignment_prompt("-ts_rank(volume, 10)", "缩量意味着抛压衰竭,后市看涨")
    assert not PLACEHOLDER_RE.search(a)
    assert "-ts_rank(volume, 10)" in a and "抛压衰竭" in a and "alignment_score" in a
    b = lfr.build_interpretability_prompt("close / delay(close, 231) 的 5 日均值")
    assert not PLACEHOLDER_RE.search(b)
    assert "delay(close, 231)" in b and "clarity_score" in b and "red_flags" in b


def test_parse_llm_json_plain_fenced_and_chatty():
    want = ["relevance_score", "activation", "reasoning", "regime_dependence"]
    raw = ('{"relevance_score": 0.72, "activation": true, '
           '"reasoning": "含 {大括号} 的理由", "regime_dependence": "高波有效"}')
    assert lfr.parse_llm_json(raw, want)["relevance_score"] == 0.72
    fenced = "```json\n" + raw + "\n```"
    assert lfr.parse_llm_json(fenced, want)["activation"] is True
    chatty = "好的,我分析如下{这不是json:\n" + raw + "\n以上就是我的判断。"
    out = lfr.parse_llm_json(chatty, want, clamp01=["relevance_score"])
    assert out["reasoning"].startswith("含")


def test_parse_llm_json_missing_key_raises_listing_missing():
    with pytest.raises(ValueError, match="regime_dependence"):
        lfr.parse_llm_json('{"relevance_score": 0.5}',
                           ["relevance_score", "regime_dependence"])
    with pytest.raises(ValueError):
        lfr.parse_llm_json("完全没有 JSON 的回复", ["a"])


def test_parse_llm_json_clamp01_and_type_check():
    out = lfr.parse_llm_json('{"s": 1.7, "t": -0.3}', ["s", "t"], clamp01=["s", "t"])
    assert out["s"] == 1.0 and out["t"] == 0.0
    with pytest.raises(ValueError, match="numeric"):
        lfr.parse_llm_json('{"s": "high"}', ["s"], clamp01=["s"])
    with pytest.raises(ValueError, match="numeric"):
        lfr.parse_llm_json('{"s": true}', ["s"], clamp01=["s"])


def test_context_aware_factor_gate():
    factors = {"mom": "动量", "val": "价值", "rev": "反转", "size": "市值"}
    parsed = {
        "mom": {"relevance_score": 0.9, "activation": True, "reasoning": "趋势延续"},
        "val": {"relevance_score": 0.15, "activation": True, "reasoning": "利率环境不利"},
        "rev": {"relevance_score": 0.8, "activation": False, "reasoning": "新规冲击反转前提"},
        # size 无 LLM 判定 → 默认保留
    }
    out = lfr.context_aware_factor_gate(factors, parsed, min_relevance=0.4)
    assert out["active"] == ["mom", "size"]
    deact = dict(out["deactivated"])
    assert set(deact) == {"val", "rev"}
    assert "0.15" in deact["val"] and "利率环境不利" in deact["val"]
    assert "新规冲击反转前提" in deact["rev"]


def test_debate_prompt_consumes_real_upstream_outputs():
    from scripts.research.crowding import crowding_score
    from scripts.research.decay_monitor import decay_warning
    from scripts.research.factor_lab import alpha_eval

    np.random.seed(3)
    idx = pd.date_range("2022-01-03", periods=140, freq="B")
    cols = [f"S{i}" for i in range(8)]
    close = pd.DataFrame(
        100 * np.exp(np.cumsum(np.random.normal(0.0002, 0.012, (140, 8)), axis=0)),
        index=idx, columns=cols)
    fac = close.pct_change(21)
    other = close.rolling(5).mean() / close - 1.0

    crowd = crowding_score(fac, {"other": other}, close)          # 真实 §2.3 输出
    decay = decay_warning(fac, close, rebalance_days=21)          # 真实 §2.4 输出
    stats = alpha_eval(fac, close, horizon=5)                     # 真实 §2.16 输出
    card = {"ic": 0.011, "ic_ir": 0.2, "autocorr": 0.95,
            "coverage": 0.85, "max_corr_existing": 0.3,
            "verdict": "weak predictive power; unstable IC"}

    p = lfr.build_debate_prompt("mom21", stats, crowd, decay, card)
    assert not PLACEHOLDER_RE.search(p)
    assert "mom21" in p
    # 上游各块的关键字段都进入了辩论输入
    assert "holdings_overlap" in p or "return_correlation" in p   # crowding components
    assert "half_life" in p and "recent_ic" in p                  # decay_warning 字段
    assert "predictive" in p and "robustness" in p                # alpha_eval 维度
    assert "weak predictive power" in p                            # scorecard verdict
    # alpha_eval 附带的长文本 prompt 不得混入辩论证据块
    assert "interpretability_prompt" not in p
    assert "consensus_score" in p and '"keep"' in p
