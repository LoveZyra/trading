
/* ============================================================================
   alpha-forge · 报告渲染器
   Reads a single report object (schema in SCHEMA.md) and builds the DOM.
   Every section is optional — present a key, it renders; omit it, it's skipped.
   ========================================================================== */
(function (global) {
  "use strict";

  /* ---- tiny DOM helper ------------------------------------------------- */
  // Minimal HTML hygiene for DATA-derived strings. Reports are built by the model from
  // curated data, but news / web / AI text can carry stray markup. This strips the
  // genuinely dangerous constructs (script & friends, inline on*= handlers, javascript:
  // URLs) while leaving the benign inline formatting the schema documents (<b>/<i>/<span
  // class=…>) intact. Code-controlled markup (the SVG charts) bypasses this via `raw:`.
  function safeHtml(s) {
    if (s == null) return "";
    s = String(s)
      .replace(/<\s*\/?\s*(script|style|iframe|object|embed|link|meta|svg|img|video|audio|base|form|input)\b[^>]*>/gi, "")
      .replace(/\son\w+\s*=\s*("[^"]*"|'[^']*'|[^\s>]+)/gi, "")
      .replace(/(href|src)\s*=\s*("\s*javascript:[^"]*"|'\s*javascript:[^']*'|javascript:[^\s>]+)/gi, '$1="#"');
    return s;
  }
  function el(tag, attrs, children) {
    const n = document.createElement(tag);
    if (attrs) for (const k in attrs) {
      if (k === "class") n.className = attrs[k];
      else if (k === "html") n.innerHTML = safeHtml(attrs[k]);   // data -> sanitized
      else if (k === "raw") n.innerHTML = attrs[k];              // code-controlled markup (SVG)
      else if (k === "style") n.setAttribute("style", attrs[k]);
      else if (attrs[k] != null) n.setAttribute(k, attrs[k]);
    }
    if (children != null) (Array.isArray(children) ? children : [children]).forEach(function (c) {
      if (c == null) return;
      n.appendChild(typeof c === "string" || typeof c === "number" ? document.createTextNode(String(c)) : c);
    });
    return n;
  }

  /* ---- number formatting ---------------------------------------------- */
  function commas(x) { return String(x).replace(/\B(?=(\d{3})+(?!\d))/g, ","); }
  function price(v) {
    if (v == null) return "—";
    const a = Math.abs(v);
    const d = a >= 100 ? (Number.isInteger(v) ? 0 : 1) : a >= 10 ? 2 : 2;
    return commas(Number(v).toFixed(d));
  }
  // signed percent where value is already in percent units (e.g. 3.27 -> "+3.27%")
  function pctSigned(v, dp) {
    if (v == null) return "—";
    const s = (v > 0 ? "+" : "") + Number(v).toFixed(dp == null ? (Math.abs(v) >= 100 ? 0 : 1) : dp) + "%";
    return s;
  }
  function signedSpan(v, dp) {
    if (v == null) return el("span", { class: "muted" }, "—");
    return el("span", { class: "num " + (v > 0 ? "pos" : v < 0 ? "neg" : "muted") }, pctSigned(v, dp));
  }
  function scoreStr(v, dp) {
    if (v == null) return "—";
    return (v > 0 ? "+" : "") + Number(v).toFixed(dp == null ? 2 : dp);
  }
  function pct01(x) { return Math.max(0, Math.min(1, x)) * 100; }
  function pctStr(v, dp) { dp = dp == null ? 1 : dp; return (v > 0 ? "+" : "") + Number(v).toFixed(dp) + "%"; }
  function vsPrice(level, p) { return (p != null && p > 0 && level != null) ? pctStr((level / p - 1) * 100) : null; }
  function clampPos(p) { return Math.max(4, Math.min(96, p)); }

  /* ---- score meter (−1 … +1) ------------------------------------------ */
  function scoreMeter(value, opts) {
    opts = opts || {};
    const min = opts.min == null ? -1 : opts.min, max = opts.max == null ? 1 : opts.max;
    const pos = pct01((value - min) / (max - min));
    return el("div", { class: "meter" }, [
      el("div", { class: "track" }, [
        el("span", { class: "zero" }),
        el("span", { class: "needle", style: "left:" + pos + "%" })
      ]),
      el("div", { class: "scale" }, [el("span", null, String(min)), el("span", null, "0"), el("span", null, "+" + max)]),
      opts.label ? el("div", { class: "label", html: opts.label }) : null
    ]);
  }

  /* ---- R/R inline bar -------------------------------------------------- */
  function rrColor(rr) {
    return rr >= 1.8 ? "var(--pos)" : rr >= 1.2 ? "var(--accent)" : rr >= 0.9 ? "var(--warn)" : "var(--neg)";
  }
  function rrBar(rr) {
    if (rr == null) return el("span", { class: "muted" }, "—");
    const w = pct01(rr / 3);
    return el("span", { class: "rrbar" }, [
      el("span", { class: "bar" }, el("i", { style: "width:" + w + "%;background:" + rrColor(rr) })),
      el("span", { class: "v", style: "color:" + rrColor(rr) }, Number(rr).toFixed(2))
    ]);
  }

  /* ---- signal badge ---------------------------------------------------- */
  const SIG = { long: ["long", "做多"], watch: ["watch", "观望"], short: ["short", "做空"] };
  function sigBadge(s, textOverride) {
    // Two badge styles, both supported:
    //  (1) technical codes long/watch/short -> 做多/观望/做空 (legacy).
    //  (2) event-sentiment strings judged by the AI from news, optionally with a holding
    //      period, e.g. "利多·短线" / "利多·中线" / "利空·短线" / "中性". Colored by the
    //      sentiment word (利多/利好→red, 利空→green, else amber); the text shows verbatim.
    var m = SIG[s], cls, label;
    if (m) { cls = m[0]; label = m[1]; }
    else {
      var str = s || "—";
      cls = /利多|利好|偏多|做多|看多|bull/i.test(str) ? "long"
          : /利空|利淡|偏空|做空|看空|bear/i.test(str) ? "short" : "watch";
      label = str;
    }
    return el("span", { class: "sig " + cls }, textOverride || label);
  }

  /* ====================================================================== */
  /*  SECTION BUILDERS                                                      */
  /* ====================================================================== */

  function masthead(meta) {
    meta = meta || {};
    const TYPE = { single: "个股分析", portfolio: "组合 / 自选池", market: "市场扫描",
      backtest: "策略回测", attribution: "业绩归因", macro: "宏观复盘" };
    const metaItems = [];
    if (meta.market) metaItems.push(el("span", null, [el("b", null, "市场 "), meta.market]));
    if (meta.data_source) metaItems.push(el("span", null, [el("b", null, "数据 "), meta.data_source]));
    if (meta.universe) metaItems.push(el("span", null, [el("b", null, "范围 "), meta.universe]));
    if (meta.tag) metaItems.push(el("span", { class: "flagcell", style: "color:var(--flag)" }, meta.tag));
    return el("header", { class: "masthead" }, [
      el("div", { class: "kicker" }, [
        el("span", { class: "mark" }, meta.generated_by || "alpha-forge"),
        meta.report_type ? el("span", { class: "type-badge" }, TYPE[meta.report_type] || meta.report_type) : null,
        el("span", { class: "spacer" }),
        el("span", null, "量化分析报告")
      ]),
      el("h1", null, meta.title || "量化分析报告"),
      meta.subtitle ? el("p", { class: "subtitle" }, meta.subtitle) : null,
      metaItems.length ? el("div", { class: "metaline" }, metaItems) : null,
      meta.date ? el("div", { class: "stamp" }, [
        el("div", { class: "d num" }, meta.date),
        meta.weekday ? el("div", { class: "wd" }, meta.weekday) : null
      ]) : null
    ]);
  }

  function verdict(v, envScore) {
    if (!v) return null;
    // 5-level stance. Priority: explicit v.score > envScore (objective: blended regime/macro
    // meters) > NET keyword lean (bullish − bearish hits; fixes the old first-match bug).
    var lvl;
    var sc = (typeof v.score === "number") ? v.score : (typeof envScore === "number") ? envScore : null;
    if (sc !== null) {
      var x = sc;
      lvl = x >= 0.6 ? 2 : x >= 0.15 ? 1 : x <= -0.6 ? -2 : x <= -0.15 ? -1 : 0;
    } else {
      var sx = v.stance || "";
      var bull = (sx.match(/多|涨|强|升|新高|突破|反弹|看多|做多|bull|牛/gi) || []).length;
      var bear = (sx.match(/空|跌|弱|新低|破位|回落|下行|看空|做空|bear|熊/gi) || []).length;
      var net = bull - bear;
      if (net === 0) { lvl = 0; }
      else {
        var strong = (/强烈|显著|大幅|坚定|重仓|满仓|全仓/.test(sx) || Math.abs(net) >= 2);
        var temper = /超买|超卖|延伸|不追|谨慎|高位|震荡|观望|温和|分化|控制仓位|中性/.test(sx);
        var mag = (strong && !temper) ? 2 : 1;
        lvl = net > 0 ? mag : -mag;
      }
    }
    var dir = lvl > 0 ? "up" : lvl < 0 ? "down" : "flat";
    var arrow = lvl >= 2 ? "▲▲▲" : lvl === 1 ? "▲" : lvl === 0 ? "◆" : lvl === -1 ? "▼" : "▼▼▼";
    var name = lvl >= 2 ? "强烈看多" : lvl === 1 ? "偏多" : lvl === 0 ? "中性" : lvl === -1 ? "偏空" : "强烈看空";
    return el("div", { class: "verdict " + dir }, [
      el("div", { class: "stance" }, [
        el("div", { class: "lab" }, "综合立场"),
        el("div", { class: "arrow" }, arrow),
        el("div", { class: "lvl-big" }, name)
      ]),
      el("div", { class: "body-v" }, [
        v.stance ? el("div", { class: "vhead", html: v.stance }) : null,
        (v.points && v.points.length) ? el("ul", { class: "vpoints" }, v.points.map(function (pt) {
          var t = (typeof pt === "string") ? { text: pt } : (pt || {});
          return el("li", { class: "vpt" }, [
            t.icon ? el("span", { class: "vpi" }, t.icon) : null,
            el("span", { class: "vptx", html: t.text || "" })
          ]);
        })) : (v.action ? el("p", { class: "action", html: v.action }) : null),
        v.summary ? el("p", { class: "summary", html: v.summary }) : null
      ])
    ]);
  }

  function alerts(list) {
    if (!list || !list.length) return null;
    return el("div", { class: "alerts" }, list.map(function (a) {
      const top = [];
      if (a.symbol) top.push(el("span", { class: "sym" }, a.symbol));
      if (a.name) top.push(el("span", { class: "nm" }, a.name));
      if (a.hold) top.push(el("span", { class: "chip hold star" }, a.hold === true ? "持仓" : a.hold));
      if (a.signal) top.push(sigBadge(a.signal));
      return el("div", { class: "alert " + (a.level === "mid" ? "mid" : "") }, [
        el("span", { class: "dot" }),
        el("div", { class: "a-main" }, [
          top.length ? el("div", { class: "a-top" }, top) : null,
          a.headline ? el("div", { class: "headline", html: a.headline }) : null,
          a.detail ? el("div", { class: "detail", html: a.detail }) : null,
          a.action ? el("div", { class: "act", html: "<b>动作 · </b>" + a.action }) : null
        ])
      ]);
    }));
  }

  function envPanel(title, score, bodyChildren, note) {
    return el("div", { class: "panel" }, [
      el("div", { class: "p-head" }, [
        el("div", { class: "t" }, title),
        score != null ? el("div", { class: "score", style: "color:" + (score > 0.05 ? "var(--pos)" : score < -0.05 ? "var(--neg)" : "var(--warn)") }, scoreStr(score)) : null
      ]),
      el("div", { class: "p-body" }, bodyChildren.concat(note ? [el("p", { class: "p-note", html: note })] : []))
    ]);
  }

  function vCell(value, tone) {
    var t = (value == null) ? "" : String(value);
    if (!/[一-鿿]/.test(t)) return el("div", { class: "v", html: t });  // 数字/英文 -> 等宽
    var tn = tone;
    if (!tn) {
      if (/延伸|偏紧|收紧|紧张|谨慎|降温|回落|承压|高位|拥挤|峰值|震荡|分化|观望|过热|风险|回调|压力|疲软|放缓|降|弱/.test(t)) tn = "warn";
      else if (/加速|超级|强|高|利好|扩张|改善|新高|大超|暴击|领先|饱满|确定|顺风|放量|景气|增长|回暖|向好|缓和|稳健|健康|宽松/.test(t)) tn = "pos";
      else tn = "neu";
    }
    return el("div", { class: "v tag" }, el("span", { class: "vbadge " + tn }, t));
  }
  function regimePanel(r) {
    const rows = (r.rows || []).map(function (row) {
      return el("div", { class: "row" }, [
        el("div", { class: "k" }, row.item),
        vCell(row.value, row.tone),
        el("div", { class: "r", html: row.read || "" })
      ]);
    });
    const body = [scoreMeter(r.score, { label: r.label ? "<b>" + scoreStr(r.score) + "</b> · " + r.label : null })];
    if (rows.length) body.push(el("div", { class: "dl" }, rows));
    return envPanel(r.title || "📊 大盘环境", r.score, body, r.note);
  }

  function macroPanel(m) {
    const body = [scoreMeter(m.risk_score, { label: m.label ? "<b>" + scoreStr(m.risk_score) + "</b> · " + m.label : null })];
    const dl = [];
    if (m.vix != null) dl.push(el("div", { class: "row" }, [el("div", { class: "k" }, "VIX"), vCell(m.vix), el("div", { class: "r" }, m.vix_note || "")]));
    (m.rows || []).forEach(function (row) {
      dl.push(el("div", { class: "row" }, [el("div", { class: "k" }, row.item), vCell(row.value, row.tone), el("div", { class: "r", html: row.read || "" })]));
    });
    if (dl.length) body.push(el("div", { class: "dl" }, dl));
    return envPanel(m.title || "🌐 全球宏观", m.risk_score, body, m.note);
  }

  function calendarPanel(cal) {
    const evs = cal.map(function (e) {
      return el("div", { class: "ev" }, [
        el("div", { class: "when" }, e.date),
        el("div", { class: "nm" + (e.flagged ? " flag" : "") }, [
          el("span", { class: "imp " + (e.impact || "med") }), e.event
        ]),
        el("div", { class: "countdown" }, e.in_days != null ? (e.in_days === 0 ? "今日" : e.in_days + "天后") : (e.note || ""))
      ]);
    });
    return el("div", { class: "panel" }, [
      el("div", { class: "p-head" }, el("div", { class: "t" }, "📅 事件前瞻")),
      el("div", { class: "p-body" }, el("div", { class: "cal" }, evs))
    ]);
  }

  /* ---- level ladder (single-name) ------------------------------------- */
  function levelLadder(L) {
    const tgtHi = L.target2 != null ? L.target2 : L.target;
    const lo = Math.min(L.stop, L.buy_low != null ? L.buy_low : L.stop);
    const hi = Math.max(tgtHi, L.price);
    const span = (hi - lo) || 1;          // guard degenerate stop==price==target (no NaN%)
    const pad = 7;
    const map = function (p) { return clampPos(pad + (p - lo) / span * (100 - 2 * pad)); };
    const buyMid = L.buy_low != null && L.buy_high != null ? (L.buy_low + L.buy_high) / 2 : L.buy_low;

    const parts = [el("div", { class: "axis" })];
    // zones
    if (L.buy_low != null) parts.push(el("div", { class: "zone loss", style: "left:" + map(L.stop) + "%;width:" + (map(L.buy_low) - map(L.stop)) + "%" }));
    if (L.buy_low != null && L.buy_high != null) parts.push(el("div", { class: "zone buy", style: "left:" + map(L.buy_low) + "%;width:" + (map(L.buy_high) - map(L.buy_low)) + "%" }));
    if (L.target != null) parts.push(el("div", { class: "zone gain", style: "left:" + map(L.buy_high != null ? L.buy_high : buyMid) + "%;width:" + (map(L.target) - map(L.buy_high != null ? L.buy_high : buyMid)) + "%" }));
    // ticks
    parts.push(el("div", { class: "tick", style: "left:" + map(L.stop) + "%" }));
    if (L.target != null) parts.push(el("div", { class: "tick", style: "left:" + map(L.target) + "%" }));
    parts.push(el("div", { class: "tick now", style: "left:" + map(L.price) + "%" }));
    // caps  (top: buy + target ; bottom: stop + now)
    function cap(side, cls, k, v) { return el("div", { class: "cap " + side + " " + cls, style: "left:" + map(v) + "%" }, [el("span", { class: "k" }, k), el("span", { class: "pv num" }, price(v))]); }
    if (buyMid != null) parts.push(el("div", { class: "cap top buy", style: "left:" + map(buyMid) + "%" }, [el("span", { class: "k" }, "买入区"), el("span", { class: "pv num" }, price(L.buy_low) + "–" + price(L.buy_high))]));
    if (L.target != null) parts.push(cap("top", "target", "目标", L.target));
    parts.push(cap("bot", "stop", "止损", L.stop));
    parts.push(cap("bot", "now", "现价", L.price));

    const foot = [
      ["现价", price(L.price)],
      ["买入区", L.buy_low != null ? price(L.buy_low) + "–" + price(L.buy_high) + (L.price ? " (" + pctStr((L.buy_low / L.price - 1) * 100) + "~" + pctStr((L.buy_high / L.price - 1) * 100) + ")" : "") : "—"],
      ["止损", price(L.stop) + (L.price ? " (" + pctStr((L.stop / L.price - 1) * 100) + ")" : "")],
      ["目标", L.target != null ? price(L.target) + (L.target2 ? "→" + price(L.target2) : "") + (L.price ? " (" + pctStr((L.target / L.price - 1) * 100) + ")" : "") : "—"]
    ];
    return el("div", { class: "ladder-card" }, [
      el("div", { class: "ladder-head" }, [
        el("div", null, [el("span", { class: "px num" }, price(L.price)), L.change_pct != null ? el("small", null, " ") : null, L.change_pct != null ? signedSpan(L.change_pct) : null]),
        el("div", { class: "rrwrap" }, [el("div", { class: "lab" }, "盈亏比 R/R"), el("div", { class: "val", style: "color:" + rrColor(L.rr) }, L.rr != null ? Number(L.rr).toFixed(2) : "—")])
      ]),
      el("div", { class: "ladder" }, parts),
      el("div", { class: "ladder-foot" }, foot.map(function (f) { return el("div", { class: "cell" }, [el("div", { class: "k" }, f[0]), el("div", { class: "v" }, f[1])]); }))
    ]);
  }

  /* ---- technical stat strip ------------------------------------------- */
  function statStrip(stats) {
    return el("div", { class: "statgrid" }, stats.map(function (s) {
      return el("div", { class: "s" }, [
        el("div", { class: "k" }, s.k),
        el("div", { class: "v", html: s.vhtml || (s.v != null ? String(s.v) : "—"), style: s.color ? "color:" + s.color : "" }),
        s.x ? el("div", { class: "x" }, s.x) : null
      ]);
    }));
  }

  /* ---- levels table (portfolio) --------------------------------------- */
  function levelsTable(rows, opts) {
    opts = opts || {};
    const cols = [
      { k: "symbol", h: "标的", cls: "l sym" },
      { k: "name", h: "名称", cls: "l name", opt: true },
      { k: "sector", h: "板块", cls: "l sector", opt: true },
      { k: "price", h: "现价", f: price },
      { k: "change_pct", h: "今日", f: function (v) { return v == null ? "" : pctSigned(v); }, colorPct: true, opt: true },
      { k: "signal", h: "信号", sig: true, cls: "" },
      { k: "regime", h: "仓位", opt: true },
      { k: "buy", h: "建议买入区", buy: true, cls: "" },
      { k: "support1", h: "支撑", supp: true, opt: true },
      { k: "stop", h: "止损", f: price },
      { k: "target", h: "目标1", f: price },
      { k: "rr", h: "盈亏比 R/R", rr: true },
      { k: "rsi", h: "RSI", f: function (v) { return v == null ? "" : v; } },
      { k: "pctb", h: "%B", f: function (v) { return v == null ? "" : Number(v).toFixed(2); }, opt: true },
      { k: "note", h: "备注", cls: "l", note: true, opt: true }
    ];
    const present = cols.filter(function (c) {
      if (!c.opt) return true;
      return rows.some(function (r) { return c.buy ? r.buy_low != null : r[c.k] != null && r[c.k] !== ""; });
    });
    const thead = el("thead", null, el("tr", null, present.map(function (c) {
      return el("th", { class: /l/.test(c.cls || "") || c.k === "symbol" || c.k === "name" || c.k === "sector" || c.k === "note" ? "l" : "" }, c.h);
    })));
    const tbody = el("tbody", null, rows.map(function (r) {
      const watch = r.signal === "watch";
      return el("tr", { class: watch ? "is-watch" : "" }, present.map(function (c) {
        if (c.sig) return el("td", { class: "" }, sigBadge(r.signal));
        if (c.rr) return el("td", null, rrBar(r.rr));
        if (c.buy) { if (r.buy_low == null) return el("td", null, "—"); var _s = vsPrice(r.buy_low, r.price), _e = vsPrice(r.buy_high, r.price); return el("td", null, [el("span", { class: "num" }, price(r.buy_low) + "–" + price(r.buy_high)), (_s && _e) ? el("small", { class: "muted" }, " (" + _s + "~" + _e + ")") : null]); }
        if (c.supp) { if (r.support1 == null) return el("td", null, "—"); var _s1 = vsPrice(r.support1, r.price); var _kids = [el("span", { class: "num" }, price(r.support1)), _s1 ? el("small", { class: "muted" }, " (" + _s1 + ")") : null]; if (r.support2 != null) { var _s2 = vsPrice(r.support2, r.price); _kids.push(el("small", { class: "muted" }, " / " + price(r.support2) + (_s2 ? " (" + _s2 + ")" : ""))); } return el("td", null, _kids); }
        if (c.note) return el("td", { class: "l note", html: (r.flag ? '<span class="flagcell">🔴 </span>' : "") + (r.note || "") });
        let v = r[c.k];
        if ((c.k === "stop" || c.k === "target") && v != null && v !== "") { var _pp = vsPrice(v, r.price); return el("td", null, [el("span", { class: "num" }, price(v)), _pp ? el("small", { class: "muted" }, " (" + _pp + ")") : null]); }
        if (c.colorPct && v != null && v !== "") {
          return el("td", { class: c.cls || "" }, el("span", { class: v > 0 ? "pos" : v < 0 ? "neg" : "" }, (c.f ? c.f(v) : v)));
        }
        return el("td", { class: c.cls || "" }, c.f ? c.f(v) : (v == null ? "—" : String(v)));
      }));
    }));
    return el("div", null, [
      el("div", { class: "tablewrap" }, el("table", { class: "grid" }, [thead, tbody])),
      opts.hint ? el("div", { class: "colhint", html: opts.hint }) : null
    ]);
  }

  /* ---- factor ranking table ------------------------------------------- */
  function factorTable(fr) {
    const heads = ["排名", "标的", "现价", "6月动量", "12月动量", "年化波动", "仓位", "综合分"];
    const thead = el("thead", null, el("tr", null, heads.map(function (h, i) {
      return el("th", { class: i === 1 ? "l" : "" }, h);
    })));
    const tbody = el("tbody", null, fr.rows.map(function (r, i) {
      return el("tr", null, [
        el("td", null, r.rank != null ? r.rank : i + 1),
        el("td", { class: "l sym" }, r.symbol + (r.leveraged ? "*" : "")),
        el("td", null, price(r.price)),
        el("td", null, el("span", { class: r.m6 > 0 ? "pos" : "neg" }, pctSigned(r.m6, 0))),
        el("td", null, el("span", { class: r.m12 > 0 ? "pos" : "neg" }, pctSigned(r.m12, 0))),
        el("td", null, r.vol != null ? r.vol + "%" : "—"),
        el("td", null, r.regime || "—"),
        el("td", null, el("span", { class: r.score > 0 ? "pos" : "neg", style: "font-weight:700" }, scoreStr(r.score)))
      ]);
    }));
    return el("div", null, [
      el("div", { class: "tablewrap" }, el("table", { class: "grid" }, [thead, tbody])),
      fr.hint ? el("div", { class: "colhint", html: fr.hint }) : null
    ]);
  }

  /* ---- sentiment ------------------------------------------------------- */
  function sentiment(s) {
    function bar(v) {
      const pos = pct01((v + 1) / 2);
      const center = 50, w = Math.abs(pos - center);
      const left = v >= 0 ? center : pos;
      const col = v > 0.05 ? "var(--pos)" : v < -0.05 ? "var(--neg)" : "var(--warn)";
      return el("div", { class: "sbar-wrap" }, [
        el("div", { class: "sbar", style: "flex:1" }, [el("span", { class: "mid" }), el("span", { class: "fill", style: "left:" + left + "%;width:" + w + "%;background:" + col })]),
        el("span", { class: "sv", style: "color:" + col }, scoreStr(v))
      ]);
    }
    const rows = (s.layers || []).map(function (l) {
      return el("div", { class: "s-row" }, [el("div", { class: "lay" }, l.layer), bar(l.score), el("div", { class: "key", html: l.key || "" })]);
    });
    if (s.composite != null) rows.push(el("div", { class: "s-row comp" }, [el("div", { class: "lay" }, "复合"), bar(s.composite), el("div", { class: "key", html: s.note || "三层加权" })]));
    return el("div", { class: "senti" }, rows);
  }

  /* ---- portfolio health ----------------------------------------------- */
  function portfolioHealth(ph) {
    const thead = el("thead", null, el("tr", null, [el("th", { class: "l" }, "指标"), el("th", null, "数值"), el("th", { class: "l" }, "解读")]));
    const tbody = el("tbody", null, ph.rows.map(function (r) {
      return el("tr", null, [
        el("td", { class: "l", style: "font-family:var(--sans)" }, r.metric),
        el("td", { html: r.value }),
        el("td", { class: "l", html: (r.flag ? '<span class="flagcell">🔴 </span>' : "") + (r.read || "") })
      ]);
    }));
    return el("div", null, [
      el("div", { class: "tablewrap" }, el("table", { class: "grid pf" }, [thead, tbody])),
      ph.conclusion ? el("div", { class: "callout-quote", html: ph.conclusion }) : null
    ]);
  }

  /* ---- generic prose / groups ----------------------------------------- */
  function proseBlock(p) {
    if (typeof p === "string") return el("div", { class: "prose", html: p });
    return el("div", { class: "prose" }, (p || []).map(function (x) { return el("p", { html: x }); }));
  }
  function conclusionBlock(p) {
    if (typeof p === "string") return el("div", { class: "prose", html: p });
    var arr = p || [];
    var structured = arr.some(function (x) { return x && typeof x === "object"; });
    if (!structured) return el("div", { class: "prose" }, arr.map(function (x) { return el("p", { html: x }); }));
    function stanceCls(s) {
      if (!s) return "mut";
      if (/利空|利淡|偏空|做空|看空|bear/i.test(s)) return "neg";
      if (/风险|警惕|谨慎|观望|延伸|watch|caution|risk/i.test(s)) return "warn";
      if (/利多|利好|偏多|做多|看多|强劲|bull/i.test(s)) return "pos";
      return "mut";
    }
    var ICON = { "基本面": "📊", "技术": "📈", "期权": "📐", "情景": "🔭", "场景": "🔭", "宏观": "🌐" };
    return el("div", { class: "concl" }, arr.map(function (x) {
      if (typeof x === "string") return el("div", { class: "ccard mut" }, el("div", { class: "cbody", html: x }));
      var st = stanceCls(x.stance);
      var head = el("div", { class: "chead" }, [
        el("span", { class: "cicon", html: x.icon || ICON[x.label] || "•" }),
        el("span", { class: "clabel", html: x.label || "" }),
        x.stance ? el("span", { class: "ctag " + st, html: x.stance }) : null
      ]);
      return el("div", { class: "ccard " + st }, [head, el("div", { class: "cbody", html: x.text || "" })]);
    }));
  }
  function groups(list) {
    function senti(s) {
      if (!s) return null;
      if (/利多|利好|偏多|做多|看多|bull/i.test(s)) return { cls: "strong", txt: s };
      if (/利空|利淡|偏空|做空|看空|bear/i.test(s)) return { cls: "weak", txt: s };
      return { cls: "neutral", txt: s };
    }
    function chip(s) { var x = senti(s); return x ? '<span class="tag ' + x.cls + '">' + x.txt + "</span>" : ""; }
    function cardLayout(g) {
      var cards = el("div", { class: "gcards" }, (g.cards || []).map(function (c) {
        var x = senti(c.sentiment);
        return el("div", { class: "gcard" + (x ? " " + x.cls : "") }, [
          el("div", { class: "gch", html: (c.label || "") + chip(c.sentiment) }),
          el("ul", { class: "gci" }, (c.items || []).map(function (it) { return el("li", { html: it }); }))
        ]);
      }));
      if (!g.foot) return cards;
      return el("div", null, [cards, el("div", { class: "gfoot", html: g.foot })]);
    }
    function volLayout(g) {
      var v = g.vol || {};
      var lo = (v.implied_low != null) ? v.implied_low : -(v.implied != null ? v.implied : 8);
      var hi = (v.implied_high != null) ? v.implied_high : (v.implied != null ? v.implied : 8);
      var act = (v.actual != null) ? v.actual : 0;
      var span = Math.max(Math.abs(act), Math.abs(lo), Math.abs(hi)) * 1.4;
      if (span < 12) span = 12;
      function pos(x) { return (x + span) / (2 * span) * 100; }
      function pct(x) { return (x > 0 ? "+" : "") + (Math.round(x * 10) / 10) + "%"; }
      var ac = act >= 0 ? "pos" : "neg";
      var pa = pos(act);
      var tShift = pa >= 70 ? "translateX(-100%)" : pa <= 30 ? "translateX(0)" : "translateX(-50%)";
      var bar = el("div", { class: "volbar" }, [
        el("div", { class: "vbtrack" }),
        el("div", { class: "vbband", style: "left:" + pos(lo).toFixed(1) + "%;width:" + (pos(hi) - pos(lo)).toFixed(1) + "%" }),
        el("div", { class: "vbzero", style: "left:" + pos(0).toFixed(1) + "%" }),
        el("div", { class: "vbtick", style: "left:" + pos(0).toFixed(1) + "%", html: "0" }),
        el("div", { class: "vbtick", style: "left:" + pos(lo).toFixed(1) + "%", html: pct(lo) }),
        el("div", { class: "vbtick", style: "left:" + pos(hi).toFixed(1) + "%", html: pct(hi) }),
        el("div", { class: "vbdot " + ac, style: "left:" + pa.toFixed(1) + "%" }),
        el("div", { class: "vbact " + ac, style: "left:" + pa.toFixed(1) + "%;transform:" + tShift, html: "实际 " + pct(act) })
      ]);
      var legend = el("div", { class: "vblegend" }, [
        el("span", { html: '<i class="sw band"></i>期权隐含区间' }),
        el("span", { html: '<i class="sw dot ' + ac + '"></i>实际跳空(冲出区间)' })
      ]);
      var kids = [bar, legend];
      if (v.iv != null) kids.push(el("div", { class: "voliv", html: "IV " + v.iv + "%" + (v.iv_pctile != null ? "(52周 " + v.iv_pctile + " 百分位)" : "") + " → 财报落地后通常回落(vol crush)" }));
      if (v.note) kids.push(el("div", { class: "gd", style: "margin-top:8px", html: v.note }));
      return el("div", { class: "volwrap" }, kids);
    }
    return el("div", { class: "groups" }, list.map(function (g) {
      var inner;
      if (g.layout === "cards" && g.cards) inner = cardLayout(g);
      else if (g.layout === "vol" && g.vol) inner = volLayout(g);
      else inner = el("div", { class: "gd", html: g.body });
      return el("div", { class: "group" }, [
        el("div", { class: "gt", html: g.title + (g.tag ? '<span class="tag ' + (g.tone || "neutral") + '">' + g.tag + "</span>" : "") }),
        inner
      ]);
    }));
  }

  /* ====================================================================== */
  /*  ORCHESTRATION                                                         */
  /* ====================================================================== */
  function block(no, title, hnote, content) {
    if (!content) return null;
    return el("section", { class: "block" }, [
      el("div", { class: "sec-head" }, [
        no ? el("div", { class: "no" }, no) : null,
        el("h2", null, title),
        hnote ? el("div", { class: "h-note" }, hnote) : null
      ]),
      content
    ]);
  }

  function methodsTable(m) {
    var syms = m.symbols || [];
    var data = m.data || {};
    function cell(sym, key) {
      var d = (data[sym] || {})[key];
      if (!d) return el("td", null, "—");
      return el("td", null, [
        el("span", { class: "mlab " + (d.tone || "neu") }, d.label || "—"),
        d.detail ? el("div", { class: "mdet" }, d.detail) : null
      ]);
    }
    var thead = el("thead", null, el("tr", null,
      [el("th", null, "判定方法"), el("th", null, "说明")].concat(
        syms.map(function (s) { return el("th", null, s.name || s.key); }))));
    var tbody = el("tbody", null, (m.rows || []).map(function (r) {
      return el("tr", { class: r.key === "old" ? "mrow-old" : "" },
        [el("td", null, el("div", { class: "mname" }, r.m)),
         el("td", null, el("div", { class: "mdesc" }, r.desc || ""))]
        .concat(syms.map(function (s) { return cell(s.key, r.key); })));
    }));
    var wrap = el("div", { class: "tablewrap" }, el("table", { class: "mtbl" }, [thead, tbody]));
    return m.note ? el("div", null, [wrap, el("p", { class: "p-note", style: "margin-top:10px", html: m.note })]) : wrap;
  }
  function tradesChart(t) {
    var P = t.price || []; var n = P.length; if (n < 2) return null;
    var W = 1000, H = 300, pl = 56, pr = 12, ptp = 10, pb = 20, logy = !!t.logy;
    var mn = Math.min.apply(null, P), mx = Math.max.apply(null, P);
    (t.overlays || []).forEach(function (o) { (o.data || []).forEach(function (v) { if (v != null && isFinite(v)) { if (v < mn) mn = v; if (v > mx) mx = v; } }); });
    var llo = Math.log(mn > 0 ? mn : 1e-9), lhi = Math.log(mx > 0 ? mx : 1);
    function X(i) { return pl + i / (n - 1) * (W - pl - pr); }
    function Y(v) { var f = logy ? (Math.log(v) - llo) / ((lhi - llo) || 1) : (v - mn) / ((mx - mn) || 1); return ptp + (1 - f) * (H - ptp - pb); }
    function fmt(v) { var u = t.unit || ""; return v >= 1e6 ? u + (v / 1e6).toFixed(1) + "M" : v >= 1e3 ? u + (v / 1e3).toFixed(0) + "k" : u + v.toFixed(0); }
    function pathOf(a) { var d = "", pen = false; for (var i = 0; i < a.length; i++) { var v = a[i]; if (v == null || !isFinite(v)) { pen = false; continue; } d += (pen ? "L" : "M") + X(i).toFixed(1) + " " + Y(v).toFixed(1) + " "; pen = true; } return d; }
    var pr2 = [];
    var labs = logy ? [mx, Math.exp((llo + lhi) / 2), mn] : [mx, (mn + mx) / 2, mn];
    [0, 0.5, 1].forEach(function (g, gi) { var y = ptp + g * (H - ptp - pb);
      pr2.push('<line x1="' + pl + '" x2="' + (W - pr) + '" y1="' + y + '" y2="' + y + '" stroke="#e7e3d8" stroke-width="1"/>');
      pr2.push('<text x="' + (pl - 6) + '" y="' + (y + 3.5) + '" text-anchor="end" font-size="11" fill="#8a8474">' + fmt(labs[gi]) + '</text>'); });
    (t.hold || []).forEach(function (sp) { var x1 = X(sp[0]), x2 = X(sp[1]); pr2.push('<rect x="' + x1.toFixed(1) + '" y="' + ptp + '" width="' + Math.max(0.5, x2 - x1).toFixed(1) + '" height="' + (H - ptp - pb) + '" fill="#b8923f" opacity="0.16"/>'); });
    (t.overlays || []).forEach(function (o) { var d = pathOf(o.data || []); if (d) pr2.push('<path d="' + d + '" fill="none" stroke="' + (o.color || "#888") + '" stroke-width="1.2"' + (o.dash ? ' stroke-dasharray="5 4"' : '') + ' opacity="0.95"/>'); });
    pr2.push('<path d="' + pathOf(P) + '" fill="none" stroke="#222" stroke-width="1.5"/>');
    var dlab = (t.dates && (t.buys || []).length + (t.sells || []).length <= 36);
    function dstr(i) { var ss = t.dates && t.dates[i] ? String(t.dates[i]) : ""; return ss.length >= 10 ? ss.slice(5) : ss; }
    (t.buys || []).forEach(function (i) { var x = X(i), y = Y(P[i]); pr2.push('<polygon points="' + x + ',' + (y - 10) + ' ' + (x - 6.5) + ',' + (y + 3) + ' ' + (x + 6.5) + ',' + (y + 3) + '" fill="#c0392b" stroke="#fff" stroke-width="0.8"/>'); if (dlab) pr2.push('<text x="' + x.toFixed(1) + '" y="' + (y + 16) + '" text-anchor="middle" font-size="9.5" fill="#c0392b">' + dstr(i) + '</text>'); });
    (t.sells || []).forEach(function (i) { var x = X(i), y = Y(P[i]); pr2.push('<polygon points="' + x + ',' + (y + 10) + ' ' + (x - 6.5) + ',' + (y - 3) + ' ' + (x + 6.5) + ',' + (y - 3) + '" fill="#147a43" stroke="#fff" stroke-width="0.8"/>'); if (dlab) pr2.push('<text x="' + x.toFixed(1) + '" y="' + (y - 12) + '" text-anchor="middle" font-size="9.5" fill="#147a43">' + dstr(i) + '</text>'); });
    pr2.push('<circle cx="' + X(n - 1) + '" cy="' + Y(P[n - 1]) + '" r="4" fill="#111"/>');
    if (t.date_start) pr2.push('<text x="' + pl + '" y="' + (H - 5) + '" font-size="11" fill="#8a8474">' + t.date_start + '</text>');
    if (t.date_end) pr2.push('<text x="' + (W - pr) + '" y="' + (H - 5) + '" text-anchor="end" font-size="11" fill="#8a8474">' + t.date_end + '</text>');
    var svg = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ' + W + ' ' + H + '" preserveAspectRatio="xMidYMid meet" style="width:100%;height:auto;display:block">' + pr2.join("") + '</svg>';
    var leg = [el("span", { html: '<b style="color:#c0392b">▲</b> 买入' }), el("span", { html: '<b style="color:#147a43">▼</b> 卖出' }), el("span", { html: '<i class="hsw"></i> 持仓期' })];
    (t.overlays || []).forEach(function (o) { leg.push(el("span", { html: '<i style="display:inline-block;width:15px;height:0;border-top:2px ' + (o.dash ? "dashed" : "solid") + ' ' + (o.color || "#888") + ';vertical-align:middle;margin-right:5px"></i>' + (o.label || "") })); });
    if (t.now_label) leg.push(el("span", { class: "tc-now", html: t.now_label }));
    var holder = el("div", null);
    try { var _pd = new DOMParser().parseFromString(svg, "image/svg+xml"); var _r = _pd.documentElement; if (_r && String(_r.nodeName).toLowerCase() === "svg") holder.appendChild(document.importNode(_r, true)); else holder.innerHTML = svg; } catch (e) { holder.innerHTML = svg; }
    return el("div", { class: "chart-card" }, [el("div", { class: "tc-legend" }, leg), holder]);
  }
  function researchSection(r) {
    var kids = (r.items || []).map(function (it) {
      var w = it.winner || {};
      var head = el("div", { class: "rs-head" }, [
        el("span", { class: "rs-name" }, it.name || it.symbol),
        el("span", { class: "rs-win" }, "冠军 " + (w.strategy || "") + (w.params ? (" · " + w.params) : "")),
        w.signal ? el("span", { class: "mlab " + (w.signal_tone || "neu") }, w.signal) : null
      ]);
      function fact(k, v) { return el("div", null, [el("span", { class: "rk" }, k), el("span", { class: "rv" }, v)]); }
      var facts = el("div", { class: "rs-facts" }, [
        fact("OOS 夏普", w.oos_sharpe != null ? String(w.oos_sharpe) : "—"),
        fact("OOS 收益", w.oos_return || "—"),
        fact("当前信号", w.signal || "—"),
        fact("离场线", w.exit || "—")
      ]);
      var tg = w.triggers || null;
      var trig = tg ? el("div", { class: "rs-trig" }, [
        el("span", { class: "rs-trig-h" }, "具体买卖点"),
        el("span", { html: "现在 · <b>" + (tg.action || w.signal || "—") + "</b>" }),
        tg.sell ? el("span", { html: "<b style='color:#147a43'>▼</b> 卖出/离场:<b>" + tg.sell + "</b>" }) : null,
        tg.buy ? el("span", { html: "<b style='color:#c0392b'>▲</b> 买入/回补:<b>" + tg.buy + "</b>" }) : null
      ]) : null;
      var sel = it.selection_text ? el("div", { class: "rs-sel", html: "<b>搜索过程(bandit):</b> " + it.selection_text }) : null;
      var lb = null;
      if (it.leaderboard && it.leaderboard.length) {
        var body = el("tbody", null, it.leaderboard.map(function (x) {
          return el("tr", { class: x.win ? "rmwin" : "" }, [
            el("td", null, String(x.rank)), el("td", null, x.strategy),
            el("td", { class: "rmp" }, x.params || ""),
            el("td", { style: "text-align:right" }, String(x.oos_sharpe)),
            el("td", { style: "text-align:right" }, x.oos_return || ""),
            el("td", null, x.signal || ""),
            el("td", { class: "rmbuy" }, x.buy || ""),
            el("td", { class: "rmsell" }, x.sell || "")
          ]);
        }));
        var thead = el("thead", null, el("tr", null, [el("th", null, "#"), el("th", null, "策略族"),
          el("th", null, "参数"), el("th", { style: "text-align:right" }, "OOS夏普"), el("th", { style: "text-align:right" }, "OOS收益"), el("th", null, "当前"), el("th", null, "买入触发"), el("th", null, "卖出触发")]));
        lb = el("div", null, [el("div", { class: "rs-sub" }, "① 策略选择对比 · 所有模拟策略排行(样本外 walk-forward · 含买卖触发价)"),
          el("div", { class: "tablewrap" }, el("table", { class: "rmtbl" }, [thead, body]))]);
      }
      var chart = null;
      if (it.trades) {
        var statsrow = (it.stats || []).length ? el("div", { class: "chart-stats" }, it.stats.map(function (sx) { return el("div", { class: "cs", html: sx.k + "<b>" + sx.v + "</b>" }); })) : null;
        var cap = el("div", { class: "rs-cap", html: "图说:实线=股价" + (it.trades.logy ? "(对数轴)" : "") + ";<b style='color:#c0392b'>▲买入</b> <b style='color:#147a43'>▼卖出</b> 金色阴影=持仓期。下方数字 <b>策略收益</b>=这套规则的总收益,<b>买入持有</b>=一直拿着不动的总收益(常更高);本策略赢在<b>回撤更小/夏普更高</b>,不是赢在绝对收益。" });
        chart = el("div", null, [el("div", { class: "rs-sub" }, "② 买卖点与持仓(冠军策略在价格上的进出)"), tradesChart(it.trades), statsrow, cap]);
      }
      return el("div", { class: "rs-item" }, [head, facts, trig, sel, lb, chart].filter(Boolean));
    });
    if (r.glossary && r.glossary.length) {
      kids.unshift(el("div", { class: "rs-gloss" }, [
        el("div", { class: "rs-sub" }, "📖 策略方法说明 · 本次测试涵盖的策略族"),
        el("ul", { style: "margin:6px 0 16px;padding-left:18px;font-size:12.5px;line-height:1.75;color:var(--ink-soft)" },
          r.glossary.map(function (g) {
            return el("li", null, [el("b", null, (g.name || g.family) + "："), (g.intro || "") + (g.edge ? " " + g.edge : "")]);
          }))
      ]));
    }
    if (r.note) kids.push(el("p", { class: "p-note", style: "margin-top:4px", html: r.note }));
    return el("div", null, kids);
  }
  /* ---- B2 稳健性体检:CPCV 分布 + DSR/PBO/SPA 计量表 -------------------- */
  function pctS(x) { return (x == null || isNaN(x)) ? "—" : (x > 0 ? "+" : "") + (100 * x).toFixed(1) + "%"; }
  function _svgHolder(svg) {
    var holder = el("div", null);
    try { var d = new DOMParser().parseFromString(svg, "image/svg+xml"); var rr = d.documentElement; if (rr && String(rr.nodeName).toLowerCase() === "svg") holder.appendChild(document.importNode(rr, true)); else holder.innerHTML = svg; } catch (e) { holder.innerHTML = svg; }
    return holder;
  }
  function robustnessSection(rb) {
    var cp = rb.cpcv || {}, spa = rb.spa || {};
    function fact(k, v, tone, note) {
      return el("div", null, [el("span", { class: "rk" }, k),
        el("span", { class: "rv", style: tone ? ("color:var(--" + tone + ")") : "" }, v),
        note ? el("span", { class: "rb-note" }, note) : null]);
    }
    var dsr = rb.deflated_sharpe, pbo = rb.pbo, sp = spa.spa_p, fp = cp.frac_positive;
    var facts = el("div", { class: "rs-facts" }, [
      fact("Deflated Sharpe", dsr != null ? String(dsr) : "—", dsr == null ? "" : (dsr >= 0.9 ? "pos" : (dsr < 0.5 ? "neg" : "")), "≥0.95 = 扣掉多重检验后仍显著"),
      fact("PBO 过拟合概率", pbo != null ? String(pbo) : "—", pbo == null ? "" : (pbo < 0.5 ? "pos" : "neg"), "<0.5 = 选择过程稳健"),
      fact("SPA p 值", sp != null ? String(sp) : "—", sp == null ? "" : (sp < 0.05 ? "pos" : "neg"), "<0.05 = 冠军非数据窥探"),
      fact("CPCV 为正占比", fp != null ? Math.round(fp * 100) + "%" : "—", fp == null ? "" : (fp >= 0.7 ? "pos" : (fp < 0.5 ? "neg" : "")), "出样本夏普>0 的路径比例")
    ]);
    var strip = null, paths = cp.sharpe_paths || [];
    if (paths.length) {
      var W = 700, H = 92, pl = 30, pr = 20;
      var ext = paths.concat([0]); if (cp.q05 != null) ext.push(cp.q05); if (cp.q95 != null) ext.push(cp.q95);
      var lo = Math.min.apply(null, ext), hi = Math.max.apply(null, ext);
      if (hi - lo < 1e-9) hi = lo + 1;
      var pad = 0.08 * (hi - lo); lo -= pad; hi += pad;
      function X(v) { return pl + (v - lo) / (hi - lo) * (W - pl - pr); }
      var s = [];
      if (cp.q05 != null && cp.q95 != null) s.push('<rect x="' + X(cp.q05).toFixed(1) + '" y="24" width="' + Math.max(1, X(cp.q95) - X(cp.q05)).toFixed(1) + '" height="28" fill="#ece5d6" rx="4"/>');
      s.push('<line x1="' + X(0).toFixed(1) + '" y1="14" x2="' + X(0).toFixed(1) + '" y2="62" stroke="#b9b2a0" stroke-dasharray="4 3"/>');
      s.push('<text x="' + X(0).toFixed(1) + '" y="76" text-anchor="middle" font-size="10" fill="#8a8474">0</text>');
      for (var i = 0; i < paths.length; i++) {
        var p = paths[i];
        s.push('<circle cx="' + X(p).toFixed(1) + '" cy="' + (38 + ((i * 7919) % 17) - 8) + '" r="3" fill="' + (p >= 0 ? "#c0392b" : "#147a43") + '" fill-opacity="0.5"/>');
      }
      if (cp.median != null) {
        s.push('<line x1="' + X(cp.median).toFixed(1) + '" y1="18" x2="' + X(cp.median).toFixed(1) + '" y2="58" stroke="#1a1a1a" stroke-width="2"/>');
        s.push('<text x="' + X(cp.median).toFixed(1) + '" y="12" text-anchor="middle" font-size="11" font-weight="700" fill="#1a1a1a">中位 ' + cp.median + '</text>');
      }
      if (cp.q05 != null) s.push('<text x="' + X(cp.q05).toFixed(1) + '" y="88" text-anchor="middle" font-size="10" fill="#8a8474">5%: ' + cp.q05 + '</text>');
      if (cp.q95 != null) s.push('<text x="' + X(cp.q95).toFixed(1) + '" y="88" text-anchor="middle" font-size="10" fill="#8a8474">95%: ' + cp.q95 + '</text>');
      var svg = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ' + W + ' ' + H + '" preserveAspectRatio="xMidYMid meet" style="width:100%;height:auto;display:block">' + s.join("") + '</svg>';
      strip = el("div", { class: "chart-card" }, [
        el("div", { class: "rs-sub" }, "CPCV 出样本夏普分布(" + (cp.n_paths || paths.length) + " 条组合净化路径 · 每点一条)"), _svgHolder(svg)]);
    }
    var msg = [];
    if (sp != null) msg.push(sp < 0.05 ? "SPA <b>通过</b>——冠军不是纯搜索幸运" : "SPA <b>未通过</b>——冠军可能只是 " + (rb.n_trials || "多") + " 个配置里最幸运的");
    if (pbo != null) msg.push("PBO " + pbo + (pbo < 0.5 ? "(选择稳健)" : "(选择在过拟合)"));
    if (fp != null) msg.push("CPCV 路径 " + Math.round(fp * 100) + "% 为正" + (fp >= 0.7 ? "——edge 不依赖单一幸运窗口" : (fp < 0.5 ? "——半数路径亏损,edge 存疑" : "")));
    var vd = el("p", { class: "p-note", html: (rb.winner ? "检验对象:<b>" + rb.winner + "</b>(从 " + (rb.n_trials || "?") + " 个配置中选出)· " : "") + msg.join(";") });
    return el("div", null, [facts, strip, vd].filter(Boolean));
  }
  /* ---- B3 下跌切片:基准最惨时段的策略对照 ------------------------------ */
  function downturnSection(dt) {
    var rows = dt.rows || [];
    function cell(v) { return el("td", { style: "text-align:right;font-weight:600;color:" + (v > 0 ? "var(--pos)" : (v < 0 ? "var(--neg)" : "inherit")) }, pctS(v)); }
    var body = el("tbody", null, rows.map(function (r) {
      return el("tr", null, [el("td", null, r.label || ""), el("td", { class: "rmp" }, r.period || ""),
        cell(r.strategy), cell(r.benchmark), cell(r.excess)]);
    }));
    var thead = el("thead", null, el("tr", null, [el("th", null, "切片"), el("th", null, "区间"),
      el("th", { style: "text-align:right" }, "策略"), el("th", { style: "text-align:right" }, "买入持有"),
      el("th", { style: "text-align:right" }, "超额")]));
    var note = el("p", { class: "p-note", html: dt.note || "只看全样本会把顺风段的运气当成 edge(StockBench:下跌段全员失效)。这里强制展示基准最惨的几段——看策略在下跌里是保护了资金,还是跟着一起摔。" });
    return el("div", null, [el("div", { class: "tablewrap" }, el("table", { class: "rmtbl" }, [thead, body])), note]);
  }
  /* ---- B4 成本敏感度:0/10/30bps 下同一策略 ----------------------------- */
  function costCurveSection(cc) {
    var rows = cc.rows || [], maxAbs = 0;
    rows.forEach(function (r) { if (r.sharpe != null) maxAbs = Math.max(maxAbs, Math.abs(r.sharpe)); });
    var body = el("tbody", null, rows.map(function (r) {
      var w = (maxAbs > 0 && r.sharpe != null) ? Math.round(Math.abs(r.sharpe) / maxAbs * 100) : 0;
      var bar = el("td", null, el("div", { class: "ccbar" }, el("i", { style: "width:" + w + "%;background:" + (r.sharpe >= 0 ? "var(--pos)" : "var(--neg)") })));
      return el("tr", null, [el("td", null, (r.bps != null ? r.bps : "?") + " bps"),
        el("td", { style: "text-align:right" }, pctS(r.total_return)),
        el("td", { style: "text-align:right;font-weight:600" }, r.sharpe != null ? Number(r.sharpe).toFixed(2) : "—"),
        el("td", { style: "text-align:right" }, pctS(r.max_drawdown)), bar]);
    }));
    var thead = el("thead", null, el("tr", null, [el("th", null, "单边成本"), el("th", { style: "text-align:right" }, "总收益"),
      el("th", { style: "text-align:right" }, "夏普"), el("th", { style: "text-align:right" }, "最大回撤"), el("th", null, "夏普对比")]));
    var note = el("p", { class: "p-note", html: cc.note || "同一策略在三档单边成本下重跑:30bps 档夏普仍为正 ⇒ edge 扛得住真实摩擦;一路衰减到负 ⇒ 纸面收益只是把成本假设调低的产物。" });
    return el("div", null, [el("div", { class: "tablewrap" }, el("table", { class: "rmtbl" }, [thead, body])), note]);
  }
  function render(data, mount) {
    mount.innerHTML = "";
    // Optional data.symbol_order: stable-sort symbol-bearing sections to ONE canonical
    // company order so every section lines up. Rows whose symbol isn't listed (e.g. a
    // market-wide "宏观" alert) sort first, keeping their relative order. factor_rank is
    // intentionally left as a ranking and never reordered.
    if (data.symbol_order && data.symbol_order.length) {
      var _ord = data.symbol_order;
      var _key = function (r) { var i = _ord.indexOf(r && r.symbol); return i < 0 ? -1 : i; };
      var _stable = function (arr) {
        return arr.map(function (r, i) { return [r, i]; })
          .sort(function (a, b) { return (_key(a[0]) - _key(b[0])) || (a[1] - b[1]); })
          .map(function (x) { return x[0]; });
      };
      if (Array.isArray(data.levels)) data.levels = _stable(data.levels);
      if (Array.isArray(data.alerts)) data.alerts = _stable(data.alerts);
    }
    const frag = document.createDocumentFragment();
    frag.appendChild(masthead(data.meta));
    const body = el("div", { class: "body" });

    // verdict is unnumbered, sits at top
    // Objective stance score: regime (sector) is primary; macro only nudges. No regime score
    // -> envScore stays null and verdict falls back to keyword lean. Set verdict.score to override.
    var _rg = data.regime && typeof data.regime.score === "number" ? data.regime.score : null;
    var _mc = data.macro && typeof data.macro.risk_score === "number" ? data.macro.risk_score : null;
    var _envScore = (_rg !== null) ? (_mc !== null ? 0.75 * _rg + 0.25 * _mc : _rg) : null;
    if (data.verdict) body.appendChild(el("section", { class: "block", style: "border-top:none;padding-top:22px" }, verdict(data.verdict, _envScore)));
    // 综合结论紧跟综合立场,作为顶部"结论"汇总(不编号);其余编号区块顺延
    if (data.conclusion) body.appendChild(block(null, data.conclusion_title || "综合结论", null, conclusionBlock(data.conclusion)));

    let no = 0;
    function add(title, hnote, content) { if (content) { no++; body.appendChild(block(no, title, hnote, content)); } }

    if (data.alerts) add("🔴 今日重点关注", data.alerts.length + " 项", alerts(data.alerts));

    // environment composite
    if (data.regime || data.macro || data.calendar) {
      const panels = [];
      if (data.regime) panels.push(regimePanel(data.regime));
      if (data.macro) panels.push(macroPanel(data.macro));
      const grid = el("div", { class: panels.length === 1 ? "env-grid one" : "env-grid" }, panels);
      const wrap = el("div", null, [grid, data.calendar ? el("div", { style: "margin-top:18px" }, calendarPanel(data.calendar)) : null]);
      add("市场环境 · 宏观 · 事件", null, wrap);
    }

    if (data.factor_rank) add(data.factor_rank.title || "多因子排序", data.factor_rank.note_head || null, factorTable(data.factor_rank));
    if (data.groups) add(data.groups_title || "分组解读", null, groups(data.groups));

    // single-name hero
    if (data.technical) {
      const parts = [];
      if (data.technical.level) parts.push(levelLadder(data.technical.level));
      if (data.technical.stats) parts.push(el("div", { style: "margin-top:18px" }, statStrip(data.technical.stats)));
      if (data.technical.note) parts.push(el("div", { class: "prose", style: "margin-top:8px" }, proseBlock(data.technical.note)));
      add(data.technical.title || "📈 技术与买卖点", data.technical.signal_note || null, el("div", null, parts));
    }

    if (data.levels) add(data.levels_title || "🎯 建议买卖点", null, levelsTable(data.levels, { hint: data.levels_hint }));
    if (data.sentiment) add(data.sentiment.title || "🗞 三层时效情绪", data.sentiment.composite != null ? "复合 " + scoreStr(data.sentiment.composite) : null, sentiment(data.sentiment));
    if (data.portfolio_health) add(data.portfolio_health.title || "🧩 组合体检", null, portfolioHealth(data.portfolio_health));
    if (data.holdings) add("你的持仓", null, proseBlock(data.holdings));
    if (data.methods) add(data.methods.title || "信号多法对照", null, methodsTable(data.methods));
    if (data.research) add(data.research.title || "自动研究详情", null, researchSection(data.research));
    // build_research 的产物把 robustness/downturn/cost_curve 一并带回 —— 自动上提为独立区块,
    // 调用方无需手工搬运;显式给了顶层 key 时以顶层为准。
    if (data.research && !data.robustness && data.research.robustness) data.robustness = data.research.robustness;
    if (data.research && !data.downturn && data.research.downturn) data.downturn = data.research.downturn;
    if (data.research && !data.cost_curve && data.research.cost_curve) data.cost_curve = data.research.cost_curve;
    if (data.robustness) add(data.robustness.title || "🧪 稳健性体检 · 冠军是真 edge 吗", null, robustnessSection(data.robustness));
    if (data.downturn) add(data.downturn.title || "📉 下跌切片 · 基准最惨时段对照", null, downturnSection(data.downturn));
    if (data.cost_curve) add(data.cost_curve.title || "💸 成本敏感度 · edge 扛得住摩擦吗", null, costCurveSection(data.cost_curve));

    frag.appendChild(body);

    // footer
    const foot = el("div", { class: "footer" }, [
      data.disclaimer ? el("div", { class: "disc", html: data.disclaimer }) : null,
      data.sources && data.sources.length ? el("div", { class: "sources", html: "Sources · " + data.sources.map(function (s) { return s.url ? '<a href="' + s.url + '" target="_blank" rel="noopener">' + s.label + "</a>" : s.label; }).join(" · ") }) : null,
      el("div", { class: "sign" }, [
        el("span", { html: '本页由 <span class="mk">' + ((data.meta && data.meta.generated_by) || "alpha-forge") + '</span> 技能生成 · 机械量化研究，非投资建议' }),
        el("span", null, (data.meta && data.meta.date) || "")
      ])
    ]);
    frag.appendChild(foot);

    mount.appendChild(frag);
    document.title = (data.meta && data.meta.title) || "量化分析报告";
  }

  global.QuantReport = { render: render };
})(window);

