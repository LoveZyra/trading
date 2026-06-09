/* ============================================================================
   alpha-forge · 报告渲染器
   Reads a single report object (schema in SCHEMA.md) and builds the DOM.
   Every section is optional — present a key, it renders; omit it, it's skipped.
   ========================================================================== */
(function (global) {
  "use strict";

  /* ---- tiny DOM helper ------------------------------------------------- */
  function el(tag, attrs, children) {
    const n = document.createElement(tag);
    if (attrs) for (const k in attrs) {
      if (k === "class") n.className = attrs[k];
      else if (k === "html") n.innerHTML = attrs[k];
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
    const m = SIG[s] || ["watch", s || "—"];
    return el("span", { class: "sig " + m[0] }, textOverride || m[1]);
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

  function verdict(v) {
    if (!v) return null;
    const dir = v.stance && /多|涨|强|看多|偏多|bull/i.test(v.stance) ? "up"
      : v.stance && /空|跌|弱|看空|偏空|bear/i.test(v.stance) ? "down" : "flat";
    const arrow = dir === "up" ? "▲" : dir === "down" ? "▼" : "◆";
    return el("div", { class: "verdict " + dir }, [
      el("div", { class: "stance" }, [
        el("div", { class: "lab" }, "综合立场"),
        el("div", { class: "arrow" }, arrow),
        el("div", { class: "val" }, v.stance || "中性")
      ]),
      el("div", { class: "body-v" }, [
        v.action ? el("p", { class: "action", html: v.action }) : null,
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

  function regimePanel(r) {
    const rows = (r.rows || []).map(function (row) {
      return el("div", { class: "row" }, [
        el("div", { class: "k" }, row.item),
        el("div", { class: "v", html: row.value },),
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
    if (m.vix != null) dl.push(el("div", { class: "row" }, [el("div", { class: "k" }, "VIX"), el("div", { class: "v" }, String(m.vix)), el("div", { class: "r" }, m.vix_note || "")]));
    (m.rows || []).forEach(function (row) {
      dl.push(el("div", { class: "row" }, [el("div", { class: "k" }, row.item), el("div", { class: "v", html: row.value }), el("div", { class: "r", html: row.read || "" })]));
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
    const pad = 7;
    const map = function (p) { return clampPos(pad + (p - lo) / (hi - lo) * (100 - 2 * pad)); };
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
      ["买入区", L.buy_low != null ? price(L.buy_low) + "–" + price(L.buy_high) : "—"],
      ["止损", price(L.stop)],
      ["目标", L.target != null ? price(L.target) + (L.target2 ? "→" + price(L.target2) : "") : "—"]
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
        if (c.buy) return el("td", null, r.buy_low != null ? price(r.buy_low) + "–" + price(r.buy_high) : "—");
        if (c.note) return el("td", { class: "l note", html: (r.flag ? '<span class="flagcell">🔴 </span>' : "") + (r.note || "") });
        let v = r[c.k];
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
    const heads = ["排名", "标的", "现价", "6月动量", "12月动量", "年化波动", "信号", "仓位", "综合分"];
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
        el("td", null, sigBadge(r.signal)),
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

  /* ---- equity / benchmark chart --------------------------------------- */
  function equityChart(bt) {
    const W = 1000, H = 300, pl = 6, pr = 6, pt = 14, pb = 24;
    const eq = bt.equity || {};
    const A = eq.strategy || [], B = eq.benchmark || [];
    const n = Math.max(A.length, B.length);
    if (n < 2) return null;
    let mn = Infinity, mx = -Infinity;
    [A, B].forEach(function (s) { s.forEach(function (v) { if (v < mn) mn = v; if (v > mx) mx = v; }); });
    const px = function (i) { return pl + i / (n - 1) * (W - pl - pr); };
    const py = function (v) { return pt + (1 - (v - mn) / (mx - mn || 1)) * (H - pt - pb); };
    const path = function (s) { return s.map(function (v, i) { return (i ? "L" : "M") + px(i).toFixed(1) + " " + py(v).toFixed(1); }).join(" "); };
    const grid = [];
    for (let g = 0; g <= 4; g++) { const y = pt + g / 4 * (H - pt - pb); grid.push('<line x1="' + pl + '" x2="' + (W - pr) + '" y1="' + y + '" y2="' + y + '" stroke="#e7e3d8" stroke-width="1"/>'); }
    const svg = '<svg viewBox="0 0 ' + W + ' ' + H + '" preserveAspectRatio="none" style="width:100%;height:230px;display:block">' +
      grid.join("") +
      (B.length ? '<path d="' + path(B) + '" fill="none" stroke="#9aa0aa" stroke-width="2" stroke-dasharray="5 4"/>' : "") +
      '<path d="' + path(A) + '" fill="none" stroke="#1b3a5b" stroke-width="2.5"/>' +
      '</svg>';
    const stats = (bt.stats || []).map(function (s) { return el("div", { class: "cs", html: s.k + "<b>" + s.v + "</b>" }); });
    return el("div", { class: "chart-card" }, [
      el("div", { class: "chart-head" }, [
        el("div", { class: "legend" }, [
          el("span", null, [el("i", { style: "background:#1b3a5b" }), bt.strategy_label || "策略"]),
          el("span", null, [el("i", { style: "background:#9aa0aa" }), bt.benchmark_label || "买入持有"])
        ])
      ]),
      el("div", { html: svg }),
      stats.length ? el("div", { class: "chart-stats" }, stats) : null
    ]);
  }

  /* ---- generic prose / groups ----------------------------------------- */
  function proseBlock(p) {
    if (typeof p === "string") return el("div", { class: "prose", html: p });
    return el("div", { class: "prose" }, (p || []).map(function (x) { return el("p", { html: x }); }));
  }
  function groups(list) {
    return el("div", { class: "groups" }, list.map(function (g) {
      return el("div", { class: "group" }, [
        el("div", { class: "gt", html: g.title + (g.tag ? '<span class="tag ' + (g.tone || "neutral") + '">' + g.tag + "</span>" : "") }),
        el("div", { class: "gd", html: g.body })
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

  function render(data, mount) {
    mount.innerHTML = "";
    const frag = document.createDocumentFragment();
    frag.appendChild(masthead(data.meta));
    const body = el("div", { class: "body" });

    // verdict is unnumbered, sits at top
    if (data.verdict) body.appendChild(el("section", { class: "block", style: "border-top:none;padding-top:22px" }, verdict(data.verdict)));

    let no = 0;
    function add(title, hnote, content) { if (content) { no++; body.appendChild(block(no, title, hnote, content)); } }

    if (data.alerts) add("🔴 今日重点关注", data.alerts.length + " 项", alerts(data.alerts));

    // environment composite
    if (data.regime || data.macro || data.calendar) {
      const panels = [];
      if (data.regime) panels.push(regimePanel(data.regime));
      if (data.macro) panels.push(macroPanel(data.macro));
      const grid = el("div", { class: "env-grid" }, panels);
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
    if (data.backtest) { const c = equityChart(data.backtest); if (c) add(data.backtest.title || "策略回测", data.backtest.head_note || null, c); }
    if (data.sentiment) add(data.sentiment.title || "🗞 三层时效情绪", data.sentiment.composite != null ? "复合 " + scoreStr(data.sentiment.composite) : null, sentiment(data.sentiment));
    if (data.portfolio_health) add(data.portfolio_health.title || "🧩 组合体检", null, portfolioHealth(data.portfolio_health));
    if (data.holdings) add("你的持仓", null, proseBlock(data.holdings));
    if (data.conclusion) add("综合结论", null, proseBlock(data.conclusion));

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
