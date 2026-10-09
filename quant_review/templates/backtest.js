(function () {
  const STRATEGY_LABELS = {
    mainline: "主线复盘策略",
    first_limit: "涨停板策略",
    first_wave: "第一波启动策略",
    canslim_tao: "CANSLIM+RPS策略",
    canslim_early: "CANSLIM早期防钝化策略",
    value_fcf: "低估值现金流策略",
    dividend_low_vol: "红利低波防守策略"
  };

  const TRADE_COLS = [
    ["entry_date", "买入日"], ["exit_date", "卖出日"], ["code", "代码"], ["name", "名称"], ["stock_type", "股票类型"],
    ["signal_score", "信号分"], ["signal_rank", "排名"], ["entry_price", "买入价"], ["exit_price", "卖出价"],
    ["return_pct", "收益%"], ["pnl", "盈亏"], ["hold_days", "持有天"], ["exit_reason", "卖出原因"], ["signal_date", "信号日"]
  ];
  const CURVE_COLS = [["date", "日期"], ["equity", "权益"], ["daily_return_pct", "日收益%"], ["drawdown_pct", "回撤%"]];

  function initBacktestPanel(options) {
    const strategy = options.strategy;
    const label = options.label || STRATEGY_LABELS[strategy] || strategy;
    const id = `backtest-${strategy}`;
    const mount = () => {
      if (document.getElementById(id)) return;
      const main = document.querySelector("main");
      if (!main) return;
      const firstToolbar = main.querySelector(".toolbar");
      const launcher = document.createElement("button");
      launcher.type = "button";
      launcher.className = "secondary backtest-launch";
      launcher.textContent = "策略回测";
      if (firstToolbar) {
        firstToolbar.insertBefore(launcher, firstToolbar.firstChild);
      } else {
        const launchBar = document.createElement("div");
        launchBar.className = "toolbar backtest-launchbar";
        launchBar.appendChild(launcher);
        const status = main.querySelector("#status") || main.querySelector(".status");
        const controls = main.querySelector("#pageControls") || main.querySelector(".controls");
        const anchor = status || controls;
        if (anchor && anchor.parentNode === main) {
          anchor.insertAdjacentElement("afterend", launchBar);
        } else {
          main.insertBefore(launchBar, main.firstChild);
        }
      }

      const panel = document.createElement("div");
      panel.id = id;
      panel.className = "modal backtest-modal";
      panel.setAttribute("aria-hidden", "true");
      panel.innerHTML = `
        <div class="modal-card backtest-modal-card" role="dialog" aria-modal="true" aria-labelledby="${id}-title">
          <div class="modal-head">
            <h2 id="${id}-title">策略回测</h2>
            <button class="icon-btn" data-bt="close" title="关闭" aria-label="关闭">x</button>
          </div>
          <div class="modal-body backtest-modal-body">
            ${template(label)}
          </div>
        </div>
      `;
      document.body.appendChild(panel);

      const start = panel.querySelector("[data-bt='start']");
      const end = panel.querySelector("[data-bt='end']");
      start.value = dateText(monthsAgo(6));
      end.value = dateText(new Date());

      launcher.addEventListener("click", () => openBacktest(panel));
      panel.querySelector("[data-bt='close']").addEventListener("click", () => closeBacktest(panel));
      panel.addEventListener("click", event => {
        if (event.target === panel) closeBacktest(panel);
      });
      document.addEventListener("keydown", event => {
        if (event.key === "Escape" && panel.classList.contains("open")) closeBacktest(panel);
      });
      panel.querySelector("[data-bt='run']").addEventListener("click", () => runBacktest(panel, strategy, options.getPayload));
      panel.querySelectorAll("[data-bt='exportTrades']").forEach(button => {
        button.addEventListener("click", () => exportRows(panel._btTrades || [], TRADE_COLS, `${strategy}_backtest_trades`));
      });
      panel.querySelectorAll("[data-bt='exportCurve']").forEach(button => {
        button.addEventListener("click", () => exportRows(panel._btCurve || [], CURVE_COLS, `${strategy}_backtest_curve`));
      });
    };
    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", mount);
    } else {
      mount();
    }
  }

  function template(label) {
    return `
      <div class="controls backtest-controls">
        <div><label>回测策略</label><input value="${btEscapeHtml(label)}" disabled /></div>
        <div><label>开始日期</label><input data-bt="start" /></div>
        <div><label>结束日期</label><input data-bt="end" /></div>
        <div><label>初始资金</label><input data-bt="cash" type="number" value="1000000" /></div>
        <div><label>最多持仓</label><input data-bt="maxPositions" type="number" value="5" /></div>
        <div><label>每日最多买入</label><input data-bt="maxEntries" type="number" value="2" /></div>
        <div><label>信号取前N</label><input data-bt="signalTopN" type="number" value="20" /></div>
        <div><label>最长持有天</label><input data-bt="maxHold" type="number" value="10" /></div>
        <div><label>止盈%</label><input data-bt="takeProfit" type="number" step="0.5" value="15" /></div>
        <div><label>止损%</label><input data-bt="stopLoss" type="number" step="0.5" value="8" /></div>
        <div><label>均线卖出</label><select data-bt="exitMa"><option value="ma5">MA5</option><option value="ma10" selected>MA10</option><option value="ma20">MA20</option><option value="none">不启用</option></select></div>
        <div><label>滑点bp</label><input data-bt="slippage" type="number" step="1" value="5" /></div>
        <div><label>买卖佣金率</label><input data-bt="fee" type="number" step="0.0001" value="0.0003" /></div>
        <div><label>卖出税费率</label><input data-bt="sellTax" type="number" step="0.0001" value="0.0005" /></div>
        <label class="check"><input data-bt="skipLimit" type="checkbox" checked /> 跳过一字涨停开盘</label>
        <button data-bt="run">开始回测</button>
        <button class="secondary" data-bt="exportTrades">导出交易</button>
        <button class="ghost" data-bt="exportCurve">导出曲线</button>
      </div>
      <div class="status" data-bt="status">等待回测。</div>
      <div class="cards" data-bt="summary"></div>
      <div class="toolbar backtest-toolbar">
        <button class="ghost" data-bt="exportTrades">导出交易</button>
        <button class="ghost" data-bt="exportCurve">导出权益曲线</button>
      </div>
      <div class="table-wrap backtest-table-wrap"><table data-bt="trades" class="backtest-table"></table></div>
      <div class="table-wrap backtest-table-wrap compact"><table data-bt="curve" class="backtest-table"></table></div>
    `;
  }

  function openBacktest(panel) {
    panel.classList.add("open");
    panel.setAttribute("aria-hidden", "false");
  }

  function closeBacktest(panel) {
    panel.classList.remove("open");
    panel.setAttribute("aria-hidden", "true");
  }

  async function runBacktest(panel, strategy, getPayload) {
    const base = getPayload();
    if (!base) return;
    const start = value(panel, "start").trim();
    const end = value(panel, "end").trim();
    if (!/^\d{8}$/.test(start) || !/^\d{8}$/.test(end)) {
      setBacktestStatus(panel, "日期格式应为 YYYYMMDD", true);
      return;
    }
    const payload = {
      ...base,
      review_mode: "after_close",
      end_date: end,
      backtest_strategy: strategy,
      backtest_start_date: start,
      backtest_end_date: end,
      backtest_initial_cash: numberValue(panel, "cash", 1000000),
      backtest_max_positions: numberValue(panel, "maxPositions", 5),
      backtest_max_entries_per_day: numberValue(panel, "maxEntries", 2),
      backtest_signal_top_n: numberValue(panel, "signalTopN", 20),
      backtest_max_hold_days: numberValue(panel, "maxHold", 10),
      backtest_take_profit_pct: numberValue(panel, "takeProfit", 15),
      backtest_stop_loss_pct: numberValue(panel, "stopLoss", 8),
      backtest_exit_ma: value(panel, "exitMa"),
      backtest_slippage_bps: numberValue(panel, "slippage", 5),
      backtest_fee_rate: numberValue(panel, "fee", 0.0003),
      backtest_sell_tax_rate: numberValue(panel, "sellTax", 0.0005),
      backtest_skip_limit_open: panel.querySelector("[data-bt='skipLimit']").checked
    };

    setBacktestStatus(panel, "正在回测，区间越长耗时越久...");
    try {
      const res = await fetch("/api/backtest", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(payload)});
      const data = await res.json();
      if (!data.ok) throw new Error(data.error || "回测失败");
      panel._btTrades = data.trades || [];
      panel._btCurve = data.equity_curve || [];
      renderSummary(panel, data.summary || {}, data.skipped || {});
      renderTable(panel.querySelector("[data-bt='trades']"), panel._btTrades, TRADE_COLS, true);
      renderTable(panel.querySelector("[data-bt='curve']"), panel._btCurve.slice(-80), CURVE_COLS, false);
      setBacktestStatus(panel, `回测完成：${data.start_date} - ${data.end_date}，交易 ${panel._btTrades.length} 笔`);
    } catch (err) {
      setBacktestStatus(panel, `回测失败：${err.message}`, true);
    }
  }

  function renderSummary(panel, summary, skipped) {
    const skippedText = Object.entries(skipped).map(([k, v]) => `${k}:${v}`).join(" / ");
    const items = [
      ["最终权益", money(summary.final_equity)], ["总收益", pct(summary.total_return_pct)], ["年化", pct(summary.annual_return_pct)],
      ["最大回撤", pct(summary.max_drawdown_pct)], ["夏普", summary.sharpe ?? ""], ["交易数", summary.trades ?? 0],
      ["胜率", pct(summary.win_rate_pct)], ["盈亏比", summary.profit_factor ?? ""], ["均笔收益", pct(summary.avg_return_pct)],
      ["平均持有", summary.avg_hold_days ?? ""], ["最好/最差", `${pct(summary.best_trade_pct)} / ${pct(summary.worst_trade_pct)}`],
      ["跳过", skippedText || "0", true]
    ];
    panel.querySelector("[data-bt='summary']").innerHTML = items.map(([k, v, wide]) => (
      `<div class="metric ${wide ? "wide" : ""}"><span>${btEscapeHtml(k)}</span><strong>${formatSigned(v)}</strong></div>`
    )).join("");
  }

  function renderTable(table, rows, cols, colorReturn) {
    if (!rows.length) {
      table.innerHTML = `<tbody><tr><td>暂无数据</td></tr></tbody>`;
      return;
    }
    const head = `<thead><tr>${cols.map(col => `<th>${btEscapeHtml(col[1])}</th>`).join("")}</tr></thead>`;
    const body = `<tbody>${rows.map(row => `<tr>${cols.map(col => {
      const value = row[col[0]] ?? "";
      const cls = colorReturn && col[0] === "return_pct" ? (Number(value) >= 0 ? "positive" : "bad") : "";
      return `<td class="${cls}">${btEscapeHtml(value)}</td>`;
    }).join("")}</tr>`).join("")}</tbody>`;
    table.innerHTML = head + body;
  }

  function exportRows(rows, cols, name) {
    if (!rows.length) { alert("暂无数据"); return; }
    const keys = cols.map(col => col[0]);
    const csv = [cols.map(col => col[1]).join(",")].concat(rows.map(row => (
      keys.map(key => `"${String(row[key] ?? "").replaceAll('"','""')}"`).join(",")
    ))).join("\n");
    const blob = new Blob(["\ufeff" + csv], {type:"text/csv;charset=utf-8"});
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `${name}_${new Date().toISOString().slice(0,10)}.csv`;
    a.click();
    URL.revokeObjectURL(a.href);
  }

  function setBacktestStatus(panel, text, bad=false) {
    const el = panel.querySelector("[data-bt='status']");
    el.textContent = text;
    el.className = bad ? "status bad" : "status";
  }

  function value(panel, key) {
    const el = panel.querySelector(`[data-bt='${key}']`);
    return el ? el.value : "";
  }

  function numberValue(panel, key, fallback) {
    const number = Number(value(panel, key));
    return Number.isFinite(number) ? number : fallback;
  }

  function money(value) {
    const number = Number(value || 0);
    return number.toLocaleString("zh-CN", {maximumFractionDigits: 0});
  }

  function pct(value) {
    const number = Number(value || 0);
    return `${number.toFixed(2)}%`;
  }

  function formatSigned(value) {
    const text = String(value ?? "");
    const number = Number(text.replace("%", "").replaceAll(",", ""));
    const cls = Number.isFinite(number) && text.includes("%") ? (number >= 0 ? "positive" : "bad") : "";
    return `<span class="${cls}">${btEscapeHtml(text)}</span>`;
  }

  function monthsAgo(months) {
    const date = new Date();
    date.setMonth(date.getMonth() - months);
    return date;
  }

  function dateText(date) {
    return `${date.getFullYear()}${String(date.getMonth() + 1).padStart(2, "0")}${String(date.getDate()).padStart(2, "0")}`;
  }

  function btEscapeHtml(value) {
    return String(value ?? "").replace(/[&<>"']/g, ch => ({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"}[ch]));
  }

  window.initBacktestPanel = initBacktestPanel;
})();
