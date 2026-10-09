(function () {
  const nav = [
    { group: "ETF", title: "ETF实时强度", href: "/etf-market", note: "全市场手动刷新" },
    { group: "实时", title: "低价暗语新低策略", href: "/price-code-low", note: "实时低点与数字结构" },
    { group: "全局", title: "大盘总览", href: "/", note: "市场温度" },
    { group: "复盘", title: "主线复盘策略", href: "/mainline-review", note: "板块与个股" },
    { group: "复盘", title: "同花顺概念雷达", href: "/concept-radar", note: "概念与自选" },
    { group: "复盘", title: "强势股票×强势概念", href: "/strong-concept-momentum", note: "双重动量交集" },
    { group: "复盘", title: "横截面动量策略", href: "/cross-sectional-momentum", note: "个股/概念排名" },
    { group: "复盘", title: "涨停板策略", href: "/first-limit", note: "首板与1进2" },
    { group: "成长", title: "第一波启动策略", href: "/first-wave", note: "低位启动" },
    { group: "成长", title: "短线主线稳定爆发", href: "/short-mainline", note: "刚爆发主线" },
    { group: "成长", title: "CANSLIM + RPS策略", href: "/canslim-tao", note: "强势成长" },
    { group: "成长", title: "CANSLIM早期防钝化", href: "/canslim-early", note: "阶段与增速" },
    { group: "成长", title: "欧奈尔模型", href: "/oneil-model", note: "原书映射·A股研究" },
    { group: "价值", title: "低估值现金流策略", href: "/value-fcf", note: "估值与FCF" },
    { group: "防守", title: "红利低波防守策略", href: "/dividend-low-vol", note: "高股息低Beta" },
    { group: "工具", title: "市场排名工具", href: "/industry-market-cap-rank", note: "行业/个股" },
    { group: "工具", title: "股票模糊搜索", href: "/stock-search", note: "代码名称拼音" },
    { group: "工具", title: "批量实时MA检查", href: "/realtime-ma", note: "rt_k与均线" }
  ];

  const STRATEGY_VISIBILITY_KEY = "quantReviewVisibleStrategies.v1";
  const DEFAULT_VISIBLE_HREFS = ["/etf-market", "/concept-radar", "/strong-concept-momentum"];
  const MOBILE_NAV_LABELS = {
    "/etf-market": "ETF",
    "/": "总览",
    "/price-code-low": "暗语",
    "/mainline-review": "主线",
    "/concept-radar": "概念",
    "/strong-concept-momentum": "强势",
    "/cross-sectional-momentum": "横截面",
    "/first-limit": "涨停",
    "/first-wave": "一波",
    "/short-mainline": "短线",
    "/canslim-tao": "成长",
    "/canslim-early": "早期",
    "/oneil-model": "欧奈尔",
    "/value-fcf": "价值",
    "/dividend-low-vol": "红利",
    "/industry-market-cap-rank": "排名",
    "/stock-search": "搜索",
    "/realtime-ma": "MA"
  };

  const pageMeta = {
    "/etf-market": {
      group: "ETF",
      title: "ETF实时强度决策",
      description: "全市场ETF统一评分；手动刷新时，交易中给出当前动作，收盘后给出下一交易日优先观察。",
      rail: [["市场评分", "#resultTable"], ["单ETF诊断", "#diagnosis"], ["运行日志", "#logBox"]],
      hint: "所有ETF都展示并评分。可买状态同时要求强度、趋势、流动性与订单容量合格。"
    },
    "/price-code-low": {
      group: "实时",
      title: "低价暗语新低策略",
      description: "用全市场实时最低价严格识别叠数、整串重复、回文和整串顺子，并要求价格同时处在可调窗口的新低。",
      rail: [
        ["筛选参数", "#filterPanel"],
        ["实时摘要", "#summaryCards"],
        ["结果列", "#columnPanel"],
        ["暗语新低结果", "#resultTable"],
        ["运行日志", "#logBox"]
      ],
      hint: "量能变化按截至当前成交量与前若干完整交易日均量比较。盘中缩量只代表当前进度，收盘前仍会变化。"
    },
    "/": {
      group: "全局",
      title: "A股大盘总览",
      description: "先判断市场温度、赚钱效应和流动性，再决定当天各策略的进攻性。",
      rail: [
        ["市场温度", "#marketCards"],
        ["运行日志", "#logBox"]
      ],
      hint: "先看市场状态和建议仓位，再切换到具体策略。弱市里候选池更适合做观察清单。"
    },
    "/mainline-review": {
      group: "复盘",
      title: "主线复盘策略",
      description: "从板块强度、扩散、资金和持续性里找主线，再给出主线内个股观察池。",
      rail: [
        ["市场温度", "#market"],
        ["主线板块", "#boards"],
        ["个股候选", "#stocks"],
        ["运行日志", "#logs"]
      ],
      hint: "主线复盘更像每日工作台：先看方向，再看个股，最后用日志排查数据口径。"
    },
    "/concept-radar": {
      group: "复盘",
      title: "同花顺概念雷达",
      description: "按5/10/20/30/60日查看同花顺概念排行，下钻成分股，加入研究自选池，再做AI新闻和公司地位分析。",
      rail: [
        ["推荐观察", "#recommendBoardsTable"],
        ["概念排行", "#boardsTable"],
        ["成分股明细", "#stocksTable"],
        ["AI研究", "#aiPanel"],
        ["运行日志", "#logBox"]
      ],
      hint: "默认不启用评分、不做强过滤。先看客观排行，再手动加入自选池；AI只用于新闻、主营和公司地位复盘。"
    },
    "/strong-concept-momentum": {
      group: "复盘",
      title: "强势股票×强势概念",
      description: "先按个股区间涨幅取前N%或前N名，再与区间涨幅前N名概念的全部成分股取交集。支持收盘复盘、盘中快照和可见列配置。",
      rail: [
        ["筛选参数", ".controls"],
        ["结果摘要", "#summary"],
        ["交集结果", "#resultTable"],
        ["运行日志", "#status"]
      ],
      hint: "这是候选生成器，不是自动买入信号。盘中结果会随实时价格、概念成员和数据源状态变化。"
    },
    "/cross-sectional-momentum": {
      group: "复盘",
      title: "横截面动量策略",
      description: "在指定 T 基准交易日和有效股票池内比较个股/概念动量，叠加可调涨停、回撤和开盘条件，并生成可解释的推荐股票。",
      rail: [
        ["T与数据模式", "#endDate"],
        ["条件配置", ".panel"],
        ["推荐股票", "#recommendBox"],
        ["筛选结果", "#resultTable"]
      ],
      hint: "实时数据默认关闭；勾选后仍需手动点击获取当前快照。推荐分只代表复盘优先级，不代表买入建议。"
    },
    "/first-limit": {
      group: "复盘",
      title: "涨停板策略",
      description: "筛选首板、连板和次日1进2候选，结合市值、行业排名和实时承接做排除。",
      rail: [
        ["筛选参数", ".controls"],
        ["涨停结果", "#resultTable"],
        ["1进2判断", "#liveTable"],
        ["运行日志", "#logBox"]
      ],
      hint: "涨停策略更看执行纪律。结果表里的风控、排名和实时动作比单个分数更重要。"
    },
    "/first-wave": {
      group: "成长",
      title: "第一波启动策略",
      description: "寻找低位平台压缩后的首次放量突破，以及启动后的首次缩量回踩确认，避免成熟高RPS二波追高。",
      rail: [
        ["策略参数", ".controls"],
        ["市场摘要", "#marketCards"],
        ["启动候选", "#resultTable"],
        ["运行日志", "#logBox"]
      ],
      hint: "第一波策略输出的是观察池和次日计划。盘中模式只做试探，真正买点仍要看收盘确认、次日缺口和板块承接。"
    },
    "/canslim-tao": {
      group: "成长",
      title: "CANSLIM + RPS策略",
      description: "把成长质量、相对强度、形态位置和市场环境合成强势成长观察池。",
      rail: [
        ["筛选条件", "#filterPanel"],
        ["市场摘要", "#marketCards"],
        ["单股分析", "#probeCards"],
        ["候选结果", "#resultTable"],
        ["运行日志", "#logPanel"]
      ],
      hint: "单股分析只评估输入标的；开始筛选才会生成候选池。RPS 需要全市场数据来计算排名。"
    },
    "/short-mainline": {
      group: "成长",
      title: "短线主线稳定爆发策略",
      description: "用申万二级行业发现刚稳定爆发的短线主线，再只在入围行业内找前排、中军和弹性候选。",
      rail: [
        ["策略参数", "#filterPanel"],
        ["主线摘要", "#summaryCards"],
        ["主线行业", "#industryTable"],
        ["行业内个股", "#resultTable"],
        ["运行日志", "#logPanel"]
      ],
      hint: "这个策略先定方向再选个股，默认只保留少数主线和每条主线前排候选，避免把强势噪音铺满页面。"
    },
    "/canslim-early": {
      group: "成长",
      title: "CANSLIM早期防钝化",
      description: "保留CANSLIM和陶博士RPS框架，再叠加阶段识别、价格钝化、放量效率和增速趋势，优先找仍在早期加速的强势股。",
      rail: [
        ["筛选条件", "#filterPanel"],
        ["市场摘要", "#marketCards"],
        ["单股分析", "#probeCards"],
        ["防钝化结果", "#resultTable"],
        ["运行日志", "#logPanel"]
      ],
      hint: "这个页面不是替代原CANSLIM，而是过滤高位钝化版本。重点看阶段、钝化分、RPS20变化、20日涨幅和增速趋势。"
    },
    "/oneil-model": {
      group: "成长",
      title: "欧奈尔模型",
      description: "按原书、合理推导、A股适配和待确认四层来源，构建可追溯的Q0研究观察池。",
      rail: [
        ["研究状态", "#researchStatus"],
        ["市场环境", "#marketCards"],
        ["策略配置", "#filterPanel"],
        ["Q0观察结果", "#resultTable"],
        ["运行日志", "#logPanel"]
      ],
      hint: "Q0通过只表示已实现规则门通过；完整形态、机构I和M买入门控仍未完成，因此不等于买入信号。"
    },
    "/value-fcf": {
      group: "价值",
      title: "低估值现金流策略",
      description: "用 PB、PE_TTM、ROE 和自由现金流收益率，寻找估值不贵且现金流扎实的股票。",
      rail: [
        ["策略参数", ".controls"],
        ["财务摘要", "#summaryCards"],
        ["单股分析", "#probeCards"],
        ["候选结果", "#resultTable"],
        ["运行日志", "#logBox"]
      ],
      hint: "FCF 率使用 TTM 自由现金流 / 总市值。接口不足时会使用东财现金流量表做近似兜底。"
    },
    "/dividend-low-vol": {
      group: "防守",
      title: "红利低波防守策略",
      description: "面向2浪或A浪初期，用股息率、低波动、低Beta、质量和防守行业特征筛出回撤期观察池。",
      rail: [
        ["策略参数", ".controls"],
        ["策略摘要", "#summaryCards"],
        ["单股分析", "#probeCards"],
        ["候选结果", "#resultTable"],
        ["运行日志", "#logBox"]
      ],
      hint: "这不是追涨策略。优先看高股息是否可持续、波动是否真的低、价格是否没有远离MA60。"
    },
    "/stock-search": {
      group: "工具",
      title: "股票模糊搜索",
      description: "输入股票代码、中文名称片段或拼音片段，快速查出所有匹配的A股股票。",
      rail: [
        ["搜索参数", ".controls"],
        ["搜索摘要", "#summaryCards"],
        ["搜索结果", "#resultTable"],
        ["运行日志", "#logBox"]
      ],
      hint: "可以搜中文片段，也可以搜拼音片段。未安装完整拼音库时，会使用内置常见公司名用字映射。"
    },
    "/industry-market-cap-rank": {
      group: "工具",
      title: "市场排名工具",
      description: "支持行业总市值、个股总市值和个股股价排名，盘后使用收盘数据，盘中实时优先使用实时行情。",
      rail: [
        ["排名参数", ".controls"],
        ["排名摘要", "#summaryCards"],
        ["排名结果", "#resultTable"],
        ["运行日志", "#logBox"]
      ],
      hint: "市值来自 daily_basic.total_mv，盘中实时个股市值会按实时价相对昨收做近似更新；实时行情不可用时回退到最近收盘数据。"
    },
    "/realtime-ma": {
      group: "工具",
      title: "批量实时MA检查",
      description: "粘贴多只股票代码或名称，批量读取实时日K，检查MA5、MA10和MA20回踩条件。",
      rail: [
        ["输入列表", "#filterPanel"],
        ["检查摘要", "#summaryPanel"],
        ["结果列", "#columnPanel"],
        ["实时表格", "#resultTable"],
        ["运行日志", "#logPanel"]
      ],
      hint: "三种回踩条件分别显示是否。实时价来自 rt_k 或东财兜底，均线由历史收盘价加当前实时价计算。"
    }
  };

  function normalizePath() {
    const path = window.location.pathname === "/index.html" ? "/" : window.location.pathname;
    return pageMeta[path] ? path : "/";
  }

  function knownHrefSet() {
    return new Set(nav.map(item => item.href));
  }

  function uniqueKnownHrefs(hrefs) {
    const known = knownHrefSet();
    const seen = new Set();
    return (Array.isArray(hrefs) ? hrefs : [])
      .filter(href => known.has(href))
      .filter(href => {
        if (seen.has(href)) return false;
        seen.add(href);
        return true;
      });
  }

  function readVisibleHrefs() {
    try {
      const raw = window.localStorage.getItem(STRATEGY_VISIBILITY_KEY);
      if (!raw) return DEFAULT_VISIBLE_HREFS.slice();
      const parsed = JSON.parse(raw);
      const hrefs = uniqueKnownHrefs(parsed);
      return hrefs.length ? hrefs : DEFAULT_VISIBLE_HREFS.slice();
    } catch (_) {
      return DEFAULT_VISIBLE_HREFS.slice();
    }
  }

  function writeVisibleHrefs(hrefs) {
    const normalized = uniqueKnownHrefs(hrefs);
    const finalHrefs = normalized.length ? normalized : DEFAULT_VISIBLE_HREFS.slice();
    try {
      window.localStorage.setItem(STRATEGY_VISIBILITY_KEY, JSON.stringify(finalHrefs));
    } catch (_) {}
    return finalHrefs;
  }

  function visibleHrefSet() {
    return new Set(readVisibleHrefs());
  }

  function navItemByHref(href) {
    return nav.find(item => item.href === href) || null;
  }

  function getVisibleNavItems(options = {}) {
    const visible = visibleHrefSet();
    const items = nav.filter(item => visible.has(item.href));
    if (options.includeCurrent && options.currentPath && !items.some(item => item.href === options.currentPath)) {
      const current = navItemByHref(options.currentPath);
      if (current) return [current, ...items];
    }
    return items;
  }

  function groupNavItems(items) {
    const groups = [];
    items.forEach(item => {
      let group = groups.find(entry => entry.name === item.group);
      if (!group) {
        group = { name: item.group, items: [] };
        groups.push(group);
      }
      group.items.push(item);
    });
    return groups;
  }

  function escapeHtml(value) {
    return String(value ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function syncStrategySelects(currentPath) {
    const visible = visibleHrefSet();
    const items = getVisibleNavItems({ includeCurrent:true, currentPath });
    document.querySelectorAll("select.strategy-nav").forEach(select => {
      const selectedPath = navItemByHref(currentPath) ? currentPath : (select.value || "/concept-radar");
      select.innerHTML = items.map(item => {
        const hiddenLabel = visible.has(item.href) ? "" : "（已隐藏）";
        return `<option value="${escapeHtml(item.href)}"${item.href === selectedPath ? " selected" : ""}>${escapeHtml(item.title + hiddenLabel)}</option>`;
      }).join("");
      select.value = selectedPath;
    });
  }

  function makeSidebar(currentPath) {
    const aside = document.createElement("aside");
    aside.className = "app-sidebar";
    const groups = groupNavItems(getVisibleNavItems());
    aside.innerHTML = `
      <button class="app-sidebar-toggle" type="button" aria-label="收起菜单" title="收起菜单">
        <span aria-hidden="true">‹</span>
      </button>
      <div class="app-sidebar-content">
        <div class="app-back">← Home</div>
        <h2 class="app-sidebar-title">量化复盘工作台</h2>
        ${groups.length ? groups.map(group => `
          <div class="app-nav-group">${group.name}</div>
          ${group.items.map(item => `
            <a class="${item.href === currentPath ? "active" : ""}" href="${item.href}">
              <span>${item.title}</span>
              <small>${item.note}</small>
            </a>
          `).join("")}
        `).join("") : `<div class="app-nav-empty">暂无显示策略，可在设置中恢复。</div>`}
        <div class="app-sidebar-tools">
          <button class="app-nav-settings" type="button" data-open-strategy-settings="1">策略显示设置</button>
          <small>隐藏只影响入口，不会禁用页面。</small>
        </div>
      </div>
    `;
    return aside;
  }

  function makeInstallButton() {
    const button = document.createElement("button");
    button.className = "app-install-button";
    button.type = "button";
    button.hidden = true;
    button.textContent = "安装";
    return button;
  }

  function makeStrategySheet(currentPath) {
    const items = getVisibleNavItems();
    const sheet = document.createElement("div");
    sheet.className = "mobile-strategy-sheet";
    sheet.setAttribute("aria-hidden", "true");
    sheet.innerHTML = `
      <div class="mobile-strategy-backdrop" data-close-sheet="1"></div>
      <section class="mobile-strategy-panel" role="dialog" aria-modal="true" aria-label="全部策略">
        <div class="mobile-strategy-head">
          <div>
            <strong>全部策略</strong>
            <span>电脑和手机共用同一套策略引擎</span>
          </div>
          <div class="mobile-strategy-actions">
            <button type="button" class="mobile-settings-btn" data-open-strategy-settings="1">显示设置</button>
            <button type="button" class="mobile-sheet-close" data-close-sheet="1" aria-label="关闭">×</button>
          </div>
        </div>
        <div class="mobile-strategy-grid">
          ${items.map(item => `
            <a class="${item.href === currentPath ? "active" : ""}" href="${item.href}">
              <span>${item.title}</span>
              <small>${item.group} · ${item.note}</small>
            </a>
          `).join("")}
        </div>
      </section>
    `;
    return sheet;
  }

  function makeMobileBottomNav(currentPath) {
    const navItems = getVisibleNavItems().slice(0, 4).map(item => ({
      label: MOBILE_NAV_LABELS[item.href] || item.title.slice(0, 2),
      href: item.href
    }));
    const bar = document.createElement("nav");
    bar.className = "mobile-bottom-nav";
    bar.setAttribute("aria-label", "手机底部导航");
    bar.style.gridTemplateColumns = `repeat(${Math.max(2, Math.min(navItems.length + 1, 5))}, 1fr)`;
    bar.innerHTML = `
      ${navItems.map(item => `
        <a class="${item.href === currentPath ? "active" : ""}" href="${item.href}">
          <span>${item.label}</span>
        </a>
      `).join("")}
      <button type="button" data-open-strategy-sheet="1">
        <span>全部</span>
      </button>
    `;
    return bar;
  }

  function openStrategySheet() {
    const sheet = document.querySelector(".mobile-strategy-sheet");
    if (!sheet) return;
    sheet.classList.add("open");
    sheet.setAttribute("aria-hidden", "false");
    document.body.classList.add("mobile-sheet-open");
  }

  function closeStrategySheet() {
    const sheet = document.querySelector(".mobile-strategy-sheet");
    if (!sheet) return;
    sheet.classList.remove("open");
    sheet.setAttribute("aria-hidden", "true");
    document.body.classList.remove("mobile-sheet-open");
  }

  function makeStrategySettingsModal() {
    const modal = document.createElement("div");
    modal.className = "strategy-settings-modal";
    modal.setAttribute("aria-hidden", "true");
    modal.innerHTML = `
      <div class="strategy-settings-backdrop" data-close-strategy-settings="1"></div>
      <section class="strategy-settings-card" role="dialog" aria-modal="true" aria-label="策略显示设置">
        <div class="strategy-settings-head">
          <div>
            <h2>策略显示设置</h2>
            <p>勾选后才显示在左侧导航、手机策略面板和顶部策略下拉中。</p>
          </div>
          <button type="button" class="strategy-settings-close" data-close-strategy-settings="1" aria-label="关闭">×</button>
        </div>
        <div class="strategy-settings-toolbar">
          <input type="search" placeholder="搜索策略..." data-strategy-settings-search />
        </div>
        <div class="strategy-settings-list" data-strategy-settings-list></div>
        <div class="strategy-settings-footer">
          <span data-strategy-settings-status></span>
          <div>
            <button type="button" class="secondary" data-strategy-settings-action="all">全部显示</button>
            <button type="button" class="secondary" data-strategy-settings-action="default">恢复默认</button>
            <button type="button" data-strategy-settings-action="save">保存</button>
          </div>
        </div>
      </section>
    `;
    return modal;
  }

  function renderStrategySettingsList() {
    const modal = document.querySelector(".strategy-settings-modal");
    if (!modal) return;
    const list = modal.querySelector("[data-strategy-settings-list]");
    if (!list) return;
    const visible = visibleHrefSet();
    const groups = groupNavItems(nav);
    list.innerHTML = groups.map(group => `
      <div class="strategy-settings-group">
        <div class="strategy-settings-group-title">${escapeHtml(group.name)}</div>
        ${group.items.map(item => `
          <label class="strategy-settings-item" data-strategy-settings-item data-search-text="${escapeHtml(`${item.group} ${item.title} ${item.note} ${item.href}`.toLowerCase())}">
            <input type="checkbox" value="${escapeHtml(item.href)}" ${visible.has(item.href) ? "checked" : ""} />
            <span>
              <strong>${escapeHtml(item.title)}</strong>
              <small>${escapeHtml(item.note)} · ${escapeHtml(item.href)}</small>
            </span>
          </label>
        `).join("")}
      </div>
    `).join("");
    filterStrategySettingsList();
  }

  function filterStrategySettingsList() {
    const modal = document.querySelector(".strategy-settings-modal");
    if (!modal) return;
    const search = modal.querySelector("[data-strategy-settings-search]");
    const keyword = String(search?.value || "").trim().toLowerCase();
    modal.querySelectorAll(".strategy-settings-group").forEach(group => {
      let visibleCount = 0;
      group.querySelectorAll("[data-strategy-settings-item]").forEach(item => {
        const matched = !keyword || String(item.dataset.searchText || "").includes(keyword);
        item.hidden = !matched;
        if (matched) visibleCount += 1;
      });
      group.hidden = visibleCount === 0;
    });
  }

  function setStrategySettingsChecks(mode) {
    const modal = document.querySelector(".strategy-settings-modal");
    if (!modal) return;
    const boxes = Array.from(modal.querySelectorAll(".strategy-settings-item input[type='checkbox']"));
    boxes.forEach(box => {
      if (mode === "all") box.checked = true;
      if (mode === "default") box.checked = DEFAULT_VISIBLE_HREFS.includes(box.value);
    });
  }

  function openStrategySettings() {
    closeStrategySheet();
    const modal = document.querySelector(".strategy-settings-modal");
    if (!modal) return;
    const status = modal.querySelector("[data-strategy-settings-status]");
    const search = modal.querySelector("[data-strategy-settings-search]");
    if (status) status.textContent = "默认显示ETF实时强度和同花顺概念雷达。";
    if (search) search.value = "";
    renderStrategySettingsList();
    modal.classList.add("open");
    modal.setAttribute("aria-hidden", "false");
    document.body.classList.add("strategy-settings-open");
    window.setTimeout(() => search?.focus(), 80);
  }

  function closeStrategySettings() {
    const modal = document.querySelector(".strategy-settings-modal");
    if (!modal) return;
    modal.classList.remove("open");
    modal.setAttribute("aria-hidden", "true");
    document.body.classList.remove("strategy-settings-open");
  }

  function saveStrategySettings() {
    const modal = document.querySelector(".strategy-settings-modal");
    if (!modal) return;
    const status = modal.querySelector("[data-strategy-settings-status]");
    const checked = Array.from(modal.querySelectorAll(".strategy-settings-item input[type='checkbox']:checked")).map(box => box.value);
    const finalHrefs = writeVisibleHrefs(checked);
    if (status) status.textContent = finalHrefs.length ? "已保存，正在刷新界面..." : "已恢复默认，正在刷新界面...";
    window.setTimeout(() => window.location.reload(), 160);
  }

  function setupStrategySettings() {
    document.addEventListener("click", event => {
      const openButton = event.target.closest("[data-open-strategy-settings]");
      if (openButton) {
        event.preventDefault();
        openStrategySettings();
        return;
      }
      if (event.target.closest("[data-close-strategy-settings]")) {
        event.preventDefault();
        closeStrategySettings();
        return;
      }
      const actionButton = event.target.closest("[data-strategy-settings-action]");
      if (!actionButton) return;
      event.preventDefault();
      const action = actionButton.dataset.strategySettingsAction;
      if (action === "all") setStrategySettingsChecks("all");
      if (action === "default") setStrategySettingsChecks("default");
      if (action === "save") saveStrategySettings();
    });
    document.addEventListener("input", event => {
      if (event.target.matches?.("[data-strategy-settings-search]")) filterStrategySettingsList();
    });
    document.addEventListener("keydown", event => {
      if (event.key === "Escape") closeStrategySettings();
    });
  }

  function setupMobileShell() {
    document.addEventListener("click", event => {
      if (event.target.closest("[data-open-strategy-sheet]")) {
        event.preventDefault();
        openStrategySheet();
      }
      if (event.target.closest("[data-close-sheet]")) {
        event.preventDefault();
        closeStrategySheet();
      }
    });
    document.addEventListener("keydown", event => {
      if (event.key === "Escape") closeStrategySheet();
    });
  }

  function setupPwaInstall(button) {
    if (!button) return;
    let deferredPrompt = null;
    window.addEventListener("beforeinstallprompt", event => {
      event.preventDefault();
      deferredPrompt = event;
      button.hidden = false;
    });
    button.addEventListener("click", async () => {
      if (!deferredPrompt) return;
      deferredPrompt.prompt();
      await deferredPrompt.userChoice;
      deferredPrompt = null;
      button.hidden = true;
    });
    window.addEventListener("appinstalled", () => {
      deferredPrompt = null;
      button.hidden = true;
    });
  }

  function registerServiceWorker() {
    if (!("serviceWorker" in navigator) || !window.isSecureContext) return;
    navigator.serviceWorker.register("/service-worker.js").catch(() => {});
  }

  function applySidebarState(collapsed) {
    document.body.classList.toggle("app-sidebar-collapsed", collapsed);
    const toggle = document.querySelector(".app-sidebar-toggle");
    if (!toggle) return;
    toggle.setAttribute("aria-label", collapsed ? "展开菜单" : "收起菜单");
    toggle.setAttribute("title", collapsed ? "展开菜单" : "收起菜单");
    const icon = toggle.querySelector("span");
    if (icon) icon.textContent = collapsed ? "›" : "‹";
  }

  function setupSidebarToggle() {
    const toggle = document.querySelector(".app-sidebar-toggle");
    if (!toggle) return;
    const saved = window.localStorage.getItem("quantReviewSidebarCollapsed") === "1";
    applySidebarState(saved);
    toggle.addEventListener("click", () => {
      const collapsed = !document.body.classList.contains("app-sidebar-collapsed");
      window.localStorage.setItem("quantReviewSidebarCollapsed", collapsed ? "1" : "0");
      applySidebarState(collapsed);
    });
  }

  function makeHero(meta) {
    const hero = document.createElement("section");
    hero.className = "app-hero";
    hero.innerHTML = `
      <div class="app-breadcrumbs">量化复盘 / ${meta.group} /</div>
      <h2>${meta.title}</h2>
      <p>${meta.description}</p>
    `;
    return hero;
  }

  function makeRail(meta) {
    const rail = document.createElement("aside");
    rail.className = "app-right-rail";
    rail.innerHTML = `
      <h2>本页导航</h2>
      ${meta.rail.map(([label, target]) => `<a href="${target}">${label}</a>`).join("")}
      <div class="hint">${meta.hint}</div>
    `;
    rail.addEventListener("click", event => {
      const link = event.target.closest("a");
      if (!link) return;
      const href = link.getAttribute("href");
      if (!href || !href.startsWith("#") && !href.startsWith(".")) return;
      const target = document.querySelector(href);
      if (!target) return;
      event.preventDefault();
      target.scrollIntoView({ behavior: "smooth", block: "start" });
    });
    return rail;
  }

  function moveHeaderContent(main) {
    const header = document.querySelector("header");
    if (!header || !main) return;
    const filterPanel = header.querySelector("#filterPanel");
    const controls = header.querySelector(".controls");
    const status = header.querySelector(".status");
    const movable = filterPanel || controls;
    if (movable) {
      movable.id = movable.id || "pageControls";
      main.insertBefore(movable, main.children[1] || null);
    }
    if (status) {
      main.insertBefore(status, movable ? movable.nextSibling : main.children[1] || null);
    }
  }

  function textOf(node) {
    return String(node?.textContent || "").replace(/\s+/g, " ").trim();
  }

  function cellHasControl(cell) {
    return Boolean(cell.querySelector("button, input, select, textarea, a"));
  }

  function ensureMobileCardList(table) {
    const wrap = table.closest(".table-wrap");
    if (!wrap) return null;
    let list = wrap.nextElementSibling;
    if (!list || !list.classList.contains("mobile-card-list") || list.dataset.tableRef !== table.dataset.mobileTableId) {
      list = document.createElement("div");
      list.className = "mobile-card-list";
      list.dataset.tableRef = table.dataset.mobileTableId;
      wrap.insertAdjacentElement("afterend", list);
    }
    return list;
  }

  function tableHeaders(table) {
    const headRow = table.querySelector("thead tr");
    if (!headRow) return [];
    return Array.from(headRow.children).map((cell, index) => {
      const text = textOf(cell);
      return text || (cell.querySelector("input") ? "选择" : `字段${index + 1}`);
    });
  }

  function cloneCell(cell) {
    const value = document.createElement("div");
    value.className = "mobile-card-value";
    Array.from(cell.childNodes).forEach(node => value.appendChild(node.cloneNode(true)));
    return value;
  }

  function makeMobileField(label, cell) {
    const field = document.createElement("div");
    field.className = "mobile-card-field";
    const labelEl = document.createElement("div");
    labelEl.className = "mobile-card-label";
    labelEl.textContent = label;
    field.appendChild(labelEl);
    field.appendChild(cloneCell(cell));
    return field;
  }

  function makeMobileCard(headers, row) {
    const cells = Array.from(row.children);
    const entries = cells.map((cell, index) => ({
      label: headers[index] || `字段${index + 1}`,
      cell,
      text: textOf(cell),
      isAction: /^(操作|选择)$/.test(headers[index] || "") || cellHasControl(cell)
    })).filter(entry => entry.text || cellHasControl(entry.cell));

    const dataEntries = entries.filter(entry => !entry.isAction);
    const actionEntries = entries.filter(entry => entry.isAction);
    const titleText = dataEntries.slice(0, 2).map(entry => entry.text).filter(Boolean).join(" ");
    const subtitleText = dataEntries.slice(2, 5).map(entry => `${entry.label} ${entry.text}`).filter(Boolean).join(" | ");

    const card = document.createElement("article");
    card.className = "mobile-table-card";

    const head = document.createElement("div");
    head.className = "mobile-table-card-head";
    const titleWrap = document.createElement("div");
    const title = document.createElement("div");
    title.className = "mobile-table-card-title";
    title.textContent = titleText || "结果";
    titleWrap.appendChild(title);
    if (subtitleText) {
      const subtitle = document.createElement("div");
      subtitle.className = "mobile-table-card-subtitle";
      subtitle.textContent = subtitleText;
      titleWrap.appendChild(subtitle);
    }
    head.appendChild(titleWrap);
    card.appendChild(head);

    const primary = document.createElement("div");
    primary.className = "mobile-card-fields";
    dataEntries.slice(0, 8).forEach(entry => primary.appendChild(makeMobileField(entry.label, entry.cell)));
    card.appendChild(primary);

    if (dataEntries.length > 8) {
      const more = document.createElement("details");
      const summary = document.createElement("summary");
      summary.textContent = `更多字段 ${dataEntries.length - 8}`;
      const moreFields = document.createElement("div");
      moreFields.className = "mobile-card-fields";
      dataEntries.slice(8).forEach(entry => moreFields.appendChild(makeMobileField(entry.label, entry.cell)));
      more.appendChild(summary);
      more.appendChild(moreFields);
      card.appendChild(more);
    }

    if (actionEntries.length) {
      const actions = document.createElement("div");
      actions.className = "mobile-card-actions";
      actionEntries.forEach(entry => actions.appendChild(cloneCell(entry.cell)));
      card.appendChild(actions);
    }
    return card;
  }

  function buildMobileCardsForTable(table) {
    const wrap = table.closest(".table-wrap");
    if (!wrap) return;
    if (wrap.classList.contains("keep-mobile-table") || wrap.dataset.mobileCards === "off" || table.dataset.mobileCards === "off") {
      wrap.classList.remove("mobile-cards-ready");
      const list = wrap.nextElementSibling;
      if (list?.classList.contains("mobile-card-list")) {
        list.remove();
      }
      return;
    }
    table.dataset.mobileTableId = table.dataset.mobileTableId || `mobile-table-${Math.random().toString(36).slice(2)}`;
    const rows = Array.from(table.querySelectorAll("tbody tr"));
    const signature = `${table.innerHTML.length}:${rows.length}`;
    if (table.dataset.mobileSignature === signature) return;
    table.dataset.mobileSignature = signature;

    const list = ensureMobileCardList(table);
    if (!list) return;
    list.innerHTML = "";
    const headers = tableHeaders(table);
    const realRows = rows.filter(row => row.children.length && !row.querySelector("td[colspan]"));
    if (!realRows.length) {
      const empty = document.createElement("div");
      empty.className = "mobile-table-empty";
      empty.textContent = textOf(rows[0]) || "暂无数据";
      list.appendChild(empty);
      wrap.classList.toggle("mobile-cards-ready", Boolean(rows.length));
      return;
    }
    realRows.forEach(row => list.appendChild(makeMobileCard(headers, row)));
    wrap.classList.add("mobile-cards-ready");
  }

  function buildMobileCards() {
    document.querySelectorAll(".table-wrap table").forEach(buildMobileCardsForTable);
  }

  function setupMobileTableCards() {
    let timer = 0;
    const schedule = () => {
      window.clearTimeout(timer);
      timer = window.setTimeout(buildMobileCards, 80);
    };
    buildMobileCards();
    const observer = new MutationObserver(mutations => {
      const shouldRebuild = mutations.some(mutation => {
        const target = mutation.target.nodeType === 1 ? mutation.target : mutation.target.parentElement;
        if (!target || target.closest(".mobile-card-list")) return false;
        return target.matches?.("table, thead, tbody, tr, td, th") || target.querySelector?.(".table-wrap table");
      });
      if (shouldRebuild) schedule();
    });
    observer.observe(document.body, { childList:true, subtree:true, characterData:true });
  }

  function initLayout() {
    if (document.body.classList.contains("app-layout-ready")) return;
    const header = document.querySelector("header");
    const main = document.querySelector("main");
    if (!header || !main) return;

    const currentPath = normalizePath();
    const meta = pageMeta[currentPath];
    const pageTitle = header.querySelector("h1");
    if (pageTitle) pageTitle.textContent = "A股量化复盘";
    syncStrategySelects(currentPath);

    document.body.classList.add("app-layout-ready");
    const installButton = makeInstallButton();
    const actions = header.querySelector(".header-actions");
    if (actions) actions.appendChild(installButton);
    main.insertBefore(makeHero(meta), main.firstChild);
    moveHeaderContent(main);
    document.body.insertBefore(makeSidebar(currentPath), main);
    document.body.appendChild(makeMobileBottomNav(currentPath));
    document.body.appendChild(makeStrategySheet(currentPath));
    document.body.appendChild(makeStrategySettingsModal());
    setupSidebarToggle();
    setupMobileShell();
    setupStrategySettings();
    setupPwaInstall(installButton);
    registerServiceWorker();
    setupMobileTableCards();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initLayout);
  } else {
    initLayout();
  }
})();
