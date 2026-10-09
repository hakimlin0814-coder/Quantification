(function () {
  const TOKEN_KEY = "quant_review_tushare_token";
  const URL_KEY = "quant_review_tushare_url";
  const DEFAULT_SETTINGS = __DATA_SOURCE_DEFAULTS__;

  function getSavedSettings() {
    return {
      token: (localStorage.getItem(TOKEN_KEY) || "").trim(),
      tushare_url: (localStorage.getItem(URL_KEY) || "").trim()
    };
  }

  function getEffectiveSettings() {
    const saved = getSavedSettings();
    return {
      token: saved.token || String(DEFAULT_SETTINGS.token || "").trim(),
      tushare_url: saved.tushare_url || String(DEFAULT_SETTINGS.tushare_url || "").trim()
    };
  }

  function setPageStatus(text, bad) {
    if (typeof window.setStatus === "function") {
      window.setStatus(text, Boolean(bad));
    }
  }

  function ensureModal() {
    if (document.getElementById("dataSourceModal")) return;
    const modal = document.createElement("div");
    modal.className = "modal";
    modal.id = "dataSourceModal";
    modal.setAttribute("aria-hidden", "true");
    modal.onclick = function (event) {
      if (event.target.id === "dataSourceModal") closeDataSourceSettings();
    };
    modal.innerHTML = `
      <div class="modal-card" role="dialog" aria-modal="true" aria-labelledby="dataSourceTitle">
        <div class="modal-head">
          <h2 id="dataSourceTitle">数据源设置</h2>
          <button class="icon-btn" type="button" onclick="closeDataSourceSettings()" title="关闭" aria-label="关闭">x</button>
        </div>
        <div class="modal-body">
          <div style="display:grid; grid-template-columns:1fr; gap:10px;">
            <div>
              <label>Tushare Token</label>
              <input id="globalTushareToken" type="password" autocomplete="off" placeholder="输入你的 Tushare token" />
            </div>
            <div>
              <label>Tushare API地址</label>
              <input id="globalTushareUrl" autocomplete="off" placeholder="输入你的 Tushare API 地址" />
            </div>
            <div id="dataSourceMessage" class="status"></div>
            <div style="display:flex; gap:8px; justify-content:flex-end; flex-wrap:wrap;">
              <button class="ghost" type="button" onclick="closeDataSourceSettings()">取消</button>
              <button type="button" onclick="saveDataSourceSettings()">保存</button>
            </div>
          </div>
        </div>
      </div>`;
    document.body.appendChild(modal);
  }

  function syncFieldsFromStorage() {
    const settings = getEffectiveSettings();
    const token = document.getElementById("globalTushareToken");
    const url = document.getElementById("globalTushareUrl");
    if (token) token.value = settings.token;
    if (url) url.value = settings.tushare_url;
  }

  function updateDataSourceButton() {
    const button = document.getElementById("dataSourceButton");
    if (!button) return;
    const settings = getEffectiveSettings();
    button.textContent = settings.token && settings.tushare_url ? "数据源" : "设置数据源";
    button.className = settings.token && settings.tushare_url ? "ghost" : "secondary";
  }

  window.openDataSourceSettings = function () {
    ensureModal();
    syncFieldsFromStorage();
    const message = document.getElementById("dataSourceMessage");
    if (message) message.textContent = "";
    const modal = document.getElementById("dataSourceModal");
    modal.classList.add("open");
    modal.setAttribute("aria-hidden", "false");
  };

  window.closeDataSourceSettings = function () {
    const modal = document.getElementById("dataSourceModal");
    if (!modal) return;
    modal.classList.remove("open");
    modal.setAttribute("aria-hidden", "true");
  };

  window.saveDataSourceSettings = function () {
    const token = (document.getElementById("globalTushareToken")?.value || "").trim();
    const url = (document.getElementById("globalTushareUrl")?.value || "").trim();
    const message = document.getElementById("dataSourceMessage");
    if (!token || !url) {
      if (message) {
        message.textContent = "请填写 Token 和 API 地址。";
        message.className = "status bad";
      }
      return;
    }
    localStorage.setItem(TOKEN_KEY, token);
    localStorage.setItem(URL_KEY, url);
    updateDataSourceButton();
    closeDataSourceSettings();
    setPageStatus("数据源设置已保存");
  };

  window.getGlobalDataSourcePayload = function () {
    const settings = getEffectiveSettings();
    if (!settings.token || !settings.tushare_url) {
      openDataSourceSettings();
      setPageStatus("请先在右上角“数据源”里填写 Tushare Token 和 API 地址。", true);
      return null;
    }
    return settings;
  };

  function initGlobalDataSourceSettings() {
    ensureModal();
    const actions = document.querySelector(".header-actions");
    if (!actions || document.getElementById("dataSourceButton")) return;
    const button = document.createElement("button");
    button.id = "dataSourceButton";
    button.type = "button";
    button.onclick = openDataSourceSettings;
    button.title = "设置全局 Tushare Token 和 API 地址";
    actions.insertBefore(button, actions.lastElementChild || null);
    updateDataSourceButton();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initGlobalDataSourceSettings);
  } else {
    initGlobalDataSourceSettings();
  }
})();
