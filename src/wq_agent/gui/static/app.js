const state = {
  configFields: [],
  currentTab: "run",
  pollTimer: null,
  csrfToken: "",
};
const CLEAR_SECRET_VALUE = "__clear_secret__";
const GLOBAL_MODEL_OPTIONS = {
  openai: ["gpt-5.4", "gpt-5.4-mini", "gpt-5.3-codex"],
  kimi: ["kimi-k2.6"],
  deepseek: ["deepseek-chat", "deepseek-reasoner"],
};

const $ = (id) => document.getElementById(id);

document.addEventListener("DOMContentLoaded", () => {
  bindTabs();
  bindActions();
  bootstrap();
});

function bindTabs() {
  document.querySelectorAll(".nav-btn").forEach((button) => {
    button.addEventListener("click", () => {
      state.currentTab = button.dataset.tab;
      document.querySelectorAll(".nav-btn").forEach((item) => item.classList.remove("active"));
      document.querySelectorAll(".tab-panel").forEach((item) => item.classList.remove("active"));
      button.classList.add("active");
      $(`tab-${state.currentTab}`).classList.add("active");
      if (state.currentTab === "results") {
        refreshResults();
      }
    });
  });
}

function bindActions() {
  $("btnSaveConfig").addEventListener("click", saveConfig);
  $("btnRefreshResults").addEventListener("click", refreshResults);
  $("btnCancelJob").addEventListener("click", cancelJob);

  $("btnGenerate").addEventListener("click", () => startTask("generate", {
    strategy: $("generateStrategy").value,
    count: $("generateCount").value,
    idea: $("generateIdea").value,
    no_backtest: $("generateNoBacktest").checked,
    verbose: $("generateVerbose").checked,
  }));

  $("btnRunFull").addEventListener("click", () => startTask("run", {
    strategy: $("runStrategy").value,
    count: $("runCount").value,
    batches: $("runBatches").value,
    interval: $("runInterval").value,
    idea: $("runIdea").value,
    verbose: $("runVerbose").checked,
  }));

  $("btnBacktest").addEventListener("click", () => startTask("backtest", {
    mode: $("backtestMode").value,
    ids: $("backtestIds").value,
    concurrent: $("backtestConcurrent").value,
    verbose: $("backtestVerbose").checked,
  }));

  $("btnRefine").addEventListener("click", () => startTask("refine", {
    base_id: $("refineBaseId").value,
    count: $("refineCount").value,
    no_backtest: $("refineNoBacktest").checked,
    verbose: $("refineVerbose").checked,
  }));
}

async function api(path, options = {}) {
  const headers = { "Content-Type": "application/json", ...(options.headers || {}) };
  if (state.csrfToken && path !== "/api/meta") {
    headers["X-WQ-Agent-CSRF"] = state.csrfToken;
  }
  const response = await fetch(path, {
    ...options,
    headers,
  });
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error || "请求失败");
  }
  return data;
}

async function bootstrap() {
  try {
    const meta = await api("/api/meta");
    state.csrfToken = meta.csrf_token;
    $("workspacePath").textContent = meta.workspace;
  } catch (error) {
    toast(error.message);
  }
  loadConfig();
  refreshResults();
  pollJob();
  state.pollTimer = window.setInterval(pollJob, 1500);
}

async function loadConfig() {
  try {
    const data = await api("/api/config");
    state.configFields = data.fields;
    $("envPath").textContent = data.env_path;
    renderConfig(data.fields);
    renderConfigStatus(data.fields);
  } catch (error) {
    toast(error.message);
  }
}

function renderConfig(fields) {
  const groups = new Map();
  fields.forEach((field) => {
    if (!groups.has(field.section)) {
      groups.set(field.section, []);
    }
    groups.get(field.section).push(field);
  });

  $("configForm").innerHTML = "";
  for (const [section, items] of groups) {
    const sectionEl = document.createElement("section");
    sectionEl.className = "config-section";
    sectionEl.innerHTML = `<h4>${escapeHtml(section)}</h4>`;
    const fieldsEl = document.createElement("div");
    fieldsEl.className = "config-fields";
    items.forEach((field) => fieldsEl.appendChild(configField(field)));
    sectionEl.appendChild(fieldsEl);
    $("configForm").appendChild(sectionEl);
  }
}

function configField(field) {
  const label = document.createElement("label");
  label.className = "config-field";
  const labelText = document.createElement("span");
  labelText.className = "config-label";
  labelText.textContent = field.label;

  if (field.kind === "boolean") {
    label.classList.add("config-switch-field");
    const input = document.createElement("input");
    input.type = "checkbox";
    input.dataset.configKey = field.key;
    input.checked = String(field.value).toLowerCase() === "true";
    input.addEventListener("change", refreshConfigStatusFromForm);
    const switchEl = document.createElement("span");
    switchEl.className = "switch";
    switchEl.setAttribute("aria-hidden", "true");
    label.appendChild(labelText);
    label.appendChild(input);
    label.appendChild(switchEl);
    return label;
  }

  const input = field.kind === "select" ? document.createElement("select") : document.createElement("input");
  input.dataset.configKey = field.key;
  input.addEventListener(field.kind === "select" ? "change" : "input", refreshConfigStatusFromForm);

  if (field.kind === "select") {
    const options = Array.isArray(field.options) ? [...field.options] : [];
    const current = field.value || "";
    if (current && !options.includes(current)) {
      options.push(current);
    }
    options.forEach((option) => {
      const optionEl = document.createElement("option");
      optionEl.value = option;
      optionEl.textContent = option || "使用供应商默认";
      input.appendChild(optionEl);
    });
    input.value = current;
  } else {
    input.value = field.secret ? "" : field.value || "";
    input.placeholder = field.secret && field.has_value ? "已设置，留空保持不变" : "";
    input.type = field.secret ? "password" : field.kind === "number" ? "number" : "text";
  }

  label.appendChild(labelText);

  if (field.secret) {
    const wrap = document.createElement("div");
    wrap.className = "secret-wrap";
    const reveal = document.createElement("button");
    reveal.type = "button";
    reveal.textContent = "显示";
    reveal.addEventListener("click", () => {
      input.type = input.type === "password" ? "text" : "password";
      reveal.textContent = input.type === "password" ? "显示" : "隐藏";
    });
    const clear = document.createElement("button");
    clear.type = "button";
    clear.textContent = "清除";
    clear.disabled = !field.has_value;
    clear.addEventListener("click", () => {
      const active = input.dataset.clearSecret !== "true";
      input.dataset.clearSecret = active ? "true" : "false";
      input.disabled = active;
      reveal.disabled = active;
      clear.textContent = active ? "保留" : "清除";
      clear.classList.toggle("danger-inline", active);
      refreshConfigStatusFromForm();
    });
    wrap.appendChild(input);
    wrap.appendChild(reveal);
    wrap.appendChild(clear);
    label.appendChild(wrap);
  } else {
    label.appendChild(input);
  }

  return label;
}

async function saveConfig() {
  const values = {};
  document.querySelectorAll("[data-config-key]").forEach((input) => {
    const key = input.dataset.configKey;
    if (input.dataset.clearSecret === "true") {
      values[key] = CLEAR_SECRET_VALUE;
    } else if (input.type === "checkbox") {
      values[key] = input.checked;
    } else {
      values[key] = input.value;
    }
  });

  try {
    const data = await api("/api/config", {
      method: "POST",
      body: JSON.stringify({ values }),
    });
    state.configFields = data.fields;
    renderConfig(data.fields);
    renderConfigStatus(data.fields);
    toast("配置已保存");
  } catch (error) {
    toast(error.message);
  }
}

function renderConfigStatus(fields) {
  const byKey = Object.fromEntries(fields.map((field) => [field.key, field]));
  const provider = String(byKey.LLM_PROVIDER?.value || "openai").toLowerCase();
  const providerKey = {
    openai: "OPENAI_API_KEY",
    kimi: "KIMI_API_KEY",
    deepseek: "DEEPSEEK_API_KEY",
  }[provider] || "OPENAI_API_KEY";
  const globalModel = byKey.LLM_MODEL?.value || "";
  const globalModelValid = !globalModel || (GLOBAL_MODEL_OPTIONS[provider] || []).includes(globalModel);
  const model = globalModel
    ? (globalModelValid ? globalModel : "")
    : byKey[`${provider.toUpperCase()}_MODEL`]?.value || byKey.OPENAI_MODEL?.value;
  const checks = [
    ["模型供应商", Boolean(provider)],
    ["模型密钥", Boolean(byKey[providerKey]?.has_value)],
    ["当前模型", globalModelValid && Boolean(model)],
    ["WQ 账号", Boolean(byKey.WQ_USERNAME?.value && byKey.WQ_PASSWORD?.has_value)],
    ["正式提交", false],
  ];
  $("configStatus").innerHTML = checks.map(([name, ok]) => {
    const label = name === "正式提交" ? "隐藏" : ok ? "就绪" : "未配置";
    const dot = ok ? "dot ok" : "dot";
    return `<div class="status-item"><span><i class="${dot}"></i>${name}</span><strong>${label}</strong></div>`;
  }).join("");
}

function refreshConfigStatusFromForm() {
  if (!state.configFields.length) {
    return;
  }
  const liveFields = state.configFields.map((field) => {
    const input = document.querySelector(`[data-config-key="${field.key}"]`);
    if (!input) {
      return field;
    }
    const next = { ...field };
    if (field.secret) {
      if (input.dataset.clearSecret === "true") {
        next.has_value = false;
        next.value = "";
      } else {
        next.has_value = Boolean(input.value) || Boolean(field.has_value);
      }
    } else if (input.type === "checkbox") {
      next.value = String(input.checked);
    } else {
      next.value = input.value;
    }
    return next;
  });
  renderConfigStatus(liveFields);
}

async function startTask(action, payload) {
  try {
    const data = await api("/api/run", {
      method: "POST",
      body: JSON.stringify({ action, ...payload }),
    });
    toast("任务已启动");
    renderJob(data.job);
    switchTab("logs");
  } catch (error) {
    toast(error.message);
  }
}

async function cancelJob() {
  try {
    const data = await api("/api/job/cancel", { method: "POST", body: "{}" });
    toast(data.cancelled ? "已请求停止任务" : "当前没有可停止任务");
  } catch (error) {
    toast(error.message);
  }
}

async function pollJob() {
  try {
    const data = await api("/api/job");
    renderJob(data.job);
  } catch {
    // keep the UI quiet during server shutdown
  }
}

function renderJob(job) {
  const pill = $("jobPill");
  pill.className = "job-pill";
  const active = Boolean(job && ["pending", "running", "cancelling"].includes(job.status));
  setButtonsDisabled(active);
  $("btnCancelJob").disabled = !active;

  if (!job) {
    pill.textContent = "空闲";
    $("jobMeta").textContent = "暂无任务";
    return;
  }

  const statusLabel = {
    pending: "等待中",
    running: "运行中",
    cancelling: "停止中",
    cancelled: "已停止",
    completed: "已完成",
    failed: "失败",
  }[job.status] || job.status;
  pill.textContent = statusLabel;
  pill.classList.toggle("running", ["pending", "running", "cancelling"].includes(job.status));
  pill.classList.toggle("failed", job.status === "failed");

  $("jobMeta").textContent = [
    `动作：${job.action}`,
    `开始：${job.started_at || "-"}`,
    `结束：${job.ended_at || "-"}`,
    `退出码：${job.returncode ?? "-"}`,
  ].join("  /  ");
  $("logOutput").textContent = job.output && job.output.length
    ? job.output.join("\n")
    : "任务已启动，等待输出...";

  if (job.status === "completed") {
    refreshResults();
  }
}

function setButtonsDisabled(disabled) {
  [
    "btnGenerate",
    "btnRunFull",
    "btnBacktest",
    "btnRefine",
  ].forEach((id) => {
    $(id).disabled = disabled;
  });
}

async function refreshResults() {
  try {
    const data = await api("/api/results");
    renderMetrics(data.stats || {});
    renderSubmittable(data.submittable || []);
    renderRecent(data.recent || []);
  } catch (error) {
    toast(error.message);
  }
}

function renderMetrics(stats) {
  const items = [
    ["Generated", stats.generated || 0],
    ["Backtesting", stats.backtesting || 0],
    ["Evaluated", stats.evaluated || 0],
    ["High Quality", stats.high_quality || 0],
    ["Failed", stats.failed || 0],
    ["Fitness >= 1", stats.high_quality_count || 0],
  ];
  $("metrics").innerHTML = items
    .map(([label, value]) => `<div class="metric"><span>${label}</span><strong>${value}</strong></div>`)
    .join("");
}

function renderSubmittable(rows) {
  $("submittableRows").innerHTML = rows.length ? rows.map((row) => `
    <tr>
      <td>${row.alpha_id ?? ""}</td>
      <td>${fmt(row.fitness)}</td>
      <td>${fmt(row.sharpe)}</td>
      <td>${fmt(row.turnover)}</td>
      <td>${grade(row.grade)}</td>
      <td class="expr">${escapeHtml(row.expression || "")}</td>
    </tr>
  `).join("") : `<tr><td colspan="6">暂无可提交候选。</td></tr>`;
}

function renderRecent(rows) {
  $("recentRows").innerHTML = rows.length ? rows.map((row) => `
    <tr>
      <td>${row.id ?? ""}</td>
      <td>${escapeHtml(row.status || "")}</td>
      <td>${fmt(row.fitness)}</td>
      <td>${fmt(row.sharpe)}</td>
      <td>${fmt(row.turnover)}</td>
      <td>${grade(row.grade)}</td>
      <td class="expr">${escapeHtml(row.expression || "")}</td>
    </tr>
  `).join("") : `<tr><td colspan="7">暂无 Alpha 记录。</td></tr>`;
}

function switchTab(tab) {
  const button = document.querySelector(`.nav-btn[data-tab="${tab}"]`);
  if (button) {
    button.click();
  }
}

function grade(value) {
  const label = value || "-";
  const cls = String(label).toLowerCase();
  return `<span class="grade ${cls}">${escapeHtml(label)}</span>`;
}

function fmt(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "-";
  }
  return Number(value).toFixed(3);
}

function toast(message) {
  const box = $("toast");
  box.textContent = message;
  box.classList.add("show");
  window.setTimeout(() => box.classList.remove("show"), 2600);
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}
