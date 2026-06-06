const state = {
  configFields: [],
  currentTab: "run",
  pollTimer: null,
  csrfToken: "",
  wikiTree: null,
  wikiUploading: false,
};
const CLEAR_SECRET_VALUE = "__clear_secret__";
const GLOBAL_MODEL_OPTIONS = {
  openai: ["gpt-5.5", "gpt-5.4", "gpt-5.4-mini", "gpt-5.3-codex"],
  kimi: ["kimi-k2.6"],
  deepseek: ["deepseek-chat", "deepseek-reasoner"],
};
const PROVIDER_MODEL_KEYS = {
  openai: "OPENAI_MODEL",
  kimi: "KIMI_MODEL",
  deepseek: "DEEPSEEK_MODEL",
};
const PROVIDER_SECRET_KEYS = {
  openai: "OPENAI_API_KEY",
  kimi: "KIMI_API_KEY",
  deepseek: "DEEPSEEK_API_KEY",
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
      if (state.currentTab === "wiki") {
        refreshWiki();
      }
    });
  });
}

function bindActions() {
  $("btnSaveConfig").addEventListener("click", saveConfig);
  $("btnRefreshResults").addEventListener("click", refreshResults);
  $("btnCancelJob").addEventListener("click", cancelJob);
  $("btnRefreshWiki").addEventListener("click", refreshWiki);
  bindWikiUpload();

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
  const isFormData = options.body instanceof FormData;
  const headers = { ...(options.headers || {}) };
  if (!isFormData && !headers["Content-Type"]) {
    headers["Content-Type"] = "application/json";
  }
  if (state.csrfToken && path !== "/api/meta") {
    headers["X-WQ-Agent-CSRF"] = state.csrfToken;
  }
  const response = await fetch(path, {
    ...options,
    headers,
  });
  const text = await response.text();
  let data = {};
  try {
    data = text ? JSON.parse(text) : {};
  } catch {
    data = { error: text || "请求失败" };
  }
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
  const visibleFields = visibleConfigFields(fields);
  const provider = currentProvider(fields);
  const modelFields = visibleFields.filter((field) => field.section === "模型");
  const generalModelFields = modelFields.filter((field) => !field.provider);
  const providerModelFields = modelFields.filter((field) => field.provider === provider);
  const nonModelFields = visibleFields.filter((field) => field.section !== "模型");
  const groups = [
    {
      key: "model-general",
      title: "通用模型参数",
      meta: "LLM Provider",
      fields: generalModelFields,
    },
    {
      key: "model-provider",
      title: providerConfigTitle(provider),
      meta: `当前供应商：${provider}`,
      fields: providerModelFields,
    },
  ];
  nonModelFields.forEach((field) => {
    let group = groups.find((item) => item.key === field.section);
    if (!group) {
      group = { key: field.section, title: field.section, meta: "", fields: [] };
      groups.push(group);
    }
    group.fields.push(field);
  });

  const form = $("configForm");
  form.innerHTML = "";
  groups
    .filter((group) => group.fields.length)
    .forEach((group) => {
      const sectionEl = document.createElement("section");
      sectionEl.className = `config-section config-section-${cssToken(group.key)}`;
      const headEl = document.createElement("div");
      headEl.className = "config-section-head";
      headEl.innerHTML = `
        <h4>${escapeHtml(group.title)}</h4>
        ${group.meta ? `<span>${escapeHtml(group.meta)}</span>` : ""}
      `;
      const fieldsEl = document.createElement("div");
      fieldsEl.className = "config-fields";
      group.fields.forEach((field) => fieldsEl.appendChild(configField(field)));
      sectionEl.appendChild(headEl);
      sectionEl.appendChild(fieldsEl);
      form.appendChild(sectionEl);
    });
}

function visibleConfigFields(fields) {
  const provider = currentProvider(fields);
  return fields.filter((field) => {
    if (field.ui_hidden) {
      return false;
    }
    if (field.provider && field.provider !== provider) {
      return false;
    }
    return true;
  });
}

function currentProvider(fields = state.configFields) {
  const provider = fields.find((field) => field.key === "LLM_PROVIDER")?.value || "openai";
  return String(provider).toLowerCase();
}

function providerConfigTitle(provider) {
  const labels = { openai: "OpenAI 参数", kimi: "Kimi 参数", deepseek: "DeepSeek 参数" };
  return labels[provider] || "供应商参数";
}

function configField(field) {
  const label = document.createElement("label");
  label.className = "config-field";
  if (field.provider) {
    label.dataset.provider = field.provider;
  }
  const labelText = document.createElement("span");
  labelText.className = "config-label";
  labelText.textContent = field.label;

  if (field.kind === "boolean") {
    label.classList.add("config-switch-field");
    const input = document.createElement("input");
    input.type = "checkbox";
    input.dataset.configKey = field.key;
    input.checked = String(field.value).toLowerCase() === "true";
    input.addEventListener("change", handleConfigInputChange);
    const switchEl = document.createElement("span");
    switchEl.className = "switch";
    switchEl.setAttribute("aria-hidden", "true");
    label.appendChild(labelText);
    label.appendChild(input);
    label.appendChild(switchEl);
    return label;
  }

  const { input, datalist } = createConfigInput(field);
  input.dataset.configKey = field.key;
  input.addEventListener(input.tagName === "SELECT" ? "change" : "input", handleConfigInputChange);

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
  if (datalist) {
    label.appendChild(datalist);
  }

  return label;
}

function createConfigInput(field) {
  if (field.kind === "select" && !field.allow_custom) {
    const select = document.createElement("select");
    const options = Array.isArray(field.options) ? [...field.options] : [];
    const current = field.value || "";
    if (current && !options.includes(current)) {
      options.push(current);
    }
    options.forEach((option) => {
      const optionEl = document.createElement("option");
      optionEl.value = option;
      optionEl.textContent = option || "使用供应商默认";
      select.appendChild(optionEl);
    });
    select.value = current;
    return { input: select, datalist: null };
  }

  const input = document.createElement("input");
  input.value = field.secret ? "" : field.value || "";
  input.placeholder = field.secret && field.has_value ? "已设置，留空保持不变" : "";
  input.type = field.secret ? "password" : field.kind === "number" ? "number" : "text";
  let datalist = null;
  if (field.kind === "select" && field.allow_custom) {
    const listId = `options-${field.key}`;
    input.setAttribute("list", listId);
    input.placeholder = field.value ? "" : "可输入任意模型名";
    datalist = document.createElement("datalist");
    datalist.id = listId;
    (field.options || []).forEach((option) => {
      if (!option) {
        return;
      }
      const optionEl = document.createElement("option");
      optionEl.value = option;
      datalist.appendChild(optionEl);
    });
  }
  return { input, datalist };
}

function handleConfigInputChange(event) {
  if (event.target.dataset.configKey === "LLM_PROVIDER") {
    state.configFields = currentConfigFieldsFromForm();
    renderConfig(state.configFields);
    renderConfigStatus(state.configFields);
    return;
  }
  refreshConfigStatusFromForm();
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

function cssToken(value) {
  return String(value).toLowerCase().replace(/[^a-z0-9_-]+/g, "-").replace(/^-+|-+$/g, "") || "group";
}

function renderConfigStatus(fields) {
  const byKey = Object.fromEntries(fields.map((field) => [field.key, field]));
  const provider = currentProvider(fields);
  const providerKey = PROVIDER_SECRET_KEYS[provider] || "OPENAI_API_KEY";
  const globalModel = byKey.LLM_MODEL?.value || "";
  const globalModelValid = !globalModel
    || provider === "openai"
    || (GLOBAL_MODEL_OPTIONS[provider] || []).includes(globalModel);
  const providerModelKey = PROVIDER_MODEL_KEYS[provider] || "OPENAI_MODEL";
  const model = globalModel
    ? (globalModelValid ? globalModel : "")
    : byKey[providerModelKey]?.value || "";
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
  renderConfigStatus(currentConfigFieldsFromForm());
}

function currentConfigFieldsFromForm() {
  return state.configFields.map((field) => {
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
        if (input.value) {
          next.value = input.value;
        }
      }
    } else if (input.type === "checkbox") {
      next.value = String(input.checked);
    } else {
      next.value = input.value;
    }
    return next;
  });
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

async function refreshWiki() {
  try {
    const data = await api("/api/wiki/tree");
    state.wikiTree = data;
    renderWiki(data);
  } catch (error) {
    toast(error.message);
  }
}

function renderWiki(data) {
  const roots = data.roots || {};
  const totalFiles = Object.values(roots).reduce((sum, root) => sum + (root.file_count || 0), 0);
  $("wikiSummary").textContent = `共 ${totalFiles} 个 Markdown 文件`;
  $("wikiRoots").innerHTML = "";
  Object.values(roots).forEach((root) => {
    $("wikiRoots").appendChild(renderWikiRoot(root));
  });
}

function renderWikiRoot(root) {
  const section = document.createElement("section");
  section.className = "wiki-root";
  section.innerHTML = `
    <div class="wiki-root-head">
      <div>
        <strong>${escapeHtml(root.label)}</strong>
        <span>${escapeHtml(root.path)}</span>
      </div>
      <em>${root.file_count || 0} 文件</em>
    </div>
  `;
  const tree = document.createElement("div");
  tree.className = "wiki-tree";
  if (!root.exists || !root.tree || !root.tree.total_files) {
    tree.innerHTML = `<p class="muted">暂无 Markdown 文件。</p>`;
  } else {
    tree.appendChild(renderWikiNode(root.key, root.tree, 0));
  }
  section.appendChild(tree);
  return section;
}

function renderWikiNode(rootKey, node, level) {
  if (node.type === "file") {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "wiki-file";
    button.dataset.wikiPath = `${rootKey}:${node.path}`;
    button.style.paddingLeft = `${12 + level * 14}px`;
    button.innerHTML = `
      <span>${escapeHtml(node.name)}</span>
      <small>${formatBytes(node.size)}</small>
    `;
    button.addEventListener("click", () => loadWikiFile(rootKey, node.path));
    return button;
  }

  const details = document.createElement("details");
  details.className = "wiki-dir";
  details.open = level < 1;
  const summary = document.createElement("summary");
  summary.style.paddingLeft = `${8 + level * 14}px`;
  summary.innerHTML = `
    <span>${escapeHtml(node.name || "root")}</span>
    <small>${node.file_count || 0} / ${node.total_files || 0}</small>
  `;
  details.appendChild(summary);
  (node.children || []).forEach((child) => {
    details.appendChild(renderWikiNode(rootKey, child, level + 1));
  });
  return details;
}

async function loadWikiFile(rootKey, path) {
  try {
    const data = await api(
      `/api/wiki/file?root=${encodeURIComponent(rootKey)}&path=${encodeURIComponent(path)}`
    );
    document.querySelectorAll(".wiki-file.active").forEach((item) => {
      item.classList.remove("active");
    });
    const active = Array.from(document.querySelectorAll(".wiki-file")).find(
      (item) => item.dataset.wikiPath === `${rootKey}:${path}`
    );
    if (active) {
      active.classList.add("active");
    }
    $("wikiFileTitle").textContent = data.name;
    $("wikiFileMeta").textContent = `${rootKey} / ${data.path} / ${formatBytes(data.size)} / ${data.modified_at}`;
    $("wikiPreview").textContent = data.content || "文件为空。";
  } catch (error) {
    toast(error.message);
  }
}

function bindWikiUpload() {
  const zone = $("wikiUploadZone");
  const input = $("wikiUploadInput");
  zone.addEventListener("click", (event) => {
    if (!state.wikiUploading && event.target !== input) {
      input.click();
    }
  });
  input.addEventListener("change", () => {
    uploadWikiFiles(input.files);
    input.value = "";
  });
  ["dragenter", "dragover"].forEach((eventName) => {
    zone.addEventListener(eventName, (event) => {
      event.preventDefault();
      zone.classList.add("dragging");
    });
  });
  ["dragleave", "drop"].forEach((eventName) => {
    zone.addEventListener(eventName, (event) => {
      event.preventDefault();
      zone.classList.remove("dragging");
    });
  });
  zone.addEventListener("drop", (event) => {
    uploadWikiFiles(event.dataTransfer.files);
  });
}

async function uploadWikiFiles(fileList) {
  const files = Array.from(fileList || []);
  if (!files.length) {
    return;
  }
  if (state.wikiUploading) {
    return;
  }
  const form = new FormData();
  form.append("root", "private");
  files.forEach((file) => form.append("files", file));
  setWikiUploadState(true, `正在导入 ${files.length} 个文件...`);
  try {
    const data = await api("/api/wiki/upload", {
      method: "POST",
      body: form,
    });
    const count = data.uploaded?.length || 0;
    toast(`已上传 ${count} 个文件`);
    state.wikiTree = data.tree;
    renderWiki(data.tree);
    if (data.uploaded?.[0]?.path) {
      loadWikiFile("private", data.uploaded[0].path);
    }
  } catch (error) {
    toast(error.message);
  } finally {
    setWikiUploadState(false);
  }
}

function setWikiUploadState(uploading, label = "Markdown / TXT / PDF / DOCX") {
  state.wikiUploading = uploading;
  $("wikiUploadZone").classList.toggle("uploading", uploading);
  $("wikiUploadInput").disabled = uploading;
  $("wikiUploadStatus").textContent = label;
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

function formatBytes(value) {
  const size = Number(value || 0);
  if (size < 1024) {
    return `${size} B`;
  }
  if (size < 1024 * 1024) {
    return `${(size / 1024).toFixed(1)} KB`;
  }
  return `${(size / (1024 * 1024)).toFixed(1)} MB`;
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
