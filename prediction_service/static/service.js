"use strict";

const comparisonData = document.getElementById("model-comparison-data");
if (comparisonData && window.Chart && document.getElementById("model-comparison-chart")) {
  const data = JSON.parse(comparisonData.textContent);
  const colors = ["#2563eb", "#94a3b8", "#c58935", "#8766cc"];
  new Chart(document.getElementById("model-comparison-chart"), {
    type: "line",
    data: {
      labels: data.chart.map(row => String(row.date).slice(4, 6) + "/" + String(row.date).slice(6, 8)),
      datasets: data.keys.map((key, index) => ({
        label: data.models[index], data: data.chart.map(row => row[key] * 100),
        borderColor: colors[index], backgroundColor: colors[index],
        borderWidth: index === 0 ? 2.4 : 1.6, pointRadius: 0, pointHitRadius: 10,
      })),
    },
    options: {
      responsive: true, maintainAspectRatio: false, animation: false,
      interaction: { mode: "index", intersect: false },
      plugins: {
        legend: { position: "bottom", labels: { usePointStyle: true, boxWidth: 7, boxHeight: 7, padding: 18, font: { size: 11 } } },
        tooltip: { callbacks: { label: item => item.dataset.label + ": " + item.parsed.y.toFixed(2) + "%" } },
      },
      scales: {
        x: { grid: { display: false }, ticks: { maxTicksLimit: 9, maxRotation: 0, color: "#7b818d" } },
        y: { min: 0, max: 100, grid: { color: "#edf0f5" }, ticks: { callback: value => value + "%", maxTicksLimit: 6, color: "#7b818d" } },
      },
    },
  });
}

document.querySelectorAll("[data-model-table]").forEach(section => {
  const rows = Array.from(section.querySelectorAll("tbody tr"));
  const filter = section.querySelector("[data-disagreements]");
  const prev = section.querySelector("[data-comparison-prev]");
  const next = section.querySelector("[data-comparison-next]");
  let page = 0;
  function render() {
    const selected = rows.filter(row => !filter.checked || row.dataset.disagreement === "true");
    const pages = Math.max(1, Math.ceil(selected.length / 20));
    page = Math.min(page, pages - 1);
    rows.forEach(row => { row.hidden = true; });
    selected.slice(page * 20, (page + 1) * 20).forEach(row => { row.hidden = false; });
    section.querySelector("[data-comparison-empty]").hidden = selected.length > 0;
    section.querySelector("[data-comparison-pagination]").hidden = selected.length === 0;
    section.querySelector("[data-comparison-count]").textContent = "共 " + selected.length + " 个交易日";
    section.querySelector("[data-comparison-page]").textContent = (page + 1) + " / " + pages;
    prev.disabled = page === 0; next.disabled = page === pages - 1;
  }
  filter.addEventListener("change", () => { page = 0; render(); });
  prev.addEventListener("click", () => { page -= 1; render(); });
  next.addEventListener("click", () => { page += 1; render(); });
  render();
});

function showToast(message) {
  const toast = document.getElementById("toast");
  toast.textContent = message;
  toast.hidden = false;
  window.setTimeout(() => {
    toast.hidden = true;
  }, 3000);
}

if (window.lucide) window.lucide.createIcons();

document.querySelectorAll(".window-control").forEach((form) => {
  form.addEventListener("submit", (event) => {
    const input = form.querySelector("#custom-days");
    const preset =
      event.submitter?.name === "days" ? event.submitter.value : null;
    if (!preset && !input.reportValidity()) {
      event.preventDefault();
      return;
    }
    const target = new URL(form.action);
    target.searchParams.set("view", form.querySelector('[name="view"]').value);
    target.searchParams.set("days", preset || input.value);
    event.preventDefault();
    window.location.assign(target);
  });
});

document
  .querySelector("[data-toggle-password]")
  ?.addEventListener("click", (event) => {
    const input = document.getElementById("password");
    const visible = input.type === "password";
    input.type = visible ? "text" : "password";
    event.currentTarget.setAttribute("aria-pressed", String(visible));
    event.currentTarget.setAttribute(
      "aria-label",
      visible ? "隐藏密码" : "显示密码",
    );
    event.currentTarget.title = visible ? "隐藏密码" : "显示密码";
  });

document
  .querySelector("[data-copy-api]")
  ?.addEventListener("click", async () => {
    const address = new URL(
      "/api/v1/sh000001/latest.json",
      window.location.origin,
    ).href;
    try {
      await navigator.clipboard.writeText(address);
      showToast("API 地址已复制");
    } catch {
      showToast("无法访问剪贴板：" + address);
    }
  });

document.querySelectorAll("[data-submit-job]").forEach((form) => {
  form.addEventListener("submit", () => {
    form.querySelectorAll("button").forEach((button) => {
      button.disabled = true;
    });
  });
});

const chartData = document.getElementById("chart-data");
if (chartData && window.Chart) {
  const records = JSON.parse(chartData.textContent);
  const dateLabel = (value) => {
    const text = String(value).replace(/-/g, "");
    return text.slice(4, 6) + "/" + text.slice(6, 8);
  };
  new Chart(document.getElementById("returns-chart"), {
    type: "line",
    data: {
      labels: records.map((row) =>
        dateLabel(row["预测目标交易日"] || row["信号日期"]),
      ),
      datasets: [
        {
          label: "预测",
          data: records.map((row) => row["预测次日涨跌幅"] * 100),
          borderColor: "#4581df",
          backgroundColor: "#4581df",
          borderWidth: 1.8,
          pointRadius: records.length <= 10 ? 2.5 : 0,
          pointHitRadius: 12,
          tension: 0.15,
        },
        {
          label: "实际",
          data: records.map((row) => row["次日实际涨跌幅"] * 100),
          borderColor: "#9aa5b5",
          backgroundColor: "#9aa5b5",
          borderWidth: 1.4,
          pointRadius: records.length <= 10 ? 2.5 : 0,
          pointHitRadius: 12,
          tension: 0.1,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      interaction: { mode: "index", intersect: false },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: "#293341",
          padding: 11,
          callbacks: {
            label: (item) =>
              item.dataset.label + ": " + item.parsed.y.toFixed(3) + "%",
          },
        },
      },
      scales: {
        x: {
          grid: { display: false },
          border: { display: false },
          ticks: {
            color: "#9aa1ad",
            maxTicksLimit: 7,
            maxRotation: 0,
            font: { size: 10 },
          },
        },
        y: {
          border: { display: false },
          grid: { color: "#edf0f5" },
          ticks: {
            color: "#9aa1ad",
            maxTicksLimit: 5,
            font: { size: 10 },
            callback: (value) => Number(value).toFixed(1) + "%",
          },
        },
      },
    },
  });
}

document.querySelectorAll("[data-results-table]").forEach((section) => {
  const pagination = section.querySelector("[data-pagination]");
  if (!pagination) return;
  pagination.hidden = false;
  const rows = Array.from(section.querySelectorAll("tbody tr"));
  const direction = section.querySelector("[data-direction-filter]");
  const outcome = section.querySelector("[data-outcome-filter]");
  let page = 0;
  const size = 20;
  function render() {
    const selected = rows.filter(
      (row) =>
        (!direction.value || row.dataset.direction === direction.value) &&
        (!outcome.value || row.dataset.outcome === outcome.value),
    );
    const pages = Math.max(1, Math.ceil(selected.length / size));
    page = Math.min(page, pages - 1);
    rows.forEach((row) => {
      row.hidden = true;
    });
    selected.slice(page * size, (page + 1) * size).forEach((row) => {
      row.hidden = false;
    });
    section.querySelector("[data-page]").textContent = `${page + 1} / ${pages}`;
    section.querySelector("[data-result-count]").textContent =
      `${selected.length} 条记录`;
    section.querySelector("[data-prev]").disabled = page === 0;
    section.querySelector("[data-next]").disabled = page + 1 >= pages;
    section.querySelector("[data-no-results]").hidden = selected.length > 0;
  }
  [direction, outcome].forEach((control) =>
    control.addEventListener("change", () => {
      page = 0;
      render();
    }),
  );
  section.querySelector("[data-prev]").addEventListener("click", () => {
    page -= 1;
    render();
  });
  section.querySelector("[data-next]").addEventListener("click", () => {
    page += 1;
    render();
  });
  render();
});

const jobState = document.getElementById("job-state");
if (jobState) {
  const state = JSON.parse(jobState.textContent);
  const endpoints = state.jobs.map((id) => "/admin/jobs/" + id);
  if (state.shadow) endpoints.push("/admin/shadow-runs/" + state.shadow);
  let failures = 0;
  async function poll() {
    if (document.hidden) {
      window.setTimeout(poll, 5000);
      return;
    }
    try {
      const statuses = await Promise.all(
        endpoints.map(async (url) => {
          const response = await fetch(url, {
            headers: { Accept: "application/json" },
          });
          if (!response.ok || response.redirected)
            throw new Error("status unavailable");
          return response.json();
        }),
      );
      failures = 0;
      if (statuses.some((job) => !["queued", "running"].includes(job.status))) {
        if (["INPUT", "SELECT"].includes(document.activeElement?.tagName)) {
          window.setTimeout(poll, 5000);
          return;
        }
        window.location.reload();
        return;
      }
    } catch {
      failures += 1;
      if (failures === 3) showToast("任务状态暂不可用，稍后重试");
    }
    if (failures < 12)
      window.setTimeout(poll, Math.min(30000, 5000 * (failures + 1)));
  }
  if (endpoints.length) window.setTimeout(poll, 5000);
}
