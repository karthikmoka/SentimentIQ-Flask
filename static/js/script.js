/* ═══════════════════════════════════════════════════════════
   SentimentIQ — Main JavaScript
   ═══════════════════════════════════════════════════════════ */

"use strict";

/* ── Helpers ──────────────────────────────────────────────── */
function sentimentColor(s) {
  return s === "Positive" ? "#16A34A" : s === "Negative" ? "#DC2626" : "#EAB308";
}
function sentimentIcon(s) {
  return s === "Positive"
    ? '<i class="bi bi-emoji-smile-fill" style="color:#16A34A"></i>'
    : s === "Negative"
    ? '<i class="bi bi-emoji-frown-fill" style="color:#DC2626"></i>'
    : '<i class="bi bi-emoji-neutral-fill" style="color:#EAB308"></i>';
}
function badgeClass(s) {
  return s === "Positive" ? "siq-badge-pos" : s === "Negative" ? "siq-badge-neg" : "siq-badge-neu";
}
function pct(v) { return (v * 100).toFixed(1) + "%"; }

/* ── Dashboard Charts ─────────────────────────────────────── */
function initDashboardCharts(analytics) {
  const counts = analytics.counts || { Positive: 0, Negative: 0, Neutral: 0 };
  const labels = ["Positive", "Negative", "Neutral"];
  const values = [counts.Positive || 0, counts.Negative || 0, counts.Neutral || 0];
  const colors = ["#16A34A", "#DC2626", "#EAB308"];

  /* Pie */
  const pieCtx = document.getElementById("pieChart");
  if (pieCtx) {
    new Chart(pieCtx, {
      type: "doughnut",
      data: {
        labels,
        datasets: [{
          data: values,
          backgroundColor: colors,
          borderColor: "#fff",
          borderWidth: 3,
          hoverOffset: 8,
        }],
      },
      options: {
        responsive: false,
        cutout: "60%",
        plugins: {
          legend: {
            position: "bottom",
            labels: { font: { size: 12, family: "Inter" }, usePointStyle: true, padding: 12 },
          },
          tooltip: {
            callbacks: {
              label: ctx => ` ${ctx.label}: ${ctx.raw} reviews`,
            },
          },
        },
      },
    });
  }

  /* Bar */
  const barCtx = document.getElementById("barChart");
  if (barCtx) {
    new Chart(barCtx, {
      type: "bar",
      data: {
        labels,
        datasets: [{
          label: "Count",
          data: values,
          backgroundColor: colors.map(c => c + "CC"),
          borderColor: colors,
          borderWidth: 2,
          borderRadius: 8,
        }],
      },
      options: {
        responsive: true,
        plugins: {
          legend: { display: false },
          tooltip: { callbacks: { label: ctx => ` ${ctx.raw} reviews` } },
        },
        scales: {
          y: {
            beginAtZero: true,
            grid: { color: "rgba(0,0,0,.05)" },
            ticks: { font: { family: "Inter" } },
          },
          x: {
            grid: { display: false },
            ticks: { font: { family: "Inter", weight: 600 } },
          },
        },
      },
    });
  }
}

/* ── Analyze Page ─────────────────────────────────────────── */
function initAnalyzePage() {
  const textarea       = document.getElementById("reviewText");
  const analyzeBtn     = document.getElementById("analyzeBtn");
  const clearBtn       = document.getElementById("clearBtn");
  const charCount      = document.getElementById("charCount");
  const resultCard     = document.getElementById("resultCard");
  const resultPlaceholder = document.getElementById("resultPlaceholder");

  let sessionHistory = [];
  let sessionCounter = 0;

  /* Character counter */
  if (textarea) {
    textarea.addEventListener("input", () => {
      charCount.textContent = textarea.value.length;
    });
  }

  /* Clear */
  if (clearBtn) {
    clearBtn.addEventListener("click", () => {
      textarea.value = "";
      charCount.textContent = "0";
    });
  }

  /* Quick example buttons */
  document.querySelectorAll(".example-text").forEach(btn => {
    btn.addEventListener("click", () => {
      textarea.value = btn.dataset.text;
      charCount.textContent = textarea.value.length;
      textarea.focus();
    });
  });

  /* Analyze */
  if (analyzeBtn) {
    analyzeBtn.addEventListener("click", analyzeText);
    textarea && textarea.addEventListener("keydown", e => {
      if ((e.ctrlKey || e.metaKey) && e.key === "Enter") analyzeText();
    });
  }

  function setLoading(loading) {
    const txt = analyzeBtn.querySelector(".siq-btn-text");
    const spin = analyzeBtn.querySelector(".siq-btn-loading");
    analyzeBtn.disabled = loading;
    txt.classList.toggle("d-none", loading);
    spin.classList.toggle("d-none", !loading);
  }

  async function analyzeText() {
    const text = textarea.value.trim();
    if (!text) {
      textarea.classList.add("is-invalid");
      textarea.focus();
      setTimeout(() => textarea.classList.remove("is-invalid"), 2000);
      return;
    }

    setLoading(true);

    try {
      const res  = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Prediction failed.");
      showResult(data, text);
      addToSession(data);
    } catch (err) {
      showError(err.message);
    } finally {
      setLoading(false);
    }
  }

  function showResult(data) {
    resultPlaceholder.classList.add("d-none");
    resultCard.classList.remove("d-none");

    const { sentiment, confidence, probabilities, original_text } = data;
    const color = sentimentColor(sentiment);

    document.getElementById("sentimentIcon").innerHTML = `<span style="font-size:4rem">${sentimentIcon(sentiment)}</span>`;
    document.getElementById("sentimentBadge").innerHTML =
      `<span class="${badgeClass(sentiment)}" style="font-size:1.4rem;padding:.5rem 1.5rem">${sentiment}</span>`;
    document.getElementById("confidenceVal").textContent = pct(confidence);
    document.getElementById("submittedText").textContent = original_text.slice(0, 200) + (original_text.length > 200 ? "…" : "");

    /* Probability bars */
    const probDiv = document.getElementById("probBars");
    probDiv.innerHTML = "";
    const order = ["Positive", "Negative", "Neutral"];
    order.forEach(label => {
      const val = probabilities[label] || 0;
      const fillColor = sentimentColor(label);
      probDiv.innerHTML += `
        <div class="siq-prob-row d-flex align-items-center gap-2">
          <span class="siq-prob-label">${label}</span>
          <div class="siq-prob-track flex-grow-1">
            <div class="siq-prob-fill" style="width:${(val*100).toFixed(1)}%;background:${fillColor};"></div>
          </div>
          <small class="text-muted" style="min-width:42px;text-align:right">${pct(val)}</small>
        </div>`;
    });
  }

  function showError(msg) {
    resultPlaceholder.classList.remove("d-none");
    resultCard.classList.add("d-none");
    resultPlaceholder.innerHTML = `
      <div class="siq-placeholder-icon mb-3" style="background:rgba(220,38,38,.1);color:#DC2626">
        <i class="bi bi-exclamation-triangle"></i></div>
      <h5 class="fw-600 text-danger">Error</h5>
      <p class="text-muted px-3">${msg}</p>`;
  }

  function addToSession(data) {
    sessionCounter++;
    sessionHistory.unshift({ ...data, idx: sessionCounter });

    const empty = document.getElementById("sessionEmpty");
    const table = document.getElementById("sessionTable");
    const tbody = document.getElementById("sessionTbody");
    const count = document.getElementById("sessionCount");

    empty.classList.add("d-none");
    table.classList.remove("d-none");
    count.textContent = `${sessionCounter} analyse${sessionCounter !== 1 ? "s" : ""}`;

    const row = document.createElement("tr");
    row.innerHTML = `
      <td>${sessionCounter}</td>
      <td class="siq-text-cell">${data.original_text.slice(0, 80)}${data.original_text.length > 80 ? "…" : ""}</td>
      <td><span class="${badgeClass(data.sentiment)}">${sentimentIcon(data.sentiment)} ${data.sentiment}</span></td>
      <td>${pct(data.confidence)}</td>`;
    tbody.prepend(row);
  }
}

/* ── Batch Page ───────────────────────────────────────────── */
function initBatchPage() {
  const fileInput   = document.getElementById("csvFile");
  const dropZone    = document.getElementById("dropZone");
  const fileInfo    = document.getElementById("fileInfo");
  const colSelector = document.getElementById("colSelector");
  const batchBtn    = document.getElementById("batchBtn");
  let   downloadFilename = null;

  /* Drop zone drag events */
  if (dropZone) {
    ["dragover", "dragenter"].forEach(e => dropZone.addEventListener(e, ev => {
      ev.preventDefault();
      dropZone.classList.add("drag-over");
    }));
    ["dragleave", "drop"].forEach(e => dropZone.addEventListener(e, ev => {
      if (e === "drop") { ev.preventDefault(); handleFile(ev.dataTransfer.files[0]); }
      dropZone.classList.remove("drag-over");
    }));
  }

  if (fileInput) {
    fileInput.addEventListener("change", () => handleFile(fileInput.files[0]));
  }

  function handleFile(file) {
    if (!file) return;
    if (!file.name.endsWith(".csv")) {
      alert("Please select a .csv file.");
      return;
    }
    document.getElementById("fileName").textContent = file.name;
    document.getElementById("fileSize").textContent = `(${(file.size / 1024).toFixed(1)} KB)`;
    fileInfo.classList.remove("d-none");
    colSelector.classList.remove("d-none");
    batchBtn.disabled = false;
  }

  if (batchBtn) {
    batchBtn.addEventListener("click", runBatch);
  }

  function setLoading(loading) {
    const txt  = batchBtn.querySelector(".siq-btn-text");
    const spin = batchBtn.querySelector(".siq-btn-loading");
    batchBtn.disabled = loading;
    txt.classList.toggle("d-none",  loading);
    spin.classList.toggle("d-none", !loading);
  }

  async function runBatch() {
    const file    = fileInput.files[0];
    const textCol = document.getElementById("textColumn").value.trim() || "text";
    if (!file) { alert("Please select a CSV file."); return; }

    setLoading(true);
    const formData = new FormData();
    formData.append("file", file);
    formData.append("text_column", textCol);

    try {
      const res  = await fetch("/batch-predict", { method: "POST", body: formData });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Batch prediction failed.");
      showBatchResults(data);
      downloadFilename = data.download_file;
    } catch (err) {
      alert("Error: " + err.message);
    } finally {
      setLoading(false);
    }
  }

  function showBatchResults(data) {
    document.getElementById("batchPlaceholder").classList.add("d-none");
    document.getElementById("batchSummary").classList.remove("d-none");
    document.getElementById("batchResultsWrap").classList.remove("d-none");

    /* Summary cards */
    const counts   = data.counts || {};
    const total    = data.total || 0;
    const cardDefs = [
      { label: "Total", val: total, icon: "bi-table", color: "#2563EB" },
      { label: "Positive", val: counts.Positive || 0, icon: "bi-emoji-smile-fill", color: "#16A34A" },
      { label: "Negative", val: counts.Negative || 0, icon: "bi-emoji-frown-fill", color: "#DC2626" },
      { label: "Neutral",  val: counts.Neutral  || 0, icon: "bi-emoji-neutral-fill", color: "#EAB308" },
    ];
    const summaryRow = document.getElementById("summaryCards");
    summaryRow.innerHTML = cardDefs.map(c => `
      <div class="col-6 col-md-3">
        <div class="siq-stat-card text-center" style="border-top:4px solid ${c.color}">
          <i class="bi ${c.icon} fs-4 mb-1" style="color:${c.color}"></i>
          <div class="siq-stat-val" style="color:${c.color}">${c.val}</div>
          <div class="siq-stat-lbl">${c.label}</div>
        </div>
      </div>`).join("");

    /* Download button */
    const dlBtn = document.getElementById("downloadBtn");
    if (dlBtn && data.download_file) {
      dlBtn.onclick = () => {
        window.location.href = `/download-results/${data.download_file}`;
      };
    }

    /* Reset button */
    const resetBtn = document.getElementById("resetBatch");
    if (resetBtn) {
      resetBtn.onclick = () => {
        location.reload();
      };
    }

    /* Preview table */
    const tbody = document.getElementById("batchTbody");
    tbody.innerHTML = "";
    (data.preview || []).forEach((row, i) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td>${i + 1}</td>
        <td class="siq-text-cell">${(row.text || "").slice(0, 90)}…</td>
        <td><span class="${badgeClass(row.sentiment)}">${sentimentIcon(row.sentiment)} ${row.sentiment}</span></td>
        <td>${row.confidence}</td>`;
      tbody.appendChild(tr);
    });
  }
}

/* ── Performance Page ─────────────────────────────────────── */
function initPerformancePage(perClass, labelOrder) {
  const ctx = document.getElementById("classChart");
  if (!ctx || !perClass) return;

  const labels    = labelOrder;
  const colors    = { Positive: "#16A34A", Negative: "#DC2626", Neutral: "#EAB308" };
  const precision = labels.map(l => +(perClass[l]?.precision * 100 || 0).toFixed(1));
  const recall    = labels.map(l => +(perClass[l]?.recall    * 100 || 0).toFixed(1));
  const f1        = labels.map(l => +(perClass[l]?.f1        * 100 || 0).toFixed(1));

  new Chart(ctx, {
    type: "bar",
    data: {
      labels,
      datasets: [
        { label: "Precision", data: precision, backgroundColor: "#2563EBCC", borderColor: "#2563EB", borderWidth: 2, borderRadius: 6 },
        { label: "Recall",    data: recall,    backgroundColor: "#7C3AEDCC", borderColor: "#7C3AED", borderWidth: 2, borderRadius: 6 },
        { label: "F1-Score",  data: f1,        backgroundColor: "#0891B2CC", borderColor: "#0891B2", borderWidth: 2, borderRadius: 6 },
      ],
    },
    options: {
      responsive: true,
      plugins: {
        legend: { position: "top", labels: { font: { family: "Inter" }, usePointStyle: true } },
        tooltip: { callbacks: { label: ctx => ` ${ctx.dataset.label}: ${ctx.raw}%` } },
      },
      scales: {
        y: {
          beginAtZero: true,
          max: 100,
          ticks: { callback: v => v + "%", font: { family: "Inter" } },
          grid: { color: "rgba(0,0,0,.05)" },
        },
        x: {
          grid: { display: false },
          ticks: { font: { family: "Inter", weight: 600 } },
        },
      },
    },
  });
}
