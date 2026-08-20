async function fetchJson(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error("Request failed");
  return res.json();
}

async function loadDashboard() {
  const [batches, drift, quality, model, alerts] = await Promise.all([
    fetchJson("/batches"),
    fetchJson("/drift"),
    fetchJson("/quality"),
    fetchJson("/model_health"),
    fetchJson("/alerts"),
  ]);

  document.getElementById("batch-count").textContent = batches.length;
  const reference = batches.find(b => b.is_reference);
  document.getElementById("reference-batch").textContent = reference ? `#${reference.id}` : "-";

  const latestMape = model.filter(m => m.metric === "mape").slice(-1)[0];
  document.getElementById("latest-mape").textContent = latestMape ? latestMape.value.toFixed(3) : "-";

  renderAlerts(alerts);
  renderDriftChart(drift);
  renderQualityChart(quality);
  renderModelChart(model);
}

// Alerts were written to the database on every batch and never displayed. The
// dashboard showed PSI bars and left it to you to know which ones crossed a
// threshold.
function renderAlerts(alerts) {
  const open = alerts.filter(a => a.status === "open");
  document.getElementById("alert-count").textContent = open.length;

  const body = document.querySelector("#alerts tbody");
  const empty = document.getElementById("alerts-empty");
  body.innerHTML = "";
  empty.hidden = alerts.length > 0;

  for (const alert of alerts.slice(0, 50)) {
    const row = body.insertRow();
    row.insertCell().textContent = `#${alert.batch_id}`;
    const severity = row.insertCell();
    severity.textContent = alert.severity;
    severity.className = `severity ${alert.severity}`;
    row.insertCell().textContent = alert.type;
    row.insertCell().textContent = alert.message;
  }
}

function renderDriftChart(drift) {
  const psi = drift.filter(d => d.metric === "psi" && d.comparison === "reference");
  const labels = psi.map(p => `${p.feature}#${p.batch_id}`);
  const data = psi.map(p => p.value);

  new Chart(document.getElementById("driftChart"), {
    type: "bar",
    data: { labels, datasets: [{ label: "PSI", data, backgroundColor: "#0b5d4d" }] },
    options: { responsive: true, scales: { y: { beginAtZero: true } } },
  });
}

function renderQualityChart(quality) {
  const missing = quality.filter(q => q.metric === "missing_rate");
  const labels = missing.map(m => `${m.feature}#${m.batch_id}`);
  const data = missing.map(m => m.value);

  new Chart(document.getElementById("qualityChart"), {
    type: "bar",
    data: { labels, datasets: [{ label: "Missing Rate", data, backgroundColor: "#b36a5e" }] },
    options: { responsive: true, scales: { y: { beginAtZero: true } } },
  });
}

function renderModelChart(model) {
  const mae = model.filter(m => m.metric === "mae");
  const mape = model.filter(m => m.metric === "mape");
  const labels = mae.map(m => `#${m.batch_id}`);
  const maeData = mae.map(m => m.value);
  const mapeData = mape.map(m => m.value);

  new Chart(document.getElementById("modelChart"), {
    type: "line",
    data: {
      labels,
      datasets: [
        { label: "MAE", data: maeData, borderColor: "#1c1b19" },
        { label: "MAPE", data: mapeData, borderColor: "#0b5d4d" },
      ],
    },
    options: { responsive: true },
  });
}

// A failed load used to log to the console and leave every card reading "-",
// which looks the same as a running system with no data in it.
loadDashboard().catch(err => {
  console.error(err);
  const banner = document.createElement("p");
  banner.className = "error";
  banner.textContent = `Could not load the dashboard: ${err.message}`;
  document.querySelector("header").append(banner);
});
