async function fetchJson(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error("Request failed");
  return res.json();
}

async function loadDashboard() {
  const batches = await fetchJson("/batches");
  const drift = await fetchJson("/drift");
  const quality = await fetchJson("/quality");
  const model = await fetchJson("/model_health");

  document.getElementById("batch-count").textContent = batches.length;
  const reference = batches.find(b => b.is_reference);
  document.getElementById("reference-batch").textContent = reference ? `#${reference.id}` : "-";

  const latestMape = model.filter(m => m.metric === "mape").slice(-1)[0];
  document.getElementById("latest-mape").textContent = latestMape ? latestMape.value.toFixed(3) : "-";

  renderDriftChart(drift);
  renderQualityChart(quality);
  renderModelChart(model);
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

loadDashboard().catch(err => {
  console.error(err);
});
