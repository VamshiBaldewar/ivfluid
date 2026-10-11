const $ = (id) => document.getElementById(id);

async function post(url, body) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : "{}",
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || res.statusText);
  return data;
}

$("btn-start").onclick = async () => {
  await post("/api/pipeline/start", { source: $("source").value.trim() || "0" });
};

$("btn-stop").onclick = async () => {
  await post("/api/pipeline/stop");
};

$("btn-rec-start").onclick = async () => {
  const label = $("rec-label").value.trim();
  if (!label) {
    $("rec-status").textContent = "Label is required (e.g. 20dpm or 50).";
    return;
  }
  const data = await post("/api/record/start", {
    module: $("rec-module").value,
    label,
    kind: $("rec-kind").value,
    interval: Number($("rec-interval").value) || 1,
  });
  $("rec-status").textContent = "Recording → " + data.path;
};

$("btn-rec-stop").onclick = async () => {
  const data = await post("/api/record/stop");
  $("rec-status").textContent = data.active ? "Still active" : "Stopped.";
};

function tick() {
  fetch("/api/state")
    .then((r) => r.json())
    .then((s) => {
      $("fps").textContent = s.fps.toFixed(1);
      const cam = $("cam-status");
      cam.textContent = s.camera_ok ? "live" : "no frame";
      cam.className = "pill " + (s.camera_ok ? "ok" : "bad");

      $("m1-value").textContent = s.fluid.level_pct == null ? "—" : `${s.fluid.level_pct.toFixed(0)}%`;
      $("m1-detail").textContent = `${s.fluid.status}: ${s.fluid.detail}`;
      $("m2-value").textContent = s.drip.drops_per_min == null ? "—" : `${s.drip.drops_per_min.toFixed(1)} dpm`;
      $("m2-detail").textContent = `${s.drip.status}: ${s.drip.detail}`;
      $("m3-value").textContent = s.patient.distress_score == null ? "—" : s.patient.distress_score.toFixed(2);
      $("m3-detail").textContent = `${s.patient.status}: ${s.patient.detail}`;
      $("m5-value").textContent =
        s.eta.minutes_remaining == null ? "—" : `${s.eta.minutes_remaining.toFixed(1)} min`;
      $("m5-detail").textContent = `${s.eta.status}: ${s.eta.detail}`;

      const card = $("risk-card");
      card.className = "risk " + s.risk.tier;
      $("risk-tier").textContent = s.risk.tier;
      $("risk-score").textContent = `score ${s.risk.score.toFixed(2)}`;
      $("risk-reasons").innerHTML = s.risk.reasons.map((r) => `<li>${r}</li>`).join("");

      $("alerts").innerHTML = (s.alerts || [])
        .map((a) => `<li><strong>${a.tier}</strong> — ${(a.reasons || [])[0] || ""}</li>`)
        .join("");

      if (s.recording && s.recording.active) {
        $("rec-status").textContent = `Recording ${s.recording.kind} (${s.recording.module}/${s.recording.label})`;
      }

      const img = $("frame");
      const ph = $("no-frame");
      if (s.camera_ok) {
        img.src = "/api/frame?" + Date.now();
        img.classList.add("show");
        ph.style.display = "none";
      } else {
        img.classList.remove("show");
        ph.style.display = "grid";
      }
    })
    .catch(() => {});
}

setInterval(tick, 400);
tick();
