let jobs = [];
let selectedId = null;

const $ = (sel) => document.querySelector(sel);
const jobList = $("#jobList");
const jobDetail = $("#jobDetail");
const toast = $("#toast");

function showToast(msg, type = "success") {
  toast.textContent = msg;
  toast.className = `toast ${type}`;
  setTimeout(() => toast.classList.add("hidden"), 4000);
}

function scoreClass(score) {
  if (score >= 80) return "score-high";
  if (score >= 60) return "score-mid";
  return "score-low";
}

async function loadStats() {
  const res = await fetch("/api/stats");
  const s = await res.json();
  $("#stats").innerHTML = `
    <div class="stat-card"><strong>${s.total}</strong><span>Total jobs</span></div>
    <div class="stat-card"><strong>${s.with_description}</strong><span>With description</span></div>
    <div class="stat-card"><strong>${s.recommended}</strong><span>Recommended</span></div>
    <div class="stat-card"><strong>${s.applied}</strong><span>Applied</span></div>
  `;
}

async function loadJobs() {
  const params = new URLSearchParams();
  const search = $("#search").value.trim();
  const portal = $("#portal").value;
  const applied = $("#applied").value;
  const minScore = $("#minScore").value;

  if (search) params.set("search", search);
  if (portal) params.set("portal", portal);
  if (applied) params.set("applied", applied);
  if (minScore) params.set("min_score", minScore);

  jobList.innerHTML = '<p class="muted" style="padding:1rem">Loading…</p>';
  const res = await fetch(`/api/jobs?${params}`);
  jobs = await res.json();
  renderList();
  if (selectedId) renderDetail(selectedId);
}

function renderList() {
  if (!jobs.length) {
    jobList.innerHTML = '<p class="muted" style="padding:1rem">No jobs found. Run <code>python main.py scan</code> first.</p>';
    return;
  }
  jobList.innerHTML = jobs.map((j) => `
    <div class="job-card ${j.id === selectedId ? "active" : ""}" data-id="${j.id}">
      <h3>${esc(j.title)}</h3>
      <div class="meta">
        <span class="badge ${scoreClass(j.match_score)}">${j.match_score}</span>
        <span class="badge portal">${j.portal}</span>
        ${j.is_applied ? '<span class="badge applied">Applied</span>' : ""}
        ${!j.has_description ? '<span class="badge" style="color:var(--warn)">No desc</span>' : ""}
      </div>
      <div class="meta">${esc(j.company)} · ${esc(j.location || "—")}</div>
    </div>
  `).join("");

  jobList.querySelectorAll(".job-card").forEach((el) => {
    el.addEventListener("click", () => {
      selectedId = el.dataset.id;
      renderList();
      renderDetail(selectedId);
    });
  });
}

async function renderDetail(id) {
  const res = await fetch(`/api/jobs/${encodeURIComponent(id)}`);
  if (!res.ok) return;
  const j = await res.json();

  const applied = j.is_applied;
  jobDetail.innerHTML = `
    <div class="detail-header">
      <h2>${esc(j.title)}</h2>
      <div class="meta">${esc(j.company)} · ${esc(j.location || "—")} · <span class="badge portal">${j.portal}</span></div>
      <div class="meta">Match score: <span class="badge ${scoreClass(j.match_score)}">${j.match_score}</span>
        ${j.recommended ? " · Recommended" : ""}
        ${applied ? ` · <span class="badge applied">Applied (${esc(j.apply_status)})</span>` : ""}
      </div>
    </div>

    <div class="detail-actions">
      <a href="${esc(j.url)}" target="_blank" rel="noopener">Open on portal</a>
      <button class="primary" id="applyBtn" ${applied ? "disabled" : ""}>${applied ? "Already applied" : "Apply"}</button>
      <button class="secondary" id="markBtn" ${applied ? "disabled" : ""}>Mark as applied</button>
    </div>

    <div class="section">
      <h3>Match reasoning</h3>
      <p>${esc(j.reasoning || "No reasoning stored.")}</p>
    </div>

    ${j.skill_overlap?.length ? `
    <div class="section">
      <h3>Skill overlap</h3>
      <div class="tags">${j.skill_overlap.map((s) => `<span class="tag">${esc(s)}</span>`).join("")}</div>
    </div>` : ""}

    ${j.gaps?.length ? `
    <div class="section">
      <h3>Gaps</h3>
      <div class="tags">${j.gaps.map((s) => `<span class="tag">${esc(s)}</span>`).join("")}</div>
    </div>` : ""}

    <div class="section">
      <h3>Job description</h3>
      <div class="description">${j.description ? esc(j.description) : '<span class="muted">Description not downloaded. Re-run scan to fetch.</span>'}</div>
    </div>

    ${j.apply_message ? `
    <div class="section">
      <h3>Application note</h3>
      <p>${esc(j.apply_message)}</p>
      ${j.applied_at ? `<p class="muted">Applied at: ${esc(j.applied_at)}</p>` : ""}
    </div>` : ""}
  `;

  if (!applied) {
    $("#applyBtn")?.addEventListener("click", () => applyJob(id));
    $("#markBtn")?.addEventListener("click", () => markApplied(id));
  }
}

async function applyJob(id) {
  $("#applyBtn").disabled = true;
  $("#applyBtn").textContent = "Applying…";
  try {
    const res = await fetch(`/api/jobs/${encodeURIComponent(id)}/apply`, { method: "POST" });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Apply failed");
    showToast(`Apply: ${data.status} — ${data.message}`);
    await loadStats();
    await loadJobs();
    renderDetail(id);
  } catch (e) {
    showToast(e.message, "error");
    $("#applyBtn").disabled = false;
    $("#applyBtn").textContent = "Apply";
  }
}

async function markApplied(id) {
  const res = await fetch(`/api/jobs/${encodeURIComponent(id)}/mark-applied`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message: "Marked applied manually in UI" }),
  });
  const data = await res.json();
  showToast(data.message || "Marked applied");
  await loadStats();
  await loadJobs();
  renderDetail(id);
}

function esc(s) {
  const d = document.createElement("div");
  d.textContent = s ?? "";
  return d.innerHTML;
}

$("#refreshBtn").addEventListener("click", () => { loadStats(); loadJobs(); });
$("#search").addEventListener("input", debounce(loadJobs, 300));
$("#portal").addEventListener("change", loadJobs);
$("#applied").addEventListener("change", loadJobs);
$("#minScore").addEventListener("change", loadJobs);

function debounce(fn, ms) {
  let t;
  return (...args) => { clearTimeout(t); t = setTimeout(() => fn(...args), ms); };
}

loadStats();
loadJobs();
