"use strict";

const TIERS = ["Normal", "Uncommon", "Rare", "Epic", "Legendary"];
const ICON = (name) => `/static/icons/${name}.png`;
const qualityIcon = (i) => ICON(`quality_${TIERS[i].toLowerCase()}`);
const moduleIcon = (kind, level) => ICON(`${kind}_module${level > 1 ? `_${level}` : ""}`);

let GAME = null;

// ---------- Small helpers ----------

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v;
    else if (k === "html") node.innerHTML = v;
    else node.setAttribute(k, v);
  }
  node.append(...children);
  return node;
}

function fmt(x, digits = 2) {
  if (!isFinite(x)) return "–";
  if (x !== 0 && Math.abs(x) < 0.01) return x.toExponential(1);
  return x.toLocaleString(undefined, { minimumFractionDigits: digits, maximumFractionDigits: digits });
}
const pct = (x, digits = 1) => `${fmt(x * 100, digits)}%`;

function tierName(i) {
  return `<span class="tier-name q${i}"><img src="${qualityIcon(i)}" alt="">${TIERS[i]}</span>`;
}

// ---------- Custom widgets ----------

function buildQualityPicker(span, initial = 0) {
  const allowNone = span.dataset.allowNone === "true";
  span.innerHTML = "";
  const options = allowNone ? [null, 0, 1, 2, 3, 4] : [0, 1, 2, 3, 4];
  for (const q of options) {
    const btn = el("button", { type: "button", title: q === null ? "None (never extract)" : TIERS[q] });
    if (q === null) {
      btn.classList.add("none");
      btn.textContent = "—";
    } else {
      btn.append(el("img", { src: qualityIcon(q), alt: TIERS[q] }));
    }
    btn.addEventListener("click", () => setQuality(span, q));
    btn.dataset.q = q === null ? "" : q;
    span.append(btn);
  }
  setQuality(span, initial);
}

function setQuality(span, q) {
  span.dataset.value = q === null ? "" : q;
  for (const btn of span.children) btn.classList.toggle("selected", btn.dataset.q === span.dataset.value);
  span.dispatchEvent(new Event("change", { bubbles: true }));
}

const getQuality = (span) => (span.dataset.value === "" ? null : Number(span.dataset.value));

function buildModulePicker(span, level = 3, quality = 0) {
  const kind = span.dataset.kind;
  span.innerHTML = "";
  const img = el("img", { class: "module-icon", alt: "" });
  const select = el("select");
  for (const l of [1, 2, 3]) select.append(el("option", { value: l }, `${kind === "quality" ? "Quality" : "Productivity"} ${l}`));
  select.value = level;
  const qp = el("span", { class: "quality-picker" });
  const updateIcon = () => (img.src = moduleIcon(kind, Number(select.value)));
  select.addEventListener("change", updateIcon);
  span.append(img, select, qp);
  buildQualityPicker(qp, quality);
  updateIcon();
}

function setModule(span, level, quality) {
  const select = $("select", span);
  select.value = level;
  select.dispatchEvent(new Event("change"));
  setQuality($(".quality-picker", span), quality);
}

const getModule = (span) => ({ level: Number($("select", span).value), quality: getQuality($(".quality-picker", span)) });

function buildTierInputs(container, values = [0, 0, 0, 0, 0]) {
  container.innerHTML = "";
  TIERS.forEach((name, i) => {
    container.append(
      el("div", { class: "tier-cell" },
        el("img", { src: qualityIcon(i), alt: name, title: name }),
        el("input", { type: "number", min: "0", step: "any", value: values[i], "data-tier": i, "aria-label": name }),
      ),
    );
  });
}

const getTierInputs = (container) => $$("input", container).map((i) => Number(i.value) || 0);
const setTierInputs = (container, values) => $$("input", container).forEach((inp, i) => (inp.value = values[i]));

// ---------- Form setup ----------

function initForms() {
  $$(".quality-picker[data-name]").forEach((s) => buildQualityPicker(s, s.dataset.allowNone === "true" ? null : 0));
  $$(".module-picker").forEach((s) => buildModulePicker(s));
  $$("[data-tier-inputs]").forEach((c) => buildTierInputs(c));

  // Machine dropdown
  const machineSelect = $("#form-assembler select[name=machine]");
  for (const [key, m] of Object.entries(GAME.machines)) machineSelect.append(el("option", { value: key }, m.name));
  machineSelect.value = "assembling-machine-3";
  machineSelect.addEventListener("change", updateMachine);

  // Recipe dropdown
  const form = $("#form-assembler");
  for (const [key, r] of Object.entries(GAME.recipes)) form.recipe.append(el("option", { value: key }, r.name));
  form.recipe.addEventListener("change", updateRecipe);
  form.ingredient.addEventListener("change", updateIngredient);

  const tbody = $("#tier-setup");
  TIERS.forEach((_, i) => {
    tbody.append(
      el("tr", {},
        el("td", { html: tierName(i) }),
        el("td", {}, el("input", { type: "number", min: "1", value: "1", name: `count_${i}` })),
        el("td", {}, el("input", { type: "number", min: "0", value: "0", name: `prod_${i}` })),
        el("td", {}, el("input", { type: "number", min: "0", value: "4", name: `qual_${i}` })),
      ),
    );
  });
  tbody.addEventListener("input", checkSlots);
  updateMachine();

  // Tabs
  $$(".tab").forEach((tab) => tab.addEventListener("click", () => showTab(tab.dataset.tab)));
  if (location.hash === "#assembler") showTab("assembler");

  $("#form-recycler").addEventListener("submit", (e) => { e.preventDefault(); calculateRecycler(); });
  $("#form-assembler").addEventListener("submit", (e) => { e.preventDefault(); calculateAssembler(); });
  $$("[data-preset]").forEach((b) => b.addEventListener("click", () => PRESETS[b.dataset.preset]()));
}

function showTab(name) {
  $$(".tab").forEach((t) => t.classList.toggle("active", t.dataset.tab === name));
  $$(".tab-panel").forEach((p) => p.classList.toggle("active", p.id === `tab-${name}`));
  history.replaceState(null, "", `#${name}`);
}

function updateMachine() {
  const form = $("#form-assembler");
  const m = GAME.machines[form.machine.value];
  $(".machine-icon", form).src = ICON(m.icon);
  checkSlots();
}

function updateRecipe() {
  const form = $("#form-assembler");
  const r = GAME.recipes[form.recipe.value];
  $(".recipe-icon", form).src = r ? ICON(r.icon) : ICON("assembling_machine_3");

  // Only offer machines that can craft the recipe
  for (const opt of form.machine.options) opt.disabled = !!r && !r.machines.includes(opt.value);
  if (r && !r.machines.includes(form.machine.value)) {
    form.machine.value = r.machines.find((m) => GAME.machines[m]);
    updateMachine();
  }

  form.ingredient.innerHTML = "";
  form.ingredient.disabled = !r;
  form.recipe_time.readOnly = form.recipe_ratio.readOnly = !!r;
  if (!r) {
    $(".ingredient-icon", form).removeAttribute("src");
    $("#ingredient-hint").textContent = "";
    return;
  }
  for (const [item, n] of Object.entries(r.ingredients)) {
    form.ingredient.append(el("option", { value: item }, `${n} × ${GAME.items[item].name}`));
  }
  // Default to the ingredient with the highest input flow requirement
  form.ingredient.value = Object.entries(r.ingredients).reduce((a, b) => (b[1] > a[1] ? b : a))[0];
  form.recipe_time.value = r.time;
  updateIngredient();
}

function updateIngredient() {
  const form = $("#form-assembler");
  const r = GAME.recipes[form.recipe.value];
  const item = form.ingredient.value;
  $(".ingredient-icon", form).src = ICON(GAME.items[item].icon);
  const products = Object.values(r.products).reduce((a, b) => a + b, 0);
  form.recipe_ratio.value = +(products / r.ingredients[item]).toPrecision(6);
  // Other ingredients are assumed to be supplied at matching quality, outside the loop
  const others = Object.entries(r.ingredients).filter(([k]) => k !== item);
  $("#ingredient-hint").textContent = others.length
    ? `Flows are per ${GAME.items[item].name}. Each one also needs ` +
      others.map(([k, n]) => `${fmt(n / r.ingredients[item], 2)} ${GAME.items[k].name}`).join(", ") +
      " of the same quality, supplied separately."
    : "";
}

function checkSlots() {
  const form = $("#form-assembler");
  const m = GAME.machines[form.machine.value];
  const hint = $("#slot-hint");
  const bad = TIERS.map((_, i) => i).filter((i) => Number(form[`prod_${i}`].value) + Number(form[`qual_${i}`].value) > m.slots);
  hint.classList.toggle("error", bad.length > 0);
  hint.textContent = bad.length
    ? `Too many modules for ${m.name} (${m.slots} slots) in: ${bad.map((i) => TIERS[i]).join(", ")}`
    : `${m.name}: ${m.slots} module slots, base speed ${m.speed}, +${m.prod_bonus * 100}% productivity`;
  return bad.length === 0;
}

// ---------- Presets (from the README's worked examples) ----------

const PRESETS = {
  tungsten() {
    const f = $("#form-recycler");
    setTierInputs($("[data-tier-inputs]", f), [45, 15, 0, 0, 0]);
    f.rec_count.value = 5;
    setQuality($(".quality-picker[data-name=rec_quality]", f), 4);
    setModule($(".module-picker[data-name=rec_qmod]", f), 3, 4);
    f.recipe_time.value = 1;
    f.recycle_time.value = "";
    setQuality($(".quality-picker[data-name=keep]", f), 4);
  },
  u238() {
    const f = $("#form-assembler");
    setTierInputs($("[data-tier-inputs]", f), [8, 0, 0, 0, 0]);
    f.recipe.value = "";
    updateRecipe();
    f.machine.value = "assembling-machine-3";
    updateMachine();
    setQuality($(".quality-picker[data-name=machine_quality]", f), 4);
    f.research_prod.value = 0;
    setModule($(".module-picker[data-name=pmod]", f), 3, 4);
    setModule($(".module-picker[data-name=qmod]", f), 3, 4);
    [40, 4, 2, 1, 1].forEach((n, i) => {
      f[`count_${i}`].value = n;
      f[`prod_${i}`].value = i === 4 ? 4 : 0;
      f[`qual_${i}`].value = i === 4 ? 0 : 4;
    });
    f.rec_count.value = 8;
    setQuality($(".quality-picker[data-name=rec_quality]", f), 4);
    setModule($(".module-picker[data-name=rec_qmod]", f), 3, 4);
    f.recipe_time.value = 10;
    f.recipe_ratio.value = 1;
    f.recycle_time.value = "";
    setQuality($(".quality-picker[data-name=ing_keep]", f), 4);
    setQuality($(".quality-picker[data-name=prod_keep]", f), null);
    checkSlots();
  },
};

// ---------- API ----------

async function post(url, body, resultsEl) {
  resultsEl.innerHTML = "<h2>Results</h2><p class='placeholder'>Calculating…</p>";
  try {
    const res = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    const data = await res.json();
    if (!res.ok) throw new Error(formatError(data));
    return data;
  } catch (err) {
    resultsEl.innerHTML = "<h2>Results</h2>";
    resultsEl.append(el("div", { class: "error-box" }, err.message));
    return null;
  }
}

function formatError(data) {
  if (Array.isArray(data.detail)) return data.detail.map((d) => `${d.loc.slice(1).join(".")}: ${d.msg}`).join("\n");
  return data.detail || "Request failed";
}

const optionalNumber = (input) => (input.value === "" ? null : Number(input.value));

function recyclerSetup(form) {
  return {
    quality: getQuality($(".quality-picker[data-name=rec_quality]", form)),
    count: Number(form.rec_count.value),
    quality_module: getModule($(".module-picker[data-name=rec_qmod]", form)),
  };
}

// ---------- Recycler loop ----------

async function calculateRecycler() {
  const f = $("#form-recycler");
  const out = $("#results-recycler");
  const data = await post("/api/recycler-loop", {
    input_rates: getTierInputs($("[data-tier-inputs]", f)),
    recycler: recyclerSetup(f),
    recipe_time: Number(f.recipe_time.value),
    recycle_time: optionalNumber(f.recycle_time),
    quality_to_keep: getQuality($(".quality-picker[data-name=keep]", f)),
  }, out);
  if (data) renderRecycler(data, out);
}

function renderRecycler(d, out) {
  const keepFrom = d.kept.indexOf(true);
  const output = d.production_rate.reduce((s, r, i) => s + (d.kept[i] ? r : 0), 0);
  out.innerHTML = `
    <h2>Results</h2>
    <div class="stat-cards">
      <div class="stat-card q4"><div class="label">Legendary output</div>
        <div class="value">${fmt(d.production_rate[4] * 3600, 1)} <small>/ hour</small></div></div>
      <div class="stat-card q${keepFrom}"><div class="label">Total kept output (${TIERS[keepFrom]}+)</div>
        <div class="value">${fmt(output * 60, 2)} <small>/ min</small></div></div>
      ${loadCard("Recycler load", d.recycler_load)}
    </div>
    <div class="section-title"><img src="${ICON("recycler")}" alt="">Products</div>
    <table class="results-table">
      <thead><tr><th>Quality</th><th>Flow / s</th><th>Produced / min</th><th>Produced / hour</th><th>Status</th></tr></thead>
      <tbody>
        ${d.flows.map((flow, i) => `
          <tr class="q${i}">
            <td>${tierName(i)}</td>
            <td class="num">${fmt(flow, 3)}</td>
            <td class="num">${fmt(d.production_rate[i] * 60, 2)}</td>
            <td class="num">${fmt(d.production_rate[i] * 3600, 1)}</td>
            <td>${statusTag(d.kept[i])}</td>
          </tr>`).join("")}
      </tbody>
    </table>
    <p class="machine-stats">
      Recyclers: ${pct(d.derived.quality_chance)} quality chance, speed ${fmt(d.derived.recycler_speed, 2)}.
      Production rates match the in-game production statistics, which don't count items recycled into themselves.
    </p>`;
}

// ---------- Assembler loop ----------

async function calculateAssembler() {
  const f = $("#form-assembler");
  const out = $("#results-assembler");
  if (!checkSlots()) return;
  const data = await post("/api/assembler-loop", {
    input_rates: getTierInputs($("[data-tier-inputs]", f)),
    machine: f.machine.value,
    recipe: f.recipe.value || null,
    machine_quality: getQuality($(".quality-picker[data-name=machine_quality]", f)),
    research_prod_bonus: Number(f.research_prod.value) / 100,
    tiers: TIERS.map((_, i) => ({
      count: Number(f[`count_${i}`].value),
      prod_modules: Number(f[`prod_${i}`].value),
      quality_modules: Number(f[`qual_${i}`].value),
    })),
    prod_module: getModule($(".module-picker[data-name=pmod]", f)),
    quality_module: getModule($(".module-picker[data-name=qmod]", f)),
    recycler: recyclerSetup(f),
    recipe_time: Number(f.recipe_time.value),
    recipe_ratio: Number(f.recipe_ratio.value),
    recycle_time: optionalNumber(f.recycle_time),
    ingredient_quality_to_keep: getQuality($(".quality-picker[data-name=ing_keep]", f)),
    product_quality_to_keep: getQuality($(".quality-picker[data-name=prod_keep]", f)),
  }, out);
  if (data) renderAssembler(data, out, GAME.machines[f.machine.value]);
}

function renderAssembler(d, out, machine) {
  const ing = d.ingredients;
  const prod = d.products;
  const keptIng = ing.flows.reduce((s, x, i) => s + (ing.kept[i] ? x : 0), 0);
  const keptProd = prod.flows.reduce((s, x, i) => s + (prod.kept[i] ? x : 0), 0);
  const loads = ing.flows.map((x, i) => (ing.kept[i] ? 0 : x / ing.max_flow[i]));
  const recLoad = d.recycler.flow / d.recycler.max_flow;
  const bottlenecks = loads.map((l, i) => (l > 1 ? TIERS[i] : null)).filter(Boolean);
  if (recLoad > 1) bottlenecks.push("Recyclers");

  out.innerHTML = `
    <h2>Results</h2>
    <div class="stat-cards">
      <div class="stat-card q4"><div class="label">Legendary ingredients${ing.kept[4] ? "" : " (assembled)"}</div>
        <div class="value">${fmt(ing.flows[4] * 3600, 1)} <small>/ hour</small></div></div>
      <div class="stat-card q4"><div class="label">Legendary products${prod.kept[4] ? "" : " (recycled)"}</div>
        <div class="value">${fmt(prod.flows[4] * 3600, 1)} <small>/ hour</small></div></div>
      ${loadCard("Recycler load", recLoad)}
      <div class="stat-card"><div class="label">Bottlenecks</div>
        <div class="value" style="font-size:18px;color:${bottlenecks.length ? "var(--red)" : "var(--green)"}">
          ${bottlenecks.length ? bottlenecks.join(", ") : "None"}</div></div>
    </div>

    <div class="section-title"><img src="${ICON("recycler")}" alt="">Ingredients
      <span class="hint" style="margin-left:auto">kept: ${fmt(keptIng * 60, 2)} / min</span></div>
    <table class="results-table">
      <thead><tr><th>Quality</th><th>Flow / s</th><th>Flow / min</th><th>Assembler capacity / s</th><th>Assembler load</th><th>Status</th></tr></thead>
      <tbody>
        ${ing.flows.map((flow, i) => `
          <tr class="q${i}">
            <td>${tierName(i)}</td>
            <td class="num">${fmt(flow, 3)}</td>
            <td class="num">${fmt(flow * 60, 2)}</td>
            <td class="num">${ing.kept[i] ? "–" : fmt(ing.max_flow[i], 2)}</td>
            <td>${ing.kept[i] ? "" : loadBar(loads[i])}</td>
            <td>${statusTag(ing.kept[i], "Assembled")}</td>
          </tr>`).join("")}
      </tbody>
    </table>

    <div class="section-title"><img src="${ICON(machine.icon)}" alt="">Products
      <span class="hint" style="margin-left:auto">kept: ${fmt(keptProd * 60, 2)} / min</span></div>
    <table class="results-table">
      <thead><tr><th>Quality</th><th>Flow / s</th><th>Flow / min</th><th>Flow / hour</th><th>Status</th></tr></thead>
      <tbody>
        ${prod.flows.map((flow, i) => `
          <tr class="q${i}">
            <td>${tierName(i)}</td>
            <td class="num">${fmt(flow, 3)}</td>
            <td class="num">${fmt(flow * 60, 2)}</td>
            <td class="num">${fmt(flow * 3600, 1)}</td>
            <td>${statusTag(prod.kept[i], "Recycled")}</td>
          </tr>`).join("")}
      </tbody>
    </table>

    <div class="section-title"><img src="${ICON(machine.icon)}" alt="">Machines</div>
    <table class="results-table">
      <thead><tr><th>Assemblers crafting</th><th>Speed</th><th>Quality chance</th><th>Productivity bonus</th></tr></thead>
      <tbody>
        ${TIERS.map((_, i) => `
          <tr class="q${i}">
            <td>${tierName(i)}</td>
            <td class="num">${fmt(d.derived.assembler_speeds[i], 3)}</td>
            <td class="num">${pct(d.derived.assembler_quality_chance[i])}</td>
            <td class="num">+${pct(d.derived.assembler_prod_bonus[i], 0)}</td>
          </tr>`).join("")}
      </tbody>
    </table>
    <p class="machine-stats">
      Recyclers: ${pct(d.derived.recycler_quality_chance)} quality chance, speed ${fmt(d.derived.recycler_speed, 2)},
      handling ${fmt(d.recycler.flow, 2)} of ${fmt(d.recycler.max_flow, 2)} products / s.
    </p>`;
}

// ---------- Shared render bits ----------

function statusTag(kept, loopLabel = "Recycled") {
  return kept ? `<span class="tag kept">Kept</span>` : `<span class="tag loop">${loopLabel}</span>`;
}

function loadClass(load) {
  return load > 1 ? "over" : load > 0.9 ? "warn" : "";
}

function loadBar(load) {
  return `<div class="load ${loadClass(load)}"><span class="num">${pct(load, 0)}</span>
    <div class="bar"><span style="width:${Math.min(load, 1) * 100}%"></span></div></div>`;
}

function loadCard(label, load) {
  const colour = load > 1 ? "var(--red)" : load > 0.9 ? "var(--orange)" : "var(--green)";
  return `<div class="stat-card" style="--q:${colour}"><div class="label">${label}</div>
    <div class="value">${pct(load, 0)} <small>${load > 1 ? "add machines" : "of capacity"}</small></div></div>`;
}

// ---------- Boot ----------

fetch("/api/game-data")
  .then((r) => r.json())
  .then((data) => {
    GAME = data;
    initForms();
    PRESETS.tungsten();
    PRESETS.u238();
    calculateRecycler();
    calculateAssembler();
  });
