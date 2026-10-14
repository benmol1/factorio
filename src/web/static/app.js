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
  // Show tiny values in scientific notation rather than letting them round to zero
  if (x !== 0 && Math.abs(x) < 0.5 * 10 ** -digits) return x.toExponential(1);
  return x.toLocaleString(undefined, { minimumFractionDigits: digits, maximumFractionDigits: digits });
}
// Flow rates to 3 significant figures, with k / M for large ones: 1260 → "1.26k"
function fmtFlow(x) {
  if (!isFinite(x)) return "–";
  const abs = Math.abs(x);
  const [scaled, suffix] = abs >= 1e6 ? [x / 1e6, "M"] : abs >= 1000 ? [x / 1e3, "k"] : [x, ""];
  // Rounding can carry a value up to the next unit (999.9 → 1,000), so show that as 1k
  if (suffix === "" && Math.abs(Number(scaled.toPrecision(3))) >= 1000) return fmtFlow(1000 * Math.sign(x));
  return scaled.toLocaleString(undefined, { maximumSignificantDigits: 3 }) + suffix;
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
  machineSelect.addEventListener("change", changeMachine);

  // Recipe dropdown
  const form = $("#form-assembler");
  for (const group of GAME.groups) {
    const options = Object.entries(GAME.recipes)
      .filter(([, r]) => r.group === group)
      .sort(([, a], [, b]) => a.name.localeCompare(b.name))
      .map(([key, r]) => el("option", { value: key }, r.name));
    form.recipe.append(el("optgroup", { label: group }, ...options));
  }
  form.recipe.addEventListener("change", updateRecipe);
  form.ingredient.addEventListener("change", updateIngredient);

  const tbody = $("#tier-setup");
  TIERS.forEach((_, i) => {
    tbody.append(
      el("tr", {},
        el("td", { html: tierName(i) }),
        el("td", {}, el("input", { type: "number", min: "1", value: "1", name: `count_${i}` })),
        el("td", {}, el("span", { class: "quality-picker", "data-name": `machine_quality_${i}` })),
        el("td", {}, el("input", { type: "number", min: "0", value: "0", name: `prod_${i}` })),
        el("td", {}, el("input", { type: "number", min: "0", value: "4", name: `qual_${i}` })),
      ),
    );
  });
  $$(".quality-picker", tbody).forEach((s) => buildQualityPicker(s, 0));
  tbody.addEventListener("input", checkSlots);
  changeMachine();

  // Tabs
  $$(".tab").forEach((tab) => tab.addEventListener("click", () => showTab(tab.dataset.tab)));
  if (location.hash === "#recycler") showTab("recycler");

  $("#form-recycler").addEventListener("submit", (e) => { e.preventDefault(); calculateRecycler(); });
  $("#form-assembler").addEventListener("submit", (e) => { e.preventDefault(); calculateAssembler(); });
  $("#solve").addEventListener("click", solveAssembler);
  $$("[data-preset]").forEach((b) =>
    b.addEventListener("click", () => {
      PRESETS[b.dataset.preset]();
      $$("form").forEach(saveState);
    }),
  );
  // Programmatic changes dispatch change events too, so pickers and presets are covered as well
  $$("form").forEach((f) => ["input", "change"].forEach((ev) => f.addEventListener(ev, () => saveState(f))));
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

// Fills every slot: quality modules in the lower tiers, productivity modules in the top one
function fillSlots() {
  const form = $("#form-assembler");
  const slots = GAME.machines[form.machine.value].slots;
  TIERS.forEach((_, i) => {
    form[`prod_${i}`].value = i === TIERS.length - 1 ? slots : 0;
    form[`qual_${i}`].value = i === TIERS.length - 1 ? 0 : slots;
  });
  checkSlots();
}

function changeMachine() {
  updateMachine();
  fillSlots();
}

function updateRecipe() {
  const form = $("#form-assembler");
  const r = GAME.recipes[form.recipe.value];
  $(".recipe-icon", form).src = r ? ICON(r.icon) : ICON("assembling_machine_3");

  // Only offer machines that can craft the recipe
  for (const opt of form.machine.options) opt.disabled = !!r && !r.machines.includes(opt.value);
  if (r && !r.machines.includes(form.machine.value)) {
    form.machine.value = r.machines.find((m) => GAME.machines[m]);
    changeMachine();
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

// ---------- Remembering inputs across refreshes ----------

const STORAGE_PREFIX = "quality-loop:";

function collectState(form) {
  const state = { fields: {}, qualities: {}, modules: {}, tiers: {} };
  for (const f of $$("input[name], select[name]", form)) state.fields[f.name] = f.value;
  for (const s of $$(".quality-picker[data-name]", form)) state.qualities[s.dataset.name] = getQuality(s);
  for (const s of $$(".module-picker[data-name]", form)) state.modules[s.dataset.name] = getModule(s);
  for (const c of $$("[data-tier-inputs]", form)) state.tiers[c.dataset.tierInputs] = getTierInputs(c);
  return state;
}

function applyState(form, state, hooks = {}) {
  const setField = (name, value) => {
    const f = form.elements[name];
    if (!f) return;
    // A select can lose options between versions (e.g. a removed recipe); keep the default then
    if (f.tagName === "SELECT" && ![...f.options].some((o) => o.value === value)) return;
    f.value = value;
  };
  // Recipe first, because choosing it rewrites the machine, ingredient and timing fields
  if ("recipe" in state.fields) {
    setField("recipe", state.fields.recipe);
    hooks.recipe?.();
  }
  for (const [name, value] of Object.entries(state.fields)) if (name !== "recipe") setField(name, value);
  for (const [name, q] of Object.entries(state.qualities)) {
    const s = $(`.quality-picker[data-name=${name}]`, form);
    if (s) setQuality(s, q);
  }
  for (const [name, m] of Object.entries(state.modules)) {
    const s = $(`.module-picker[data-name=${name}]`, form);
    if (s) setModule(s, m.level, m.quality);
  }
  for (const [name, values] of Object.entries(state.tiers)) {
    const c = $(`[data-tier-inputs=${name}]`, form);
    if (c) setTierInputs(c, values);
  }
  hooks.after?.();
}

function saveState(form) {
  try {
    localStorage.setItem(STORAGE_PREFIX + form.id, JSON.stringify(collectState(form)));
  } catch {
    // Storage can be blocked (private windows); the app works fine without it
  }
}

function loadState(form) {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_PREFIX + form.id));
  } catch {
    return null;
  }
}

// Restores a saved form and returns whether there was one to restore
function restoreForm(form, hooks) {
  const state = loadState(form);
  if (state) applyState(form, state, hooks);
  return !!state;
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
  holmium() {
    const f = $("#form-assembler");
    setTierInputs($("[data-tier-inputs]", f), [20, 0, 0, 0, 0]);
    f.recipe.value = "electromagnetic-plant";
    updateRecipe();
    f.ingredient.value = "holmium-plate";
    updateIngredient();
    f.machine.value = "electromagnetic-plant";
    changeMachine();
    f.research_prod.value = 0;
    setModule($(".module-picker[data-name=pmod]", f), 3, 4);
    setModule($(".module-picker[data-name=qmod]", f), 3, 4);
    [4, 1, 1, 1, 1].forEach((n, i) => {
      f[`count_${i}`].value = n;
      setQuality($(`.quality-picker[data-name=machine_quality_${i}]`, f), 0);
    });
    f.rec_count.value = 2;
    setQuality($(".quality-picker[data-name=rec_quality]", f), 0);
    setModule($(".module-picker[data-name=rec_qmod]", f), 3, 4);
    f.recycle_time.value = "";
    setQuality($(".quality-picker[data-name=ing_keep]", f), 4);
    setQuality($(".quality-picker[data-name=prod_keep]", f), null);
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
        <div class="value">${fmtFlow(d.production_rate[4] * 3600)} <small>/ hour</small></div></div>
      <div class="stat-card q${keepFrom}"><div class="label">Total kept output (${TIERS[keepFrom]}+)</div>
        <div class="value">${fmtFlow(output * 60)} <small>/ min</small></div></div>
      ${loadCard("Recycler load", d.recycler_load, recyclerHint($("#form-recycler")), ICON("recycler"))}
    </div>
    <div class="section-title"><img src="${ICON("recycler")}" alt="">Products</div>
    <table class="results-table">
      <thead><tr><th>Quality</th><th>Flow / s</th><th>Produced / min</th><th>Produced / hour</th><th>Status</th></tr></thead>
      <tbody>
        ${d.flows.map((flow, i) => `
          <tr class="q${i}">
            <td>${tierName(i)}</td>
            <td class="num">${fmtFlow(flow)}</td>
            <td class="num">${fmtFlow(d.production_rate[i] * 60)}</td>
            <td class="num">${fmtFlow(d.production_rate[i] * 3600)}</td>
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

function assemblerRequest(f) {
  return {
    input_rates: getTierInputs($("[data-tier-inputs]", f)),
    machine: f.machine.value,
    recipe: f.recipe.value || null,
    research_prod_bonus: Number(f.research_prod.value) / 100,
    tiers: TIERS.map((_, i) => ({
      count: Number(f[`count_${i}`].value),
      machine_quality: getQuality($(`.quality-picker[data-name=machine_quality_${i}]`, f)),
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
  };
}

async function calculateAssembler() {
  const f = $("#form-assembler");
  const out = $("#results-assembler");
  if (!checkSlots()) return;
  const data = await post("/api/assembler-loop", assemblerRequest(f), out);
  if (data) {
    renderAssembler(data, out, GAME.machines[f.machine.value], GAME.recipes[f.recipe.value], f.ingredient.value,
      getTierInputs($("[data-tier-inputs]", f)).reduce((a, b) => a + b, 0));
  }
}

// Fills in the machine counts and qualities the solver picks, then shows the result
async function solveAssembler() {
  const f = $("#form-assembler");
  const status = $("#solve-status");
  if (!checkSlots()) return;
  status.classList.remove("error");
  status.textContent = "Solving…";
  let data;
  try {
    const res = await fetch("/api/solve", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(assemblerRequest(f)) });
    data = await res.json();
    if (!res.ok) throw new Error(formatError(data));
  } catch (err) {
    status.classList.add("error");
    status.textContent = err.message;
    return;
  }
  data.tiers.forEach((t, i) => {
    // A tier that needs no machines (e.g. a kept one) still needs a valid count in the form
    f[`count_${i}`].value = Math.max(t.count, 1);
    setQuality($(`.quality-picker[data-name=machine_quality_${i}]`, f), t.machine_quality);
  });
  f.rec_count.value = Math.max(data.recycler.count, 1);
  setQuality($(".quality-picker[data-name=rec_quality]", f), data.recycler.quality);
  saveState(f);
  status.textContent = "Solved: the fewest machines per stage, at the lowest machine quality that still needs no more.";
  calculateAssembler();
}

function renderAssembler(d, out, machine, recipe, ingredient, inputFlow) {
  const ing = d.ingredients;
  const prod = d.products;
  const keptIng = ing.kept.indexOf(true);
  const keptProd = prod.kept.indexOf(true);
  const keptIngFlow = ing.flows.reduce((s, x, i) => s + (ing.kept[i] ? x : 0), 0);
  const loads = ing.flows.map((x, i) => (ing.kept[i] ? 0 : x / ing.max_flow[i]));
  const recLoad = d.recycler.flow / d.recycler.max_flow;
  const bottlenecks = loads.map((l, i) => (l > 1 ? TIERS[i] : null)).filter(Boolean);
  if (recLoad > 1) bottlenecks.push("Recyclers");

  // Icons follow the recipe: the critical ingredient and the recipe's product
  const ingIcon = ingredient ? ICON(GAME.items[ingredient].icon) : ICON("recycler");
  const prodIcon = recipe ? ICON(recipe.icon) : ICON(machine.icon);
  const name = (item, fallback) => (item ? GAME.items[item].name : fallback);
  const ingName = name(ingredient, "ingredient");
  const prodName = recipe ? recipe.name : "product";

  // Only the lowest kept tier is shown for each output the user is looking for
  const keptSection = (icon, title, flows, tier) => `
    <div class="section-title"><img src="${icon}" alt="">${title}</div>
    <table class="results-table">
      <thead><tr><th>Quality</th><th>Flow / s</th><th>Flow / min</th><th>Flow / hour</th></tr></thead>
      <tbody>
        <tr class="q${tier}">
          <td>${tierName(tier)}</td>
          <td class="num">${fmtFlow(flows[tier])}</td>
          <td class="num">${fmtFlow(flows[tier] * 60)}</td>
          <td class="num">${fmtFlow(flows[tier] * 3600)}</td>
        </tr>
      </tbody>
    </table>`;

  out.innerHTML = `
    <h2>Results</h2>
    <div class="stat-cards">
      ${keptIng >= 0 ? outputCard(ingIcon, `${TIERS[keptIng]} ${ingName}`, ing.flows[keptIng]) : ""}
      ${keptProd >= 0 ? outputCard(prodIcon, `${TIERS[keptProd]} ${prodName}`, prod.flows[keptProd]) : ""}
      ${loadCard("Recycler load", recLoad, recyclerHint($("#form-assembler")), ICON("recycler"))}
      <div class="stat-card" style="--q:${bottlenecks.length ? "var(--red)" : "var(--green)"}"><div class="label">Bottlenecks</div>
        <div class="value" style="color:${bottlenecks.length ? "var(--red)" : "var(--green)"}">
          ${bottlenecks.length ? `✘ <small>${bottlenecks.join(", ")}</small>` : "✔ <small>None</small>"}</div></div>
    </div>
    <div class="section-title"><img src="${ingIcon}" alt="">${ingName} (ingredients)
      ${keptIng >= 0 ? `<span class="hint" style="margin-left:auto">kept: ${fmtFlow(keptIngFlow * 60)} / min</span>` : ""}</div>
    <table class="results-table">
      <thead><tr><th>Quality</th><th>Flow / s</th><th>Flow / min</th><th>Assembler capacity / s</th><th>Assembler load</th><th>Status</th></tr></thead>
      <tbody>
        ${ing.flows.map((flow, i) => `
          <tr class="q${i}">
            <td>${tierName(i)}</td>
            <td class="num">${fmtFlow(flow)}</td>
            <td class="num">${fmtFlow(flow * 60)}</td>
            <td class="num">${ing.kept[i] ? "–" : fmtFlow(ing.max_flow[i])}</td>
            <td>${ing.kept[i] ? "" : loadBar(loads[i])}</td>
            <td>${statusTag(ing.kept[i], "Assembled")}</td>
          </tr>`).join("")}
      </tbody>
    </table>
    ${ingredientSupplyHint(recipe, ingredient, inputFlow)}
    ${keptProd >= 0 ? keptSection(prodIcon, `${prodName} kept (${TIERS[keptProd]}${keptProd < 4 ? "+" : ""})`, prod.flows, keptProd) : ""}
    
    <div class="section-title"><img src="${ICON(machine.icon)}" alt="">Machines</div>
    <table class="results-table">
      <thead><tr><th>Assemblers crafting</th><th>Count</th><th>Machine quality</th><th>Speed</th><th>Quality chance</th><th>Productivity bonus</th></tr></thead>
      <tbody>
        ${TIERS.map((_, i) => `
          <tr class="q${i}">
            <td>${tierName(i)}</td>
            <td class="num">${$("#form-assembler")[`count_${i}`].value}</td>
            <td>${tierName(getQuality($(`.quality-picker[data-name=machine_quality_${i}]`, $("#form-assembler"))) ?? 0)}</td>
            <td class="num">${fmt(d.derived.assembler_speeds[i], 3)}</td>
            <td class="num">${pct(d.derived.assembler_quality_chance[i])}</td>
            <td class="num">+${pct(d.derived.assembler_prod_bonus[i], 0)}</td>
          </tr>`).join("")}
      </tbody>
    </table>
    <p class="machine-stats">
      Recyclers: ${pct(d.derived.recycler_quality_chance)} quality chance, speed ${fmt(d.derived.recycler_speed, 2)},
      handling ${fmtFlow(d.recycler.flow)} of ${fmtFlow(d.recycler.max_flow)} products / s.
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

// The raw flow of every recipe ingredient to supply, scaled from the critical ingredient's input rate
function ingredientSupplyHint(recipe, ingredient, inputFlow) {
  if (!recipe || !ingredient) return "";
  const flows = Object.entries(recipe.ingredients).map(
    ([item, n]) => `${GAME.items[item].name} ${fmtFlow((inputFlow * n) / recipe.ingredients[ingredient])} / s`,
  );
  return `<p class="hint">Raw input to supply: ${flows.join(" · ")}</p>`;
}

// Headline flow of one kept output, at its lowest kept quality
function outputCard(icon, label, flow) {
  const tier = TIERS.findIndex((t) => label.startsWith(t));
  return `<div class="stat-card q${tier}"><div class="label"><img class="card-icon" src="${icon}" alt="">${label}</div>
    <div class="value">${fmtFlow(flow * 60)} <small>/ min</small></div>
    <div class="label">${fmtFlow(flow)} / s · ${fmtFlow(flow * 3600)} / hour</div></div>`;
}

function loadCard(label, load, hint = "", icon = "") {
  const colour = load > 1 ? "var(--red)" : load > 0.9 ? "var(--orange)" : "var(--green)";
  const iconTag = icon ? `<img class="card-icon" src="${icon}" alt="">` : "";
  return `<div class="stat-card" style="--q:${colour}"><div class="label">${iconTag}${label}${hint ? ` <small>(${hint})</small>` : ""}</div>
    <div class="value">${pct(load, 0)} <small>${load > 1 ? "add machines" : "of capacity"}</small></div></div>`;
}

// e.g. "8 × Legendary", from the recycler fields of a form
function recyclerHint(form) {
  const { count, quality } = recyclerSetup(form);
  return `${count} × ${TIERS[quality]}`;
}

// ---------- Boot ----------

fetch("/api/game-data")
  .then((r) => r.json())
  .then((data) => {
    GAME = data;
    initForms();
    const recyclerForm = $("#form-recycler");
    const assemblerForm = $("#form-assembler");
    if (!restoreForm(recyclerForm)) PRESETS.tungsten();
    const restored = restoreForm(assemblerForm, {
      recipe: updateRecipe,
      after() {
        if (assemblerForm.recipe.value) updateIngredient();
        updateMachine();
      },
    });
    if (!restored) PRESETS.u238();
    // Event-driven saves can miss the last programmatic edits, so save the final state explicitly
    saveState(recyclerForm);
    saveState(assemblerForm);
    calculateRecycler();
    calculateAssembler();
  });
