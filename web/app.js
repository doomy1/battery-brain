"use strict";

const $ = (id) => document.getElementById(id);
const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const clock = (s) => {
  s = Math.max(0, Math.round(s));
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = s % 60;
  return (h ? h + ":" + String(m).padStart(2, "0") : m) + ":" + String(sec).padStart(2, "0");
};

/* ---------------------------------------------------------------- worker */
const worker = new Worker("worker.js");
const pending = new Map();
let nextId = 1;

function setStatus(text, cls) {
  const el = $("status");
  el.textContent = text;
  el.className = "pill" + (cls ? " " + cls : "");
}

worker.onmessage = (event) => {
  const m = event.data;
  if (m.status) return setStatus(m.status);
  if (m.fatal) return setStatus("Could not start: " + m.fatal, "bad");
  if (m.ready) { setStatus("Live · Python running in your browser", "ok"); return start(); }
  const p = pending.get(m.id);
  pending.delete(m.id);
  if (m.error) { setStatus("Error: " + m.error, "bad"); p.reject(new Error(m.error)); }
  else p.resolve(m.result);
};

const call = (cmd, ...args) => new Promise((resolve, reject) => {
  const id = nextId++;
  pending.set(id, { resolve, reject });
  worker.postMessage({ id, cmd, args });
});

/* ----------------------------------------------------------------- chart */
function niceStep(raw) {
  const p = Math.pow(10, Math.floor(Math.log10(raw)));
  const f = raw / p;
  return (f <= 1 ? 1 : f <= 2 ? 2 : f <= 5 ? 5 : 10) * p;
}

class Chart {
  constructor(host, o) {
    this.o = o;
    host.className = "chart" + (o.tall ? " tall" : "");
    host.innerHTML =
      `<div class="chart-head"><span class="chart-title">${o.title}</span><span class="legend">` +
      o.series.map((s) => `<span><i style="background:var(${s.color})"></i>${s.label}</span>`).join("") +
      `</span></div><canvas></canvas>`;
    this.canvas = host.querySelector("canvas");
    this.last = null;
  }

  redraw() { if (this.last) this.draw(...this.last); }

  draw(x, ys, x0, x1) {
    this.last = [x, ys, x0, x1];
    const o = this.o, c = this.canvas, dpr = window.devicePixelRatio || 1;
    const W = c.clientWidth, H = c.clientHeight;
    if (!W) return;
    if (c.width !== Math.round(W * dpr) || c.height !== Math.round(H * dpr)) {
      c.width = Math.round(W * dpr);
      c.height = Math.round(H * dpr);
    }
    const g = c.getContext("2d");
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, W, H);
    const L = 46, R = 10, T = 8, B = 22, pw = W - L - R, ph = H - T - B;

    // y range
    let lo = Infinity, hi = -Infinity;
    for (const y of ys) for (const v of y) { if (v < lo) lo = v; if (v > hi) hi = v; }
    for (const lim of o.limits || []) { lo = Math.min(lo, lim.value); hi = Math.max(hi, lim.value); }
    if (!isFinite(lo)) { lo = 0; hi = 1; }
    if (o.symmetric) { hi = Math.max(Math.abs(lo), Math.abs(hi)); lo = -hi; }
    if (o.cap !== undefined) { hi = Math.min(hi, o.cap); lo = Math.max(lo, -o.cap); }
    const minSpan = o.minSpan || 1e-6;
    if (hi - lo < minSpan) { const mid = (hi + lo) / 2; lo = mid - minSpan / 2; hi = mid + minSpan / 2; }
    const pad = 0.08 * (hi - lo);
    lo -= pad; hi += pad;
    if (o.floor !== undefined) lo = Math.max(lo, o.floor);
    if (o.ceil !== undefined) hi = Math.min(hi, o.ceil);

    const X = (v) => L + ((v - x0) / (x1 - x0)) * pw;
    const Y = (v) => T + (1 - (v - lo) / (hi - lo)) * ph;
    const muted = cssVar("--muted"), grid = cssVar("--grid");
    g.font = "11px system-ui, sans-serif";
    g.lineWidth = 1;

    // y grid and labels
    const ystep = niceStep((hi - lo) / 4);
    const digits = Math.max(0, -Math.floor(Math.log10(ystep)));
    g.textAlign = "right"; g.textBaseline = "middle";
    for (let v = Math.ceil(lo / ystep) * ystep; v <= hi + 1e-9; v += ystep) {
      g.strokeStyle = grid; g.beginPath(); g.moveTo(L, Y(v)); g.lineTo(L + pw, Y(v)); g.stroke();
      g.fillStyle = muted; g.fillText(v.toFixed(digits), L - 6, Y(v));
    }
    // x labels
    const xstep = o.xStep(x1 - x0);
    g.textAlign = "center"; g.textBaseline = "top";
    for (let v = Math.ceil(x0 / xstep) * xstep; v <= x1 + 1e-9; v += xstep) {
      g.fillStyle = muted; g.fillText(o.xLabel(v), X(v), T + ph + 6);
    }
    g.fillStyle = muted; g.textAlign = "left"; g.textBaseline = "top";
    g.fillText(o.unit, 4, 0);

    g.save();
    g.beginPath(); g.rect(L, T, pw, ph); g.clip();
    // limit lines
    for (const lim of o.limits || []) {
      g.strokeStyle = cssVar("--limit"); g.setLineDash([5, 4]);
      g.beginPath(); g.moveTo(L, Y(lim.value)); g.lineTo(L + pw, Y(lim.value)); g.stroke();
      g.setLineDash([]);
      g.fillStyle = cssVar("--limit"); g.textAlign = "right"; g.textBaseline = "bottom";
      g.fillText(lim.label, L + pw - 4, Y(lim.value) - 2);
    }
    // series
    o.series.forEach((s, k) => {
      const y = ys[k];
      if (!y || !y.length) return;
      g.strokeStyle = cssVar(s.color);
      g.lineWidth = s.width || 1.6;
      g.setLineDash(s.dash || []);
      g.beginPath();
      for (let i = 0; i < y.length; i++) i ? g.lineTo(X(x[i]), Y(y[i])) : g.moveTo(X(x[i]), Y(y[i]));
      g.stroke();
    });
    g.restore();
  }
}

const secAxis = {
  xStep: (span) => [10, 30, 60, 120, 300, 600, 1200].find((s) => span / s <= 8) || 1800,
  xLabel: clock,
};

/* ------------------------------------------------------------ estimation */
const EST_WINDOW = 900;   // seconds of history shown
const EST_KEYS = ["t", "current", "voltage", "temp", "soc_true", "soc_cc", "soc_ekf", "soc_nn", "v_ekf", "v_nn"];
const est = { buf: {}, busy: false, paused: false, acc: 0, charts: {} };

function estClear() { for (const k of EST_KEYS) est.buf[k] = []; est.acc = 0; }

function estInitCharts() {
  est.charts.soc = new Chart($("c-soc"), {
    title: "State of charge", unit: "%", tall: true, minSpan: 8, ...secAxis,
    series: [
      { label: "True", color: "--true", width: 2.4 },
      { label: "Coulomb counting", color: "--cc" },
      { label: "Kalman filter", color: "--ekf" },
      { label: "Kalman + neural net", color: "--nn", dash: [6, 3] },
    ],
  });
  est.charts.err = new Chart($("c-err"), {
    title: "Estimation error", unit: "% SOC", symmetric: true, cap: 12, minSpan: 1, ...secAxis,
    limits: [{ value: 0, label: "" }],
    series: [
      { label: "Coulomb counting", color: "--cc" },
      { label: "Kalman filter", color: "--ekf" },
      { label: "Kalman + neural net", color: "--nn" },
    ],
  });
  est.charts.verr = new Chart($("c-verr"), {
    title: "How wrong is each voltage model?", unit: "mV", symmetric: true, minSpan: 20, ...secAxis,
    series: [
      { label: "Circuit model alone", color: "--ekf" },
      { label: "Circuit model + neural net", color: "--nn" },
    ],
  });
  est.charts.cur = new Chart($("c-cur"), {
    title: "Current (＋ discharge)", unit: "A", minSpan: 4, ...secAxis,
    series: [{ label: "Load", color: "--true" }],
  });
  est.charts.temp = new Chart($("c-temp"), {
    title: "Cell temperature", unit: "°C", minSpan: 4, ...secAxis,
    series: [{ label: "Cell", color: "--table" }],
  });
}

function estAppend(out) {
  const b = est.buf;
  for (const k of EST_KEYS) b[k].push(...out[k]);
  const tEnd = b.t[b.t.length - 1];
  let cut = 0;
  while (b.t[cut] < tEnd - EST_WINDOW) cut++;
  if (cut) for (const k of EST_KEYS) b[k].splice(0, cut);
  estRender(out);
}

function estRender(out) {
  const b = est.buf, n = b.t.length;
  if (!n) return;
  const x1 = b.t[n - 1], x0 = Math.max(0, x1 - EST_WINDOW), xr = Math.max(x1, x0 + 60);
  const pct = (a) => a.map((v) => 100 * v);
  const errOf = (k) => b[k].map((v, i) => 100 * (v - b.soc_true[i]));
  est.charts.soc.draw(b.t, [pct(b.soc_true), pct(b.soc_cc), pct(b.soc_ekf), pct(b.soc_nn)], x0, xr);
  est.charts.err.draw(b.t, [errOf("soc_cc"), errOf("soc_ekf"), errOf("soc_nn")], x0, xr);
  est.charts.verr.draw(b.t, [
    b.voltage.map((v, i) => 1000 * (v - b.v_ekf[i])),
    b.voltage.map((v, i) => 1000 * (v - b.v_nn[i])),
  ], x0, xr);
  est.charts.cur.draw(b.t, [b.current], x0, xr);
  est.charts.temp.draw(b.t, [b.temp], x0, xr);

  const i = n - 1, truth = b.soc_true[i];
  $("est-fill").style.width = (100 * truth).toFixed(1) + "%";
  $("est-true").textContent = (100 * truth).toFixed(1) + " %";
  $("est-i").textContent = b.current[i].toFixed(1) + " A";
  $("est-v").textContent = b.voltage[i].toFixed(3) + " V";
  $("est-t").textContent = b.temp[i].toFixed(1) + " °C";
  for (const k of ["cc", "ekf", "nn"]) {
    const v = b["soc_" + k][i];
    const e = 100 * (v - truth);
    $(`s-${k}-est`).textContent = (100 * v).toFixed(1) + " %";
    $(`s-${k}-err`).textContent = (e >= 0 ? "+" : "") + e.toFixed(2);
    if (out) $(`s-${k}-rms`).textContent = (100 * out.rmse[k]).toFixed(2);
  }
}

const estSettings = () => ({
  ambient: +$("est-ambient").value,
  bias_ma: +$("est-bias").value,
  noise_mv: +$("est-noise").value,
  fade: +$("est-fade").value / 100,
  manual_current: +$("est-manual").value,
});

async function estReset() {
  est.busy = true;
  estClear();
  await call("est_new", $("est-profile").value, +$("est-ambient").value);
  const s = estSettings();
  for (const k of Object.keys(s)) await call("est_set", k, s[k]);
  est.busy = false;
}

function estWire() {
  const slider = (id, out, param, fmt, scale = 1) => {
    $(id).addEventListener("input", () => {
      $(out).textContent = fmt(+$(id).value);
      call("est_set", param, +$(id).value * scale);
    });
  };
  slider("est-ambient", "o-ambient", "ambient", (v) => v + " °C");
  slider("est-bias", "o-bias", "bias_ma", (v) => v + " mA");
  slider("est-noise", "o-noise", "noise_mv", (v) => v + " mV");
  slider("est-fade", "o-fade", "fade", (v) => v + " %", 0.01);
  slider("est-manual", "o-manual", "manual_current", (v) => v + " A");
  $("est-profile").addEventListener("change", () => {
    const v = $("est-profile").value;
    $("manual-row").hidden = v !== "manual";
    call("est_set", "profile", v);
  });
  $("est-scramble").addEventListener("click", () => call("est_scramble"));
  $("est-pause").addEventListener("click", () => {
    est.paused = !est.paused;
    $("est-pause").textContent = est.paused ? "Resume" : "Pause";
  });
  $("est-reset").addEventListener("click", estReset);
}

/* ------------------------------------------------------------------ race */
const RACE_KEYS = ["soc", "current", "voltage", "temp"];
const race = { t: [], table: {}, mpc: {}, running: false, busy: false, acc: 0, charts: {}, limits: null };

function raceInitCharts() {
  const pairSeries = [{ label: "Table charger", color: "--table" }, { label: "MPC", color: "--mpc", width: 2 }];
  const axis = { xStep: secAxis.xStep, xLabel: clock };
  race.charts.soc = new Chart($("c-r-soc"), { title: "State of charge", unit: "%", tall: true, minSpan: 20, series: pairSeries, limits: [], ...axis });
  race.charts.cur = new Chart($("c-r-cur"), { title: "Charging current", unit: "A", minSpan: 5, floor: -0.5, series: pairSeries, limits: [], ...axis });
  race.charts.volt = new Chart($("c-r-volt"), { title: "Voltage", unit: "V", minSpan: 0.3, series: pairSeries, limits: [], ...axis });
  race.charts.temp = new Chart($("c-r-temp"), { title: "Temperature", unit: "°C", minSpan: 8, series: pairSeries, limits: [], ...axis });
}

async function raceStart() {
  race.busy = true;
  const tmax = +$("r-tmax").value, imax = +$("r-imax").value;
  race.limits = await call("race_new", +$("r-ambient").value, +$("r-soc0").value / 100, 0.8, imax, tmax);
  race.t = [];
  for (const lane of ["table", "mpc"]) { race[lane] = {}; for (const k of RACE_KEYS) race[lane][k] = []; }
  race.charts.soc.o.limits = [{ value: 80, label: "target 80 %" }];
  race.charts.cur.o.limits = [{ value: imax, label: "limit " + imax + " A" }];
  race.charts.volt.o.limits = [{ value: 4.2, label: "limit 4.2 V" }];
  race.charts.temp.o.limits = [{ value: tmax, label: "limit " + tmax + " °C" }];
  $("r-banner").innerHTML = "<b>Racing…</b> watch the temperature chart: the table charger backs off in steps, the predictive controller rides the limit.";
  $("r-start").textContent = "Restart race";
  race.acc = 0;
  race.running = true;
  race.busy = false;
}

function raceAppend(out) {
  race.t.push(...out.t);
  for (const lane of ["table", "mpc"]) for (const k of RACE_KEYS) race[lane][k].push(...out[lane][k]);
  const n = race.t.length;
  if (!n) return;
  const x1 = Math.max(race.t[n - 1], 120);
  const both = (k, f = (v) => v) => [race.table[k].map(f), race.mpc[k].map(f)];
  race.charts.soc.draw(race.t, both("soc", (v) => 100 * v), 0, x1);
  race.charts.cur.draw(race.t, both("current"), 0, x1);
  race.charts.volt.draw(race.t, both("voltage"), 0, x1);
  race.charts.temp.draw(race.t, both("temp"), 0, x1);

  for (const lane of ["table", "mpc"]) {
    const d = race[lane], s = out.summary[lane];
    const soc = 100 * d.soc[n - 1];
    $(`r-${lane}-fill`).style.width = soc.toFixed(1) + "%";
    $(`r-${lane}-soc`).textContent = soc.toFixed(1) + " %";
    $(`r-${lane}-time`).textContent = s.done ? "✓ " + clock(s.finish_time) : clock(race.t[n - 1]);
    $(`r-${lane}-i`).textContent = d.current[n - 1].toFixed(1) + " A";
    $(`r-${lane}-t`).textContent = d.temp[n - 1].toFixed(1) + " °C";
  }

  if (out.finished || race.t[n - 1] > 3 * 3600) {
    race.running = false;
    const a = out.summary.table, b = out.summary.mpc;
    if (a.done && b.done) {
      const gain = (100 * (a.finish_time - b.finish_time)) / a.finish_time;
      const verdict = gain >= 0.5
        ? `<b>MPC wins:</b> 80 % in ${clock(b.finish_time)}, ${gain.toFixed(0)} % faster than the table charger (${clock(a.finish_time)}).`
        : gain <= -0.5
          ? `<b>Table charger wins:</b> ${clock(a.finish_time)} against ${clock(b.finish_time)} for MPC. Here the cell is voltage-limited, so prediction has nothing to gain and its safety margin costs a little.`
          : `<b>Dead heat:</b> both took about ${clock(a.finish_time)}.`;
      $("r-banner").innerHTML = verdict +
        ` Peak temperature: MPC ${b.peak_temp.toFixed(1)} °C, table ${a.peak_temp.toFixed(1)} °C (limit ${race.limits.t_max} °C).` +
        ` Peak voltage: MPC ${b.peak_volt.toFixed(3)} V, table ${a.peak_volt.toFixed(3)} V.`;
    } else {
      $("r-banner").innerHTML = "<b>Stopped:</b> at least one charger did not reach 80 % within three hours.";
    }
  }
}

function raceWire() {
  const show = (id, out, fmt) => {
    const f = () => { $(out).textContent = fmt(+$(id).value); };
    $(id).addEventListener("input", f);
    f();
  };
  show("r-ambient", "o-r-ambient", (v) => v + " °C");
  show("r-tmax", "o-r-tmax", (v) => v + " °C");
  show("r-imax", "o-r-imax", (v) => `${v} A (${(v / 5).toFixed(1)}C)`);
  show("r-soc0", "o-r-soc0", (v) => v + " %");
  $("r-start").addEventListener("click", raceStart);
}

/* ------------------------------------------------------------- main loop */
let activeTab = "est";
let lastFrame = null;

function frame(now) {
  const dt = lastFrame === null ? 0 : Math.min(1.0, (now - lastFrame) / 1000);
  lastFrame = now;

  if (activeTab === "est" && !est.paused && !est.busy) {
    est.acc += dt * +$("est-speed").value;
    const n = Math.min(400, Math.floor(est.acc));
    if (n >= 1) {
      est.acc -= n;
      est.busy = true;
      call("est_step", n).then(estAppend).finally(() => { est.busy = false; });
    }
  }
  if (activeTab === "race" && race.running && !race.busy) {
    race.acc += dt * +$("r-speed").value;
    const n = Math.min(400, Math.floor(race.acc));
    if (n >= 1) {
      race.acc -= n;
      race.busy = true;
      call("race_step", n).then(raceAppend).finally(() => { race.busy = false; });
    }
  }
  requestAnimationFrame(frame);
}

function redrawAll() {
  for (const c of Object.values(est.charts)) c.redraw();
  for (const c of Object.values(race.charts)) c.redraw();
}

function wireTabs() {
  for (const tab of document.querySelectorAll(".tab")) {
    tab.addEventListener("click", () => {
      activeTab = tab.dataset.tab;
      for (const t of document.querySelectorAll(".tab")) t.classList.toggle("active", t === tab);
      for (const name of ["est", "race", "about"]) $("tab-" + name).hidden = name !== activeTab;
      redrawAll();
    });
  }
  window.addEventListener("resize", redrawAll);
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", redrawAll);
}

async function start() {
  await estReset();
  requestAnimationFrame(frame);
}

estClear();
estInitCharts();
raceInitCharts();
estWire();
raceWire();
wireTabs();
