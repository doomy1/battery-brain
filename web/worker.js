// Runs the real Python package in the browser through Pyodide (CPython compiled
// to WebAssembly), off the main thread so the page stays responsive.
importScripts("https://cdn.jsdelivr.net/pyodide/v0.26.4/full/pyodide.js");

const FILES = [
  "__init__.py", "cell.py", "ocv.py", "profiles.py", "residual_net.py", "ekf.py",
  "qp.py", "mpc.py", "sim.py", "datagen.py", "weights/residual_net.json",
];

const GLUE = `
import json, sys
sys.path.insert(0, "/home/pyodide")
from batterybrain.sim import EstimationSession, ChargeRace

S = {}

def est_new(profile, ambient):
    S["est"] = EstimationSession(profile, ambient_c=ambient)
    return "{}"

def est_step(n):
    return json.dumps(S["est"].step(n))

def est_set(name, value):
    S["est"].set_param(name, value)
    return "{}"

def est_scramble():
    S["est"].scramble()
    return "{}"

def race_new(ambient, soc0, target, i_max, t_max):
    S["race"] = ChargeRace(ambient_c=ambient, soc0=soc0, target=target, i_max=i_max, t_max=t_max)
    return json.dumps(S["race"].limits)

def race_step(n):
    return json.dumps(S["race"].step(n))
`;

let pyodide = null;

async function init() {
  postMessage({ status: "Loading Python runtime…" });
  pyodide = await loadPyodide();
  postMessage({ status: "Loading NumPy…" });
  await pyodide.loadPackage("numpy");
  postMessage({ status: "Loading battery code…" });
  pyodide.FS.mkdirTree("/home/pyodide/batterybrain/weights");
  await Promise.all(FILES.map(async (name) => {
    const res = await fetch("../batterybrain/" + name);
    if (!res.ok) throw new Error("could not fetch " + name);
    pyodide.FS.writeFile("/home/pyodide/batterybrain/" + name, await res.text());
  }));
  pyodide.runPython(GLUE);
  postMessage({ ready: true });
}

const ready = init().catch((err) => postMessage({ fatal: String(err) }));

onmessage = async (event) => {
  await ready;
  const { id, cmd, args } = event.data;
  try {
    const fn = pyodide.globals.get(cmd);
    const result = JSON.parse(fn(...args));
    postMessage({ id, result });
  } catch (err) {
    postMessage({ id, error: String(err) });
  }
};
