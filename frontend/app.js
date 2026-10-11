/* Pump Digital Twin v2 — motor first, sliders, then 5-state diagnosis */
const BACKEND_URL = window.REACT_APP_BACKEND_URL || window.BACKEND_API_URL || (window.location.hostname === "localhost" ? "http://localhost:8000" : "");
const API_URL = BACKEND_URL ? `${BACKEND_URL}/api/predict` : "/api/predict";

const PARAMETERS = [
  { name:"A_ACR_Mot.PV", label:"Motor Current (PV)", min:0,  max:120,  unit:"A"   },
  { name:"A_ACR_Mot.SV", label:"Motor Current (SP)", min:0,  max:120,  unit:"A"   },
  { name:"A_ACR_Mot.TV", label:"Motor Totalizer",    min:0,  max:50000,unit:"-"   },
  { name:"A_ACR_Pmp.PV", label:"Pump Current (PV)",  min:0,  max:120,  unit:"A"   },
  { name:"A_ACR_Pmp.SV", label:"Pump Current (SP)",  min:0,  max:120,  unit:"A"   },
  { name:"A_ACR_Pmp.TV", label:"Pump Totalizer",     min:0,  max:50000,unit:"-"   },
  { name:"A_Pres.PV",    label:"Pressure",           min:0,  max:10,   step:0.1, unit:"bar" },
  { name:"A_Temp.PV",    label:"Pump Temperature",   min:0,  max:120,  unit:"°C"  },
  { name:"Vibration_hz", label:"Vibration",          min:0,  max:120,  unit:"Hz"  },
  { name:"Temperature",  label:"Oil/Ambient Temp",   min:0,  max:120,  unit:"°C"  },
  { name:"Load_Percent", label:"Load",               min:0,  max:100,  unit:"%"   },
  { name:"RPM",          label:"Pump Speed",         min:0,  max:3600, step:10,  unit:"RPM" },
  { name:"Barometer",    label:"Barometer",          min:950,max:1050, unit:"hPa" },
];
const BASELINE = [45,45,8000,45,45,8000,6.5,62,18,52,55,1750,1013];

const MODES = {
  normal:{ f:[1,1,1,1,1,1,1,1,1,1,1,1,1] },
  light: { f:[0.75,0.75,0.6,0.75,0.75,0.6,0.95,0.85,0.6,0.85,0.65,0.85,1] },
  stress:{ f:[1.2,1.2,1.1,1.2,1.2,1.1,1.05,1.15,1.6,1.15,1.25,1.05,1] },
  heavy: { f:[1.4,1.4,1.3,1.4,1.4,1.3,0.85,1.35,2.4,1.3,1.55,0.9,0.99] },
  rough: { f:[1.7,1.7,1.6,1.7,1.7,1.6,0.6,1.7,3.6,1.6,1.9,0.75,0.97] },
};

const STATES = [
  { min:90, name:"HEALTHY",         color:"#00e676" },
  { min:75, name:"GOOD",            color:"#7bd88f" },
  { min:55, name:"NEUTRAL",         color:"#ffd600" },
  { min:35, name:"NEEDS ATTENTION", color:"#ff9100" },
  { min:0,  name:"FAILURE",         color:"#ff1744" },
];

const fmt = v => v>=1000 ? Math.round(v).toLocaleString() : (+v).toFixed(1);
const getVal = name => parseFloat(document.getElementById("in-"+name).value);
const setConn = (txt, cls) => {
  const el = document.getElementById("connStatus");
  el.textContent = txt; el.className = "conn " + cls;
};

/* ---- Build sliders ---- */
const wrap = document.getElementById("sliders");
PARAMETERS.forEach((p,i) => {
  const row = document.createElement("div");
  row.className = "param";
  row.innerHTML =
    '<div class="param-head"><span>'+p.label+'</span>' +
    '<span class="param-val"><b id="val-'+p.name+'">'+fmt(BASELINE[i])+'</b> '+p.unit+'</span></div>' +
    '<input type="range" id="in-'+p.name+'" min="'+p.min+'" max="'+p.max+'" step="'+(p.step||1)+'" value="'+BASELINE[i]+'">';
  wrap.appendChild(row);
  row.querySelector("input").addEventListener("input", e => {
    document.getElementById("val-"+p.name).textContent = fmt(+e.target.value);
  });
});

/* ---- Motor animation ---- */
function setMotor(rpm, vib) {
  document.getElementById("motorRpm").textContent = Math.round(rpm);
  document.getElementById("impeller").style.animationDuration =
    rpm > 0 ? Math.max(0.15, 3600 / rpm).toFixed(2) + "s" : "0s";
  document.getElementById("pumpStage").classList.toggle("shake", vib > 50);
}

/* ---- Mode buttons ---- */
document.querySelectorAll(".modes button").forEach(btn => {
  btn.addEventListener("click", () => applyMode(btn.dataset.mode));
});
function applyMode(key) {
  const m = MODES[key];
  PARAMETERS.forEach((p,i) => {
    let v = BASELINE[i] * m.f[i];
    v = Math.max(p.min, Math.min(p.max, v));
    document.getElementById("in-"+p.name).value = v;
    document.getElementById("val-"+p.name).textContent = fmt(v);
  });
  document.querySelectorAll(".modes button").forEach(b =>
    b.classList.toggle("active", b.dataset.mode === key));
  runDiagnosis();
}

/* ---- Local fallback diagnosis (when backend is offline) ---- */
function localDiagnosis(f) {
  const motPV=f[0], pmpPV=f[3], pres=f[6], ptemp=f[7], vib=f[8],
        otemp=f[9], load=f[10], rpm=f[11];
  let score = 100; const factors = [];
  const add = (pts,label) => { if (pts>0){ score-=pts; factors.push({label, pts:+pts.toFixed(1)}); } };
  add(Math.max(0,(vib-40)*0.6),        "High vibration ("+vib.toFixed(1)+" Hz)");
  add(Math.max(0,(ptemp-70)*0.7),      "Pump temperature high ("+ptemp.toFixed(1)+"°C)");
  add(Math.max(0,(otemp-70)*0.7),      "Oil/ambient temperature high ("+otemp.toFixed(1)+"°C)");
  add(Math.max(0,(load-80)*0.5),       "Load above 80% ("+load.toFixed(1)+"%)");
  add(Math.max(0,(3-pres)*10),         "Low pressure ("+pres.toFixed(2)+" bar)");
  add(Math.max(0,(motPV-70)*0.5) + Math.max(0,(pmpPV-70)*0.5),
      "Current overload ("+Math.max(motPV,pmpPV).toFixed(1)+" A)");
  add(Math.abs(rpm-1750)/1750*100*0.3, "Speed deviation from rated 1750 RPM");
  score = Math.max(0, Math.min(100, score));
  return { score:+score.toFixed(1),
           rul:Math.round(score*45),
           anomaly:+((100-score)*0.85).toFixed(1),
           source:"BUILT-IN SIMULATION (backend offline)",
           factors:factors.sort((a,b)=>b.pts-a.pts).slice(0,5) };
}

/* ---- Main diagnosis ---- */
async function runDiagnosis() {
  const btn = document.getElementById("diagnoseBtn");
  btn.disabled = true; btn.textContent = "⏳ ANALYZING...";
  const features = PARAMETERS.map(p => getVal(p.name));
  setMotor(features[11], features[8]);

  const local = localDiagnosis(features);
  let result = local;
  try {
    const res = await fetch(API_URL, {
      method:"POST",
      headers:{ "Content-Type":"application/json" },
      body: JSON.stringify({ features }),
    });
    if (!res.ok) throw new Error();
    const d = await res.json();
    result = { score:d.health_score, rul:Math.round(d.rul_hours),
               anomaly:d.anomaly_probability, source:"ML MODEL (backend live)",
               factors:local.factors };
    setConn("● ML MODEL LIVE","live");
  } catch(e) {
    setConn("● SIMULATION MODE","sim");
  }
  renderResult(result);
  btn.disabled = false; btn.textContent = "🔍 RUN HEALTH DIAGNOSIS";
}

/* ---- Render ---- */
function renderResult(r) {
  const st = STATES.find(s => r.score >= s.min);
  const badge = document.getElementById("stateBadge");
  badge.textContent = st.name;
  badge.style.color = st.color;
  badge.style.borderColor = st.color;
  badge.style.background = st.color + "22";
  document.getElementById("healthBar").style.width = r.score + "%";
  document.getElementById("healthBar").style.background = st.color;
  const sc = document.getElementById("healthScore");
  sc.textContent = r.score; sc.style.color = st.color;
  document.getElementById("rul").textContent = r.rul.toLocaleString() + " h";
  document.getElementById("anomaly").textContent = r.anomaly + " %";
  document.getElementById("source").textContent = "Source: " + r.source;

  const f = document.getElementById("factors");
  const list = r.factors && r.factors.length ? r.factors
             : [{ label:"All parameters within safe range ✔", pts:0 }];
  f.innerHTML = list.map(x =>
    '<div class="factor"><span>'+x.label+'</span>' +
    '<span style="color:var(--warn)">'+(x.pts ? "−"+x.pts+" pts" : "")+'</span></div>'
  ).join("");
}

document.getElementById("diagnoseBtn").addEventListener("click", runDiagnosis);

/* Motor runs on load with Normal mode */
applyMode("normal");