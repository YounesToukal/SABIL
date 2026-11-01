// SABIL engine in the browser: a line-by-line port of src/simulate.py (dispatch),
// src/controllers.py (Standard and SABIL), src/aging_model.py (Naumann LFP ageing,
// half-cycle detector, two-node thermal model) and src/load_profile.py.
// Parameters: the selected kit in params/ (commit bff04f2): 24 V DC system, 400 Wp, LFP 25.6 V 75 Ah = 1.92 kWh,
// household.json (1,164 Wh a day, 952 Wh critical), sim.json smart rules incl. cloudy_day_buffer, battery.json thermal.
"use strict";
const SABIL = (() => {
  const P = {
    sys: { pv_kwp: 0.4, derate: 0.92, tc: -0.0035, noct: 45, inv: 0.95 },
    bat: { cap: 1.92, ce: 0.97, de: 0.97, maxC: 0.5, socMin: 0.10, socMax: 1.0, eol: 0.8, tempOffset: 3.0, rTh: 0.5, cells: 8, vNom: 25.6 },
    aging: { kref: 1.2571e-5, ea: 17126, tref: 298.15, sa: 2.8575, sb: 0.60225, ca: 0.0630, cb: 0.0971, cc: 4.0253, cd: 1.0923 },
    smart: { hot: 30, capHot: 0.70, sunny: 1.0, margin: 0.15, cHot: 0.25, cloudyBuffer: 0.5 },
    thermal: { v: 25.6, r: 0.015, mass: 18, cp: 1100 },
    placements: {
      indoor_default: { label: "Indoors, about 3 °C above outdoor air (study default)", short: "Indoors, +3 °C" },
      outdoor_shaded_box: { tau: 1, off: 3, ua: 3, label: "Metal box outside, in the shade", short: "Outdoor box, shaded" },
      indoor_metal_roof: { tau: 3, off: 2, ua: 2, label: "Room under a metal roof", short: "Under a metal roof" },
      indoor_masonry: { tau: 6, off: 0, ua: 2, label: "Thick-walled room", short: "Thick-walled room" },
      cool_interior_floor: { tau: 24, off: -2, ua: 2, label: "Floor of the coolest room", short: "Cool interior floor" },
    },
    appliances: [ // params/household.json, five people
      { name: "LED lights, indoor", w: 3, q: 4, hours: [5, 18, 19, 20, 21, 22], crit: true },
      { name: "LED security light", w: 5, q: 1, hours: [19, 20, 21, 22, 23, 0, 1, 2, 3, 4], crit: true },
      { name: "Phone charging", w: 5, q: 5, hours: [19, 20], crit: true, flex: [10, 11, 12, 13, 14, 15] },
      { name: "Bedroom fan (night)", w: 20, q: 1, hours: [21, 22, 23, 0, 1, 2, 3, 4, 5], crit: true },
      { name: "Living-room fan (day)", w: 20, q: 1, hours: [12, 13, 14, 15, 16, 17], crit: false },
      { name: "TV", w: 20, q: 1, hours: [18, 19, 20, 21], crit: false },
      { name: "Radio", w: 3, q: 1, hours: [6, 7, 18, 19], crit: false },
      { name: "Fridge, daytime cooling", w: 25, q: 1, hours: [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17], crit: true },
      { name: "Fridge, night cooling", w: 17.5, q: 1, hours: [18, 19, 20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6], crit: true },
      { name: "Fridge pre-cooling", w: 16.25, q: 1, hours: [18, 19, 20, 21, 22, 23], crit: true, flex: [10, 11, 12, 13, 14, 15] },
    ],
    gen_l_per_kwh: 0.5,        // costs.json: generator at a good load factor (best case)
    gen_l_per_kwh_small: 2.0,  // costs.json: small household generator (A2EI)
    tariff_ngn_per_kwh: 505,   // costs_summary.json, SABIL break-even tariff (504.76)
  };
  const R_GAS = 8.3144598;

  // ---------- weather ----------
  let W = null; // {n, ghi, amb, fc0, fc1, fc1p10, fc0p10}
  function setWeather(d) {
    const n = d.ghi.length;
    W = { n, ghi: Float64Array.from(d.ghi), amb: Float64Array.from(d.amb10, (v) => v / 10),
      fc0: Float64Array.from(d.fc0), fc1: Float64Array.from(d.fc1), fc0p10: Float64Array.from(d.fc0p10), fc1p10: Float64Array.from(d.fc1p10),
      start: new Date(d.start + ":00Z") };
    return W;
  }
  const mod = (t) => ((t % W.n) + W.n) % W.n;
  const hod = (t) => mod(t) % 24;
  function pvDc(ghi, amb, kwp) {
    const tCell = amb + (P.sys.noct - 20) / 800 * ghi;
    return Math.max(0, kwp * ghi / 1000 * P.sys.derate * (1 + P.sys.tc * (tCell - 25)));
  }

  // ---------- household (load_profile.build, deterministic) ----------
  function household(scale) {
    const crit = new Float64Array(24), nonc = new Float64Array(24), flex = [];
    for (const a of P.appliances) {
      const kw = a.w * a.q / 1000 * scale;
      if (a.flex) { flex.push({ name: a.name, kw, n: a.hours.length, use: a.hours, allowed: a.flex, crit: a.crit }); continue; }
      for (const h of a.hours) (a.crit ? crit : nonc)[h] += kw;
    }
    return { crit, nonc, flex };
  }

  // ---------- ageing model (aging_model.py) ----------
  function newAging() { return { qCal: 0, qCyc: 0, fec: 0, prev: null, start: null, dir: 0, hours: 0, pending: [], tRoom: null, tBat: null }; }
  function observeSoc(s, socEnd, dt) {
    if (s.prev === null) { s.prev = s.start = socEnd; return; }
    const delta = socEnd - s.prev;
    if (Math.abs(delta) < 1e-12) return;
    const d = delta > 0 ? 1 : -1;
    if (s.dir === 0 || d === s.dir) { s.dir = d; s.hours += dt; s.prev = socEnd; return; }
    const dod = Math.abs(s.prev - s.start);
    const cRate = s.hours > 0 ? dod / s.hours : 0;
    s.pending.push([dod, cRate]);
    s.start = s.prev; s.dir = d; s.hours = dt; s.prev = socEnd;
  }
  function batteryTemp(s, th, amb, powerKw) {
    if (th.mode !== "rc") { // aging_nrel.battery_temp: offset + steady self-heating from conversion losses
      const eff = powerKw >= 0 ? P.bat.ce : P.bat.de;
      return amb + P.bat.tempOffset + P.bat.rTh * Math.abs(powerKw) * 1000 * (1 - eff);
    }
    const pl = P.placements[th.placement];
    if (s.tRoom === null) s.tRoom = s.tBat = amb + pl.off;
    s.tRoom += Math.min(1, 1 / pl.tau) * (amb + pl.off - s.tRoom);
    const I = Math.abs(powerKw) * 1000 / P.thermal.v;
    const qHeat = I * I * P.thermal.r;
    s.tBat += 3600 / (P.thermal.mass * P.thermal.cp) * (pl.ua * (s.tRoom - s.tBat) + qHeat);
    return s.tBat;
  }
  function tempEstimate(th, amb) { return th.mode !== "rc" ? amb + P.bat.tempOffset : amb + P.placements[th.placement].off; }
  function calRate(soc, tempC) {
    const a = P.aging, T = tempC + 273.15;
    return a.kref * Math.exp(-a.ea / R_GAS * (1 / T - 1 / a.tref)) * (a.sa * (soc - 0.5) ** 3 + a.sb);
  }
  function stepLoss(s, dt, soc, tempC) {
    const a = P.aging;
    const k = calRate(soc, tempC);
    let dCal = 0, dCyc = 0;
    if (k > 0) { const tv = (s.qCal / k) ** 2; dCal = k * Math.sqrt(tv + dt * 3600) - s.qCal; s.qCal += dCal; }
    for (const [dod, cRate] of s.pending) {
      const dfec = dod / 2;
      const kc = (a.ca * cRate + a.cb) * (a.cc * (dod - 0.6) ** 3 + a.cd);
      if (dfec > 0 && kc > 0) {
        const fv = (s.qCyc * 100 / kc) ** 2;
        const dq = kc * Math.sqrt(fv + dfec) / 100 - s.qCyc;
        s.qCyc += dq; dCyc += dq; s.fec += dfec;
      }
    }
    s.pending.length = 0;
    return Math.max(0, dCal) + Math.max(0, dCyc);
  }

  // ---------- forecasts ----------
  // PV forecast (DC kW) issued at midnight t0 for t0..t0+H-1, learned mode (forecast_lab ensemble):
  // same-day forecast for hours 0-23, next-day forecast for hours 24-47; temperature by persistence.
  function pvForecastLearned(t0, H, kwp, p10) {
    const out = new Float64Array(H);
    for (let k = 0; k < H; k++) {
      const i = mod(t0 + k), ambP = W.amb[mod(t0 + k - 24)];
      const g = k < 24 ? (p10 ? W.fc0p10 : W.fc0)[i] : (p10 ? W.fc1p10 : W.fc1)[i];
      out[k] = pvDc(g, ambP, kwp);
    }
    return out;
  }

  // ---------- controllers ----------
  class Standard {
    constructor(hh) {
      this.name = "standard";
      const b = P.bat;
      this.limits = { soc_max: b.socMax, soc_floor_crit: b.socMin, soc_floor_nonc: b.socMin, chg_c: b.maxC, dis_c_crit: b.maxC, dis_c_nonc: b.maxC, priority: false };
      const c = new Float64Array(24), nc = new Float64Array(24);
      for (const f of hh.flex) for (const h of f.use) (f.crit ? c : nc)[h] += f.kw;
      this.usual = [c, nc]; this.today = null;
    }
    planDay() { return this.usual; }
    decide() { return this.limits; }
  }

  class Smart {
    constructor(hh, th) {
      this.name = "sabil"; this.hh = hh; this.th = th; this.today = null;
      this.etaOut = P.bat.de * P.sys.inv;
    }
    // pvAc: 48 h forecast of PV AC kW from t0; ambFc: 24 h temperature forecast
    planDay(t0, pvAc, ambFc) {
      const H = 48, hh = this.hh;
      const crit = new Float64Array(H), nonc = new Float64Array(H);
      for (let k = 0; k < H; k++) { crit[k] = hh.crit[hod(t0 + k)]; nonc[k] = hh.nonc[hod(t0 + k)]; }
      const fc = new Float64Array(24), fnc = new Float64Array(24);
      const surplus = new Float64Array(24);
      for (let k = 0; k < 24; k++) surplus[k] = pvAc[k] - crit[k] - nonc[k];
      const moved = [];
      for (const f of [...hh.flex].sort((a, b) => b.kw - a.kw)) {
        const hours = f.allowed.length ? f.allowed : f.use;
        const best = [...hours].sort((a, b) => surplus[b] - surplus[a]).slice(0, f.n);
        for (const h of best) { (f.crit ? fc : fnc)[h] += f.kw; surplus[h] -= f.kw; }
        moved.push({ name: f.name, from: f.use, to: best });
      }
      for (let k = 0; k < 24; k++) { crit[k] += fc[k]; nonc[k] += fnc[k]; }
      const load = new Float64Array(H);
      for (let k = 0; k < H; k++) load[k] = crit[k] + nonc[k];
      let tsum = 0;
      for (let k = 0; k < 24; k++) tsum += tempEstimate(this.th, ambFc[k]);
      const meanTemp = tsum / 24;
      const hot = meanTemp >= P.smart.hot;
      let pvTom = 0, loadTom = 0;
      for (let k = 24; k < 48; k++) { pvTom += pvAc[k]; loadTom += load[k]; }
      const sunnyTomorrow = pvTom >= P.smart.sunny * loadTom;
      let start = H;
      for (let k = 12; k < H; k++) if (pvAc[k] < load[k]) { start = k; break; }
      let end = H;
      for (let k = Math.max(start, 24); k < H; k++) if (pvAc[k] >= load[k]) { end = k; break; }
      let eNight = 0;
      for (let k = start; k < end; k++) eNight += Math.max(0, load[k] - pvAc[k]);
      // cloudy-day buffer (Youcef, sizing review): keep part of tomorrow's critical energy in the capped charge
      let critTom = 0;
      for (let k = 24; k < 48; k++) critTom += crit[k];
      const eTonight = eNight;
      eNight += P.smart.cloudyBuffer * critTom;
      const critDef = new Float64Array(H), sunOk = new Uint8Array(H);
      for (let k = 0; k < H; k++) { critDef[k] = Math.max(0, crit[k] - pvAc[k]); sunOk[k] = pvAc[k] > 0 && pvAc[k] >= crit[k] ? 1 : 0; }
      const remCrit = new Float64Array(24), sunBack = new Int16Array(24);
      for (let h = 0; h < 24; h++) {
        let k = h + 1;
        while (k < H && sunOk[k]) k++;
        let e = 0;
        while (k < H && !sunOk[k]) { e += critDef[k]; k++; }
        remCrit[h] = e; sunBack[h] = k;
      }
      this.today = { t0, hot, sunnyTomorrow, eNight, eTonight, critTom, remCrit, sunBack, meanTemp, pvTom, loadTom, moved, nightStart: start, nightEnd: end };
      return [fc, fnc];
    }
    decide(t, state) {
      const d = this.today, s = P.smart, b = P.bat;
      const cap = state.capacity_kwh, m = 1 + s.margin, h = t - d.t0;
      const needSoc = b.socMin + d.eNight * m / (cap * this.etaOut);
      const capped = d.hot && d.sunnyTomorrow;
      const soc_max = capped ? Math.min(b.socMax, Math.max(s.capHot, needSoc)) : b.socMax;
      const reserve = Math.min(soc_max, b.socMin + d.remCrit[h] * m / (cap * this.etaOut));
      const hotNow = state.last_battery_temp_c >= s.hot;
      const gentle = hotNow ? s.cHot : b.maxC;
      return { soc_max, soc_floor_crit: b.socMin, soc_floor_nonc: reserve, chg_c: gentle, dis_c_crit: b.maxC, dis_c_nonc: gentle,
        priority: true, capped, needSoc, hotNow, reserveKwh: d.remCrit[h] };
    }
  }

  // ---------- one-hour dispatch (simulate.run inner loop) ----------
  function dispatch(soc, cap, lim, pvDcKw, crit, nonc) {
    const inv = P.sys.inv, ce = P.bat.ce, de = P.bat.de, maxC = P.bat.maxC, dt = 1;
    const load = crit + nonc, pvAc = pvDcKw * inv;
    let pvC, pvN;
    if (lim.priority) { pvC = Math.min(pvAc, crit); pvN = Math.min(pvAc - pvC, nonc); }
    else { const share = load > 0 ? Math.min(pvAc, load) / load : 0; pvC = crit * share; pvN = nonc * share; }
    const surplusDc = pvDcKw - (pvC + pvN) / inv;
    const needC = crit - pvC, needN = nonc - pvN;
    let chg = 0, dis = 0, curt = 0, supC = 0, supN = 0;
    if (surplusDc > 1e-12) {
      const room = Math.max(0, lim.soc_max - soc) * cap;
      chg = Math.min(surplusDc, lim.chg_c * cap, room / (ce * dt));
      soc += chg * ce * dt / cap; curt = surplusDc - chg;
    }
    if (needC + needN > 1e-12) {
      const avail = (floor, c) => Math.min(c * cap, Math.max(0, soc - floor) * cap * de / dt) * inv;
      if (lim.priority) {
        supC = Math.min(needC, avail(lim.soc_floor_crit, lim.dis_c_crit));
        dis += supC / inv; soc -= supC / inv / de * dt / cap;
        supN = Math.min(needN, avail(lim.soc_floor_nonc, Math.min(lim.dis_c_nonc, maxC)), Math.max(0, maxC * cap - dis) * inv);
        dis += supN / inv; soc -= supN / inv / de * dt / cap;
      } else {
        const need = needC + needN;
        const sup = Math.min(need, avail(lim.soc_floor_crit, lim.dis_c_crit));
        supC = needC * sup / need; supN = needN * sup / need;
        dis = sup / inv; soc -= sup / inv / de * dt / cap;
      }
    }
    return { soc, chg, dis, curt, batKw: chg - dis, servedC: pvC + supC, servedN: pvN + supN, pvAc, load, crit, nonc };
  }

  // ---------- LFP pack voltage (display only; 16 cells in series) ----------
  const OCV = [[0, 2.90], [0.05, 3.15], [0.1, 3.20], [0.2, 3.25], [0.3, 3.28], [0.4, 3.29], [0.5, 3.30], [0.6, 3.31], [0.7, 3.32], [0.8, 3.33], [0.9, 3.35], [0.95, 3.38], [1, 3.45]];
  function packVolts(soc, batKw) {
    soc = Math.min(1, Math.max(0, soc));
    let v = 3.3;
    for (let i = 1; i < OCV.length; i++) if (soc <= OCV[i][0]) { const [x0, y0] = OCV[i - 1], [x1, y1] = OCV[i]; v = y0 + (y1 - y0) * (soc - x0) / (x1 - x0); break; }
    const ocv = P.bat.cells * v, I = batKw * 1000 / ocv;
    return { v: ocv + I * P.thermal.r, a: I };
  }

  // ---------- a home: one kit, one controller ----------
  class Kit {
    constructor(cfg, kind) {
      this.cfg = cfg; this.kind = kind;
      this.hh = household(cfg.loadScale);
      this.th = { mode: cfg.placement && cfg.placement !== "indoor_default" ? "rc" : "default", placement: cfg.placement || "indoor_default" };
      this.ctrl = kind === "sabil" ? new Smart(this.hh, this.th) : new Standard(this.hh);
      this.reset(cfg.t0 || 0);
    }
    reset(t0) {
      this.age = newAging();
      this.capFrac = 1; this.soc = P.bat.socMax; this.t = t0; this.installedAt = t0;
      observeSoc(this.age, this.soc, 1);
      this.lastTemp = tempEstimate(this.th, W.amb[mod(t0)]);
      this.plan = [new Float64Array(24), new Float64Array(24)]; this.dayT0 = t0; this.lim = null;
      this.pi = 1; this.piN = 0; // panel performance index (dust check)
      this.stats = { served: 0, unmet: 0, critH: 0, critOkH: 0, hours: 0, pvUnused: 0, critDemKwh: 0, critSrvKwh: 0 };
      this.log = []; this.last = null;
    }
    // forecast of PV AC (kW) for 48 h from midnight t0, used by the plan
    planForecast(t0) {
      const fc = pvForecastLearned(t0, 48, this.cfg.pvKwp, false);
      const pvAc = new Float64Array(48);
      for (let k = 0; k < 48; k++) pvAc[k] = fc[k] * P.sys.inv * this.pi;
      const amb = new Float64Array(24);
      for (let k = 0; k < 24; k++) amb[k] = W.amb[mod(t0 + k - 24)];
      return { pvAc, amb };
    }
    // inp: {ghi, amb, pvDc?, tBat?, soc?, extraKw?, otherOff?}
    step(inp) {
      const t = this.t, h0 = hod(t);
      if (t === this.installedAt || h0 === 0) {
        const f = this.planForecast(t);
        this.plan = this.ctrl.planDay(t, f.pvAc, f.amb);
        this.dayT0 = t;
      }
      const h = t - this.dayT0;
      let crit = inp.critKw != null ? inp.critKw : this.hh.crit[h0] + this.plan[0][h];
      let nonc = inp.noncKw != null ? inp.noncKw : this.hh.nonc[h0] + this.plan[1][h] + (inp.extraKw || 0);
      if (inp.otherOff) nonc = 0;
      if (inp.soc != null) this.soc = inp.soc;
      const cap = P.bat.cap * this.capFrac;
      const state = { capacity_kwh: cap, soc: this.soc, last_battery_temp_c: this.lastTemp };
      const lim = this.ctrl.decide(t, state);
      const expected = pvDc(inp.ghi, inp.amb, this.cfg.pvKwp);
      const pv = inp.pvDc != null ? inp.pvDc : expected;
      const soc0 = this.soc;
      const r = dispatch(this.soc, cap, lim, pv, crit, nonc);
      this.soc = r.soc;
      const temp = inp.tBat != null ? inp.tBat : batteryTemp(this.age, this.th, inp.amb, r.batKw);
      if (inp.tBat != null) { this.age.tBat = inp.tBat; }
      observeSoc(this.age, this.soc, 1);
      const loss = stepLoss(this.age, 1, 0.5 * (soc0 + this.soc), temp);
      this.capFrac -= Math.max(0, loss);
      this.lastTemp = temp;
      // dust check: measured PV vs what clean panels give for the same sun
      if (expected > 0.3 * this.cfg.pvKwp) {
        const ratio = Math.min(1.3, pv / expected), a = 1 / 20;
        this.pi = this.pi + a * (ratio - this.pi); this.piN++;
      }
      const st = this.stats;
      const served = r.servedC + r.servedN;
      st.served += served; st.unmet += r.load - served; st.hours++; st.pvUnused += r.curt;
      st.critDemKwh += crit; st.critSrvKwh += r.servedC;
      if (crit > 0.001) { st.critH++; if (r.servedC >= crit - 0.001) st.critOkH++; }
      const rec = { t, ghi: inp.ghi, amb: inp.amb, pv, pvExp: expected, pvAc: r.pvAc, crit, nonc, servedC: r.servedC, servedN: r.servedN,
        batKw: r.batKw, soc: this.soc, temp, cap: this.capFrac, lim, curt: r.curt };
      this.last = rec;
      this.log.push(rec); if (this.log.length > 72) this.log.shift();
      this.t = t + 1;
      return rec;
    }
    // re-plan the current day after a new forecast arrives (keeps the day's hour index)
    replan() {
      const f = this.planForecast(this.dayT0);
      this.plan = this.ctrl.planDay(this.dayT0, f.pvAc, f.amb);
    }
    ageHours() { return this.t - this.installedAt; }
    health() { return this.capFrac; }
    projectedLifeYears() {
      const q = 1 - this.capFrac, yrs = this.ageHours() / 8760;
      if (q <= 0 || yrs <= 0) return null;
      return yrs * ((1 - P.bat.eol) / q) ** 2;
    }
    // project the next H hours with the forecast, no ageing (what the family sees as "tonight")
    project(H, pvFcDc, ambFc) {
      let soc = this.soc, t = this.t;
      const cap = P.bat.cap * this.capFrac;
      const saved = this.ctrl.today, savedPlan = this.plan, savedT0 = this.dayT0;
      const out = [];
      let lastTemp = this.lastTemp;
      for (let k = 0; k < H; k++, t++) {
        const h0 = hod(t);
        if (h0 === 0) {
          const pvAc = new Float64Array(48), amb = new Float64Array(24);
          for (let j = 0; j < 48; j++) pvAc[j] = (j + k < H ? pvFcDc[j + k] : pvDc(W.fc1[mod(t + j)], W.amb[mod(t + j - 24)], this.cfg.pvKwp) * this.pi) * P.sys.inv;
          for (let j = 0; j < 24; j++) amb[j] = W.amb[mod(t + j - 24)];
          this.plan = this.ctrl.planDay(t, pvAc, amb); this.dayT0 = t;
        }
        const h = t - this.dayT0;
        const crit = this.hh.crit[h0] + this.plan[0][h], nonc = this.hh.nonc[h0] + this.plan[1][h];
        const lim = this.ctrl.decide(t, { capacity_kwh: cap, soc, last_battery_temp_c: lastTemp });
        const r = dispatch(soc, cap, lim, pvFcDc[k], crit, nonc);
        soc = r.soc;
        lastTemp = tempEstimate(this.th, ambFc[k]);
        out.push({ t, soc, pv: pvFcDc[k], crit, nonc, servedC: r.servedC, servedN: r.servedN, floor: lim.soc_floor_nonc, socMax: lim.soc_max });
      }
      this.ctrl.today = saved; this.plan = savedPlan; this.dayT0 = savedT0;
      return out;
    }
  }

  return { P, setWeather, get W() { return W; }, pvDc, household, Kit, packVolts, calRate, tempEstimate, hod, mod, pvForecastLearned };
})();
if (typeof module !== "undefined") module.exports = SABIL;
