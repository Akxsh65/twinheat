// Feasibility check: deterministic pointwise certificates for a surrogate prediction on the TwinHeat FV system (M-matrix).
const TH = require('./core.js');
const b = TH.defaultBoard(), s = b.assemble(), N = 64, n = N * N;
const xs = TH.pcg(s, null, 1e-13, 6000).x;
const pk = Math.max(...xs), R = TH.rng(11);
const Ax = v => TH.applyA(s, v, new Float64Array(n));
const maxAbs = v => v.reduce((m, x) => Math.max(m, Math.abs(x)), 0);
// emulated FNO-like error: smooth blobs (~1% of peak) + ringing at copper edges + small white noise
function makeErr(level) {
  const e = new Float64Array(n);
  const blobs = Array.from({ length: 6 }, () => [R() * 64, R() * 64, 6 + R() * 10, (R() - .5) * 2]);
  for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
    const i = y * N + x; let v = 0;
    for (const [cx, cy, sg, a] of blobs) v += a * Math.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sg * sg));
    let g = 0; if (x > 0 && x < N - 1 && y > 0 && y < N - 1) g = Math.abs(b.phi[i + 1] - b.phi[i - 1]) + Math.abs(b.phi[i + N] - b.phi[i - N]);
    e[i] = level * pk * (0.01 * v + 0.004 * g * Math.sin(1.7 * x + 0.9 * y) + 0.0008 * TH.gauss(R));
  }
  return e;
}
const mg = TH.mgSetup(s);
function vcycles(rhs, k) { // approximate A^{-1} rhs with k multigrid V-cycles
  const st = TH.mgStepper({ ...s, b: rhs }); for (let i = 0; i < k; i++) st.step(); return Float64Array.from(st.x);
}
function certificate(mu, kmg) {
  const r = Ax(mu); for (let i = 0; i < n; i++) r[i] -= s.b[i];          // residual A mu - b = A e
  const absr = r.map(Math.abs);
  let w = vcycles(absr, kmg);                                              // cheap approximate supersolution
  const z = vcycles(new Float64Array(n).fill(1), kmg);                     // cheap approx of A^{-1} 1
  const Aw = Ax(w), Az = Ax(z);
  let m = Infinity, c = 0;
  for (let i = 0; i < n; i++) { m = Math.min(m, Az[i]); }
  if (!(m > 0)) return null;
  for (let i = 0; i < n; i++) c = Math.max(c, (absr[i] - Aw[i]) / m);
  c = Math.max(0, c) * 1.000001;
  for (let i = 0; i < n; i++) w[i] += c * z[i];
  // verify A w >= |r| (the only thing the guarantee needs)
  const Aw2 = Ax(w); let ok = true; for (let i = 0; i < n; i++) if (Aw2[i] < absr[i] - 1e-12 * (1 + absr[i])) ok = false;
  return { w, ok, absr };
}
const rows = [];
for (const level of [1, 3]) {
  const e0 = makeErr(level);
  for (const sweeps of [0, 2, 5, 10]) {
    const mu = Float64Array.from(xs); for (let i = 0; i < n; i++) mu[i] += e0[i];
    // optional cheap smoothing of the prediction: red-black Gauss-Seidel sweeps (one SpMV each)
    const sm = TH.sorStepper(s, mu, 1.0); for (let k = 0; k < sweeps; k++) sm.step(); const mu2 = sm.x;
    const e = mu2.map((v, i) => v - xs[i]), einf = maxAbs(e);
    const peakErr = Math.abs(Math.max(...mu2) - pk);
    // Varah (row-dominance) bound: ||A^-1||_inf <= 1/min_i (a_ii - sum|a_ij|) = 1/min(H dx^2 + dirichlet)
    const r = Ax(mu2); for (let i = 0; i < n; i++) r[i] -= s.b[i];
    let marg = Infinity; for (let i = 0; i < n; i++) marg = Math.min(marg, s.Hd[i] + s.dterm[i]);
    const varah = maxAbs(r) / marg;
    // ideal supersolution w* = A^{-1}|r| (exact solve) = best possible certificate of this type
    const wst = TH.pcg({ ...s, b: r.map(Math.abs) }, null, 1e-12, 6000).x, ideal = Math.max(...wst);
    const c2 = certificate(mu2, 2), c4 = certificate(mu2, 4);
    const imax = mu2.indexOf(Math.max(...mu2));
    const peakInt = c2 ? Math.max(...mu2.map((v, i) => v + c2.w[i])) - (mu2[imax] - c2.w[imax]) : NaN;
    rows.push({ errLevel: level + '%', presmooth: sweeps, trueMaxErr: einf.toFixed(3), truePeakErr: peakErr.toFixed(3), varah: varah.toExponential(1), idealCert: ideal.toFixed(3), cert2V: c2 ? Math.max(...c2.w).toFixed(3) : 'fail', cert4V: c4 ? Math.max(...c4.w).toFixed(3) : 'fail', valid: c2 && c4 ? (c2.ok && c4.ok) : false, ratio2V: c2 ? (Math.max(...c2.w) / einf).toFixed(1) : '-', peakIntervalWidth2V: peakInt.toFixed(3) });
  }
}
console.log('peak theta', pk.toFixed(2), 'K');
console.table(rows);

// --- cost-aware variant: the surrogate supplies w ≈ A^{-1}|r| and z ≈ A^{-1}1 (emulated with smooth multiplicative network errors)
console.log('\nNetwork-supplied certificate fields (validity enforced by 2 SpMVs + fix-up):');
const rows2 = [];
for (const netErr of [0.05, 0.10, 0.20]) {
  for (const sweeps of [5, 10]) {
    const e0 = makeErr(1), mu = Float64Array.from(xs); for (let i = 0; i < n; i++) mu[i] += e0[i];
    const sm = TH.sorStepper(s, mu, 1.0); for (let k = 0; k < sweeps; k++) sm.step(); const mu2 = sm.x;
    const einf = maxAbs(mu2.map((v, i) => v - xs[i]));
    const r = Ax(mu2); for (let i = 0; i < n; i++) r[i] -= s.b[i]; const absr = r.map(Math.abs);
    const wex = TH.pcg({ ...s, b: absr }, null, 1e-12, 6000).x, zex = TH.pcg({ ...s, b: new Float64Array(n).fill(1) }, null, 1e-12, 6000).x;
    const pert = (v, a) => { const ph = [R() * 7, R() * 7]; return v.map((x, i) => x * (1 + a * Math.sin((i % N) / 9 + ph[0]) * Math.cos(((i / N) | 0) / 11 + ph[1]))); };
    const w = pert(wex, netErr), z = pert(zex, netErr);
    const Aw = Ax(w), Az = Ax(z); let m = Infinity, c = 0;
    for (let i = 0; i < n; i++) m = Math.min(m, Az[i]);
    for (let i = 0; i < n; i++) c = Math.max(c, (absr[i] - Aw[i]) / m);
    c = Math.max(0, c); const wc = w.map((x, i) => x + c * z[i]);
    const Aw2 = Ax(wc); let ok = m > 0; for (let i = 0; i < n; i++) if (Aw2[i] < absr[i] * (1 - 1e-9)) ok = false;
    rows2.push({ netFieldError: (netErr * 100) + '%', presmooth: sweeps, trueMaxErr: einf.toFixed(3), idealCert: Math.max(...wex).toFixed(3), learnedCert: Math.max(...wc).toFixed(3), valid: ok, ratio: (Math.max(...wc) / einf).toFixed(2), minAz_positive: m > 0 });
  }
}
console.table(rows2);

console.log('\nNetwork fields polished by k red-black Gauss-Seidel sweeps before the 2-SpMV check:');
const rows3 = [];
function polish(x0, rhs, k) { const st = TH.sorStepper({ ...s, b: rhs }, x0, 1.0); for (let i = 0; i < k; i++) st.step(); return Float64Array.from(st.x); }
for (const netErr of [0.10, 0.20]) for (const k of [5, 10, 20]) {
  let okCount = 0, ratios = [], certs = [];
  for (let trial = 0; trial < 20; trial++) {
    const e0 = makeErr(1), mu = Float64Array.from(xs); for (let i = 0; i < n; i++) mu[i] += e0[i];
    const sm = TH.sorStepper(s, mu, 1.0); for (let j = 0; j < 5; j++) sm.step(); const mu2 = sm.x;
    const einf = maxAbs(mu2.map((v, i) => v - xs[i]));
    const r = Ax(mu2); for (let i = 0; i < n; i++) r[i] -= s.b[i]; const absr = r.map(Math.abs), one = new Float64Array(n).fill(1);
    const wex = TH.pcg({ ...s, b: absr }, null, 1e-12, 6000).x, zex = TH.pcg({ ...s, b: one }, null, 1e-12, 6000).x;
    const pert = (v, a) => { const ph = [R() * 7, R() * 7]; return v.map((x, i) => x * (1 + a * Math.sin((i % N) / 9 + ph[0]) * Math.cos(((i / N) | 0) / 11 + ph[1]))); };
    const w = polish(pert(wex, netErr), absr, k), z = polish(pert(zex, netErr), one, k);
    const Aw = Ax(w), Az = Ax(z); let m = Infinity, c = 0;
    for (let i = 0; i < n; i++) m = Math.min(m, Az[i]);
    if (!(m > 0)) continue;
    for (let i = 0; i < n; i++) c = Math.max(c, (absr[i] - Aw[i]) / m);
    const wc = w.map((x, i) => x + Math.max(0, c) * z[i]);
    const Aw2 = Ax(wc); let ok = true; for (let i = 0; i < n; i++) if (Aw2[i] < absr[i] * (1 - 1e-9)) ok = false;
    if (ok) { okCount++; ratios.push(Math.max(...wc) / einf); certs.push(Math.max(...wc)); }
  }
  const med = a => a.length ? a.slice().sort((x, y) => x - y)[Math.floor(a.length / 2)] : NaN;
  rows3.push({ netFieldError: (netErr * 100) + '%', polishSweeps: k, certificatesIssued: okCount + '/20', medianCertK: med(certs).toFixed(3), medianTightness: med(ratios).toFixed(2), worstTightness: ratios.length ? Math.max(...ratios).toFixed(2) : '-', costVsFullMG: ((5 + 2 * k + 3) / (11 * 7) * 100).toFixed(0) + '%' });
}
console.table(rows3);
