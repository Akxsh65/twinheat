const TH = require('./core.js');
const N = 64, n = N * N;
function run(board, label, errLevel, seed) {
  const s = board.assemble(), xs = TH.pcg(s, null, 1e-13, 8000).x, pk = Math.max(...xs), R = TH.rng(seed);
  const Ax = v => TH.applyA(s, v, new Float64Array(n)), maxAbs = v => v.reduce((m, x) => Math.max(m, Math.abs(x)), 0);
  // positive vector z ≈ A^{-1}1 from 3 multigrid V-cycles (cheap, low accuracy), checked once
  const zs = TH.mgStepper({ ...s, b: new Float64Array(n).fill(1) }); for (let i = 0; i < 3; i++) zs.step(); const z = zs.x, Az = Ax(z);
  const minAz = Math.min(...Az), invNorm = Math.max(...z) / minAz;           // rigorous bound on ||A^{-1}||_inf if minAz > 0
  const trueInv = Math.max(...TH.pcg({ ...s, b: new Float64Array(n).fill(1) }, null, 1e-12, 8000).x);
  // warm start: exact + smooth error at errLevel of peak
  const mu = xs.map((v, i) => v + errLevel * pk * Math.sin((i % N) / 7 + R() * 6) * Math.cos(((i / N) | 0) / 9 + 1));
  const st = TH.pcgStepper(s, mu); let it1K = null, it01K = null, itTight = null, ratioAt1K = null;
  const tol8 = () => st.rel <= 1e-8;
  while (!tol8() && st.it < 5000) {
    st.step();
    const rinf = maxAbs(st.r), bound = rinf * invNorm, einf = maxAbs(st.x.map((v, i) => v - xs[i]));
    if (it1K === null && bound <= 1.0) { it1K = st.it; ratioAt1K = bound / Math.max(einf, 1e-12); }
    if (it01K === null && bound <= 0.1) it01K = st.it;
  }
  return { board: label, peakK: pk.toFixed(1), minAz_pos: minAz > 0, boundInvNorm_vs_true: (invNorm / trueInv).toFixed(3), itersTo1e8: st.it, itersCert1K: it1K, itersCert0_1K: it01K, savedAt1K: ((1 - it1K / st.it) * 100).toFixed(0) + '%', boundOverTrueErr: ratioAt1K && ratioAt1K.toFixed(0) };
}
const rows = [];
const b0 = TH.defaultBoard(); rows.push(run(b0, 'default, 1 heat-sunk edge', 0.02, 3));
const b1 = TH.defaultBoard(); b1.edges = [false, false, false, false]; rows.push(run(b1, 'no heat sink (worst conditioning)', 0.02, 4));
const b2 = TH.defaultBoard(); b2.chips.push({ id: 'U4', name: 'U4', x: 8, y: 56, w: 6, h: 6, P: 0.6 }); rows.push(run(b2, '4 chips, chip near corner', 0.05, 5));
const b3 = TH.defaultBoard(); b3.cuOz = 0.5; b3.hBase = 12; rows.push(run(b3, 'thin copper, weak convection', 0.02, 6));
console.table(rows);
