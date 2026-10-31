/* ===== Core numerics: TwinHeat-2D fin equation, cell-centred FV, 64x64 ===== */
const TH = (() => {
  const L = 0.1;                 // board side (m)
  const T_FR4 = 1.6e-3, K_FR4 = 0.9;
  const K_CU = 390;
  const G_LAM = K_FR4 * T_FR4;   // 1.44e-3 W/K
  function rng(seed) { let s = seed >>> 0 || 1; return () => { s ^= s << 13; s >>>= 0; s ^= s >>> 17; s ^= s << 5; s >>>= 0; return s / 4294967296; }; }
  function gauss(r) { let u = 0, v = 0; while (u === 0) u = r(); v = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }

  class Board {
    constructor(N = 64) {
      this.N = N; this.dx = L / N;
      const n = N * N;
      this.phi = new Float32Array(n);      // copper area fraction
      this.hmap = new Float32Array(n);     // total face loss H (W/m2K)
      this.chips = [];
      this.edges = [true, false, false, false]; // left, top, right, bottom heat-sunk
      this.hBase = 24; this.cuOz = 2; this.forced = null; // forced zone {x0,y0,x1,y1,h}
    }
    clone() {
      const b = new Board(this.N); b.phi.set(this.phi); b.chips = this.chips.map(c => ({ ...c }));
      b.edges = this.edges.slice(); b.hBase = this.hBase; b.cuOz = this.cuOz; b.forced = this.forced ? { ...this.forced } : null; return b;
    }
    gCu() { return K_CU * 17.5e-6 * this.cuOz; }
    // assemble coefficient arrays
    assemble() {
      const N = this.N, n = N * N, dx = this.dx;
      const G = new Float64Array(n), q = new Float64Array(n), Hd = new Float64Array(n);
      const gcu = this.gCu();
      for (let i = 0; i < n; i++) G[i] = G_LAM + gcu * this.phi[i];
      for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
        let h = this.hBase;
        const f = this.forced;
        if (f && x >= f.x0 && x < f.x1 && y >= f.y0 && y < f.y1) h = f.h;
        this.hmap[y * N + x] = h; Hd[y * N + x] = h * dx * dx;
      }
      // chips: power spread over footprint (cells, with fractional coverage)
      for (const c of this.chips) {
        const area = (c.w * L / N) * (c.h * L / N);
        const qd = c.P / area; // W/m2
        const x0 = c.x - c.w / 2, x1 = c.x + c.w / 2, y0 = c.y - c.h / 2, y1 = c.y + c.h / 2;
        for (let y = Math.max(0, Math.floor(y0)); y < Math.min(N, Math.ceil(y1)); y++)
          for (let x = Math.max(0, Math.floor(x0)); x < Math.min(N, Math.ceil(x1)); x++) {
            const ox = Math.min(x + 1, x1) - Math.max(x, x0), oy = Math.min(y + 1, y1) - Math.max(y, y0);
            if (ox > 0 && oy > 0) q[y * N + x] += qd * ox * oy * dx * dx;
          }
      }
      const gE = new Float64Array(n), gN = new Float64Array(n), dterm = new Float64Array(n);
      for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
        const i = y * N + x;
        if (x < N - 1) { const a = G[i], b = G[i + 1]; gE[i] = 2 * a * b / (a + b); }
        if (y < N - 1) { const a = G[i], b = G[i + N]; gN[i] = 2 * a * b / (a + b); }
        let d = 0;
        if (x === 0 && this.edges[0]) d += 2 * G[i];
        if (y === N - 1 && this.edges[1]) d += 2 * G[i];
        if (x === N - 1 && this.edges[2]) d += 2 * G[i];
        if (y === 0 && this.edges[3]) d += 2 * G[i];
        dterm[i] = d;
      }
      const sys = { N, n, G, b: q, Hd, gE, gN, dterm, diag: new Float64Array(n) };
      for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
        const i = y * N + x;
        let s = Hd[i] + dterm[i] + gE[i] + gN[i];
        if (x > 0) s += gE[i - 1];
        if (y > 0) s += gN[i - N];
        sys.diag[i] = s;
      }
      return sys;
    }
  }

  function applyA(s, x, out) {
    const { N, gE, gN, diag } = s;
    for (let y = 0; y < N; y++) {
      const r = y * N;
      for (let xx = 0; xx < N; xx++) {
        const i = r + xx;
        let v = diag[i] * x[i];
        if (xx < N - 1) v -= gE[i] * x[i + 1];
        if (xx > 0) v -= gE[i - 1] * x[i - 1];
        if (y < N - 1) v -= gN[i] * x[i + N];
        if (y > 0) v -= gN[i - N] * x[i - N];
        out[i] = v;
      }
    }
    return out;
  }
  function dot(a, b) { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; }
  function norm(a) { return Math.sqrt(dot(a, a)); }
  function resid(s, x, r) { applyA(s, x, r); for (let i = 0; i < s.n; i++) r[i] = s.b[i] - r[i]; return r; }

  // Jacobi-preconditioned CG. onIter(it, x, relres) optional (return true to stop)
  function pcg(s, x0, tol = 1e-8, maxit = 2000, onIter) {
    const n = s.n, x = x0 ? Float64Array.from(x0) : new Float64Array(n);
    const r = new Float64Array(n), z = new Float64Array(n), p = new Float64Array(n), Ap = new Float64Array(n);
    resid(s, x, r);
    const bn = norm(s.b) || 1;
    for (let i = 0; i < n; i++) { z[i] = r[i] / s.diag[i]; p[i] = z[i]; }
    let rz = dot(r, z), it = 0, rel = norm(r) / bn;
    if (onIter) onIter(0, x, rel);
    while (rel > tol && it < maxit) {
      applyA(s, p, Ap);
      const a = rz / dot(p, Ap);
      for (let i = 0; i < n; i++) { x[i] += a * p[i]; r[i] -= a * Ap[i]; }
      it++; rel = norm(r) / bn;
      if (onIter && onIter(it, x, rel)) break;
      if (rel <= tol) break;
      for (let i = 0; i < n; i++) z[i] = r[i] / s.diag[i];
      const rzn = dot(r, z), beta = rzn / rz; rz = rzn;
      for (let i = 0; i < n; i++) p[i] = z[i] + beta * p[i];
    }
    return { x, it, rel };
  }
  // stateful PCG stepper for animations
  function pcgStepper(s, x0) {
    const n = s.n, x = x0 ? Float64Array.from(x0) : new Float64Array(n);
    const r = new Float64Array(n), z = new Float64Array(n), p = new Float64Array(n), Ap = new Float64Array(n);
    resid(s, x, r); const bn = norm(s.b) || 1;
    for (let i = 0; i < n; i++) { z[i] = r[i] / s.diag[i]; p[i] = z[i]; }
    let rz = dot(r, z);
    const st = { x, r, it: 0, rel: norm(r) / bn, step() {
      applyA(s, p, Ap); const a = rz / dot(p, Ap);
      for (let i = 0; i < n; i++) { x[i] += a * p[i]; r[i] -= a * Ap[i]; }
      for (let i = 0; i < n; i++) z[i] = r[i] / s.diag[i];
      const rzn = dot(r, z), beta = rzn / rz; rz = rzn;
      for (let i = 0; i < n; i++) p[i] = z[i] + beta * p[i];
      st.it++; st.rel = norm(r) / bn; return st.rel;
    } };
    return st;
  }
  function jacobiStepper(s, x0) {
    const n = s.n, x = x0 ? Float64Array.from(x0) : new Float64Array(n), r = new Float64Array(n), bn = norm(s.b) || 1;
    const st = { x, it: 0, rel: 1, step() {
      resid(s, x, r); for (let i = 0; i < n; i++) x[i] += r[i] / s.diag[i];
      st.it++; resid(s, x, r); st.rel = norm(r) / bn; return st.rel;
    } };
    resid(s, x, r); st.rel = norm(r) / bn; return st;
  }
  function rbSweep(s, x, b, omega) {
    const { N, gE, gN, diag } = s;
    for (let color = 0; color < 2; color++)
      for (let y = 0; y < N; y++) for (let xx = (y + color) & 1; xx < N; xx += 2) {
        const i = y * N + xx;
        let sum = b[i];
        if (xx < N - 1) sum += gE[i] * x[i + 1];
        if (xx > 0) sum += gE[i - 1] * x[i - 1];
        if (y < N - 1) sum += gN[i] * x[i + N];
        if (y > 0) sum += gN[i - N] * x[i - N];
        x[i] += omega * (sum / diag[i] - x[i]);
      }
  }
  function sorStepper(s, x0, omega = 1.0) {
    const n = s.n, x = x0 ? Float64Array.from(x0) : new Float64Array(n), r = new Float64Array(n), bn = norm(s.b) || 1;
    const st = { x, it: 0, rel: 1, step() { rbSweep(s, x, s.b, omega); st.it++; resid(s, x, r); st.rel = norm(r) / bn; return st.rel; } };
    resid(s, x, r); st.rel = norm(r) / bn; return st;
  }
  // ---- geometric multigrid (cell-centred, rediscretised coarse operators) ----
  function coarsen(s) {
    const N = s.N, M = N / 2, n = M * M;
    const c = { N: M, n, gE: new Float64Array(n), gN: new Float64Array(n), Hd: new Float64Array(n), dterm: new Float64Array(n), diag: new Float64Array(n) };
    for (let J = 0; J < M; J++) for (let I = 0; I < M; I++) {
      const k = J * M + I, f = (2 * J) * N + 2 * I;
      if (I < M - 1) c.gE[k] = 0.5 * (s.gE[f + 1] + s.gE[f + 1 + N]);
      if (J < M - 1) c.gN[k] = 0.5 * (s.gN[f + N] + s.gN[f + N + 1]);
      c.Hd[k] = s.Hd[f] + s.Hd[f + 1] + s.Hd[f + N] + s.Hd[f + N + 1];
      c.dterm[k] = 0.5 * (s.dterm[f] + s.dterm[f + 1] + s.dterm[f + N] + s.dterm[f + N + 1]);
    }
    for (let J = 0; J < M; J++) for (let I = 0; I < M; I++) {
      const k = J * M + I; let d = c.Hd[k] + c.dterm[k] + c.gE[k] + c.gN[k];
      if (I > 0) d += c.gE[k - 1]; if (J > 0) d += c.gN[k - M]; c.diag[k] = d;
    }
    return c;
  }
  function mgSetup(s) { const lv = [s]; while (lv[lv.length - 1].N > 4) lv.push(coarsen(lv[lv.length - 1])); return lv; }
  function vcycle(lv, l, x, b) {
    const s = lv[l];
    if (l === lv.length - 1) { for (let k = 0; k < 40; k++) rbSweep(s, x, b, 1); return; }
    rbSweep(s, x, b, 1); rbSweep(s, x, b, 1);
    const r = new Float64Array(s.n); applyA(s, x, r); for (let i = 0; i < s.n; i++) r[i] = b[i] - r[i];
    const c = lv[l + 1], M = c.N, N = s.N, rc = new Float64Array(c.n), ec = new Float64Array(c.n);
    for (let J = 0; J < M; J++) for (let I = 0; I < M; I++) { const f = 2 * J * N + 2 * I; rc[J * M + I] = r[f] + r[f + 1] + r[f + N] + r[f + N + 1]; }
    vcycle(lv, l + 1, ec, rc);
    // bilinear-style prolongation (cell centred: weights 9/16,3/16,3/16,1/16)
    for (let y = 0; y < N; y++) for (let xx = 0; xx < N; xx++) {
      const I = xx >> 1, J = y >> 1;
      const I2 = Math.min(M - 1, Math.max(0, I + ((xx & 1) ? 1 : -1))), J2 = Math.min(M - 1, Math.max(0, J + ((y & 1) ? 1 : -1)));
      x[y * N + xx] += (9 * ec[J * M + I] + 3 * ec[J * M + I2] + 3 * ec[J2 * M + I] + ec[J2 * M + I2]) / 16;
    }
    rbSweep(s, x, b, 1); rbSweep(s, x, b, 1);
  }
  function mgStepper(s, x0) {
    const lv = mgSetup(s), n = s.n, x = x0 ? Float64Array.from(x0) : new Float64Array(n), r = new Float64Array(n), bn = norm(s.b) || 1;
    const st = { x, it: 0, rel: 1, step() { vcycle(lv, 0, x, s.b); st.it++; resid(s, x, r); st.rel = norm(r) / bn; return st.rel; } };
    resid(s, x, r); st.rel = norm(r) / bn; return st;
  }
  // flux F = -G grad(theta), conservative face average, per unit width (W/m)
  function flux(s, th, dx) {
    const N = s.N, n = s.n, Fx = new Float32Array(n), Fy = new Float32Array(n);
    for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
      const i = y * N + x;
      const fe = x < N - 1 ? s.gE[i] * (th[i] - th[i + 1]) : 0;
      const fw = x > 0 ? s.gE[i - 1] * (th[i - 1] - th[i]) : 0;
      const fn = y < N - 1 ? s.gN[i] * (th[i] - th[i + N]) : 0;
      const fs = y > 0 ? s.gN[i - N] * (th[i - N] - th[i]) : 0;
      Fx[i] = 0.5 * (fe + fw) / dx; Fy[i] = 0.5 * (fn + fs) / dx;
    }
    return { Fx, Fy };
  }
  function energyBalance(s, th) {
    let pin = 0, pout = 0;
    for (let i = 0; i < s.n; i++) { pin += s.b[i]; pout += (s.Hd[i] + s.dterm[i]) * th[i]; }
    return { pin, pout, err: Math.abs(pin - pout) / (pin || 1) };
  }
  // ---- radix-2 FFT (in place) ----
  function fft(re, im, inv) {
    const n = re.length;
    for (let i = 1, j = 0; i < n; i++) { let bit = n >> 1; for (; j & bit; bit >>= 1) j ^= bit; j ^= bit; if (i < j) { [re[i], re[j]] = [re[j], re[i]]; [im[i], im[j]] = [im[j], im[i]]; } }
    for (let len = 2; len <= n; len <<= 1) {
      const ang = 2 * Math.PI / len * (inv ? 1 : -1), wr = Math.cos(ang), wi = Math.sin(ang);
      for (let i = 0; i < n; i += len) {
        let cr = 1, ci = 0;
        for (let j = 0; j < len / 2; j++) {
          const a = i + j, b = a + len / 2, tr = re[b] * cr - im[b] * ci, ti = re[b] * ci + im[b] * cr;
          re[b] = re[a] - tr; im[b] = im[a] - ti; re[a] += tr; im[a] += ti;
          const ncr = cr * wr - ci * wi; ci = cr * wi + ci * wr; cr = ncr;
        }
      }
    }
    if (inv) for (let i = 0; i < n; i++) { re[i] /= n; im[i] /= n; }
  }
  function fft2(re, im, N, inv) {
    const r = new Float64Array(N), c = new Float64Array(N);
    for (let y = 0; y < N; y++) { for (let x = 0; x < N; x++) { r[x] = re[y * N + x]; c[x] = im[y * N + x]; } fft(r, c, inv); for (let x = 0; x < N; x++) { re[y * N + x] = r[x]; im[y * N + x] = c[x]; } }
    for (let x = 0; x < N; x++) { for (let y = 0; y < N; y++) { r[y] = re[y * N + x]; c[y] = im[y * N + x]; } fft(r, c, inv); for (let y = 0; y < N; y++) { re[y * N + x] = r[y]; im[y * N + x] = c[y]; } }
  }
  // keep modes |kx|,|ky| < K (FNO-style truncation)
  function truncate(field, N, K) {
    const re = Float64Array.from(field), im = new Float64Array(N * N);
    fft2(re, im, N, false);
    const spec = new Float64Array(N * N);
    for (let i = 0; i < N * N; i++) spec[i] = Math.hypot(re[i], im[i]);
    for (let ky = 0; ky < N; ky++) for (let kx = 0; kx < N; kx++) {
      const fx = kx < N / 2 ? kx : N - kx, fy = ky < N / 2 ? ky : N - ky;
      if (fx >= K || fy >= K) { re[ky * N + kx] = 0; im[ky * N + kx] = 0; }
    }
    fft2(re, im, N, true);
    return { rec: re, spec };
  }

  function defaultBoard() {
    const b = new Board(64);
    const N = 64;
    // ground pour bottom-left and a spreader under CPU, a few traces
    for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
      let p = 0;
      if (x < 22 && y > 34) p = 1;                       // pour near heat-sunk edge
      if (x >= 18 && x < 34 && y >= 20 && y < 36) p = 1; // spreader under CPU
      if (y >= 27 && y < 29 && x >= 34 && x < 50) p = 1; // trace to PMIC
      if (x >= 46 && x < 48 && y >= 12 && y < 28) p = 1;
      if (y >= 46 && y < 48 && x >= 22 && x < 56) p = 1;
      b.phi[y * N + x] = p;
    }
    b.chips = [
      { id: 'U1', name: 'CPU', x: 26, y: 28, w: 10, h: 10, P: 1.2 },
      { id: 'U2', name: 'PMIC', x: 50, y: 18, w: 7, h: 7, P: 0.8 },
      { id: 'U3', name: 'LDO', x: 46, y: 50, w: 5, h: 4, P: 0.5 },
    ];
    return b;
  }
  return { Board, L, G_LAM, rng, gauss, applyA, pcg, pcgStepper, jacobiStepper, sorStepper, mgStepper, mgSetup, resid, norm, dot, flux, energyBalance, fft, fft2, truncate, defaultBoard };
})();
if (typeof module !== 'undefined') module.exports = TH;
