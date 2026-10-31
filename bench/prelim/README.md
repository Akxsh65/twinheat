# TwinHeat v3.1 — preliminary certificate scripts (bench/prelim)

Reproduces the two "Preliminary evidence" tables in section 03 of TwinHeat_plan_v3_1.html
on the 64×64 single-plate development model, with emulated surrogate errors.

    node stop.js   # Lemma 2(b): certified early stopping of warm-started PCG (4 board variants)
    node cert.js   # Lemma 2(c): pointwise certificates, Varah bound, and the failed learned-certificate test

Requires Node.js 18+. core.js holds the finite-volume assembly and the PCG, SOR and multigrid solvers.
These are toy results that justify the plan; experiment E13 repeats them on the 4-layer board with a trained FNO.
