"""M-matrix checker for assembled operators (invariant B1, CLAUDE.md §5; Lemma 2, §8.4).

`check_m_matrix(A)` tests the hypotheses under which A is a nonsingular M-matrix, which is
what the certificate (Lemma 2) needs: A⁻¹ ≥ 0 entrywise. Run it on every assembly.

The four checks, and the part of Lemma 2 that needs each:
  1. Off-diagonals ≤ 0 (A is a Z-matrix). With 2–4 this gives A⁻¹ ≥ 0, which is used in
     (a) A⁻¹1 ≤ z/min(Az), (b) e = A⁻¹r, and (c) |A⁻¹r| ≤ A⁻¹|r| ≤ A⁻¹Aw = w.
  2. Diagonals > 0. A Z-matrix with a positive diagonal and weak diagonal dominance (check 3)
     is an M-matrix; a nonpositive diagonal makes the dominance test meaningless.
  3. Every row sum ≥ −1e-12·|diag| (weak diagonal dominance) and at least one row sum
     > 1e-12·|diag| (strict dominance somewhere, e.g. convection H > 0 or a heat-sunk edge).
  4. The off-diagonal coupling graph is connected (strongly, which equals connected for the
     symmetric patterns our assemblies have). With 3 this makes A irreducibly diagonally
     dominant, so a single strict row makes the whole matrix nonsingular. Without it, a block
     with no strict row (e.g. a pure-Neumann island) would be singular.

These are sufficient conditions, not necessary ones: a triangular Z-matrix is an M-matrix but is
rejected here as reducible. That is intended, since the certificate's proof uses this form.

Floating-point note: the 1e-12 row-sum tolerance is relative to each row's diagonal, so a
matrix assembled in FP64 passes; FP32 assemblies should be checked after casting to FP64.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

__all__ = ["MMatrixError", "assert_m_matrix", "check_m_matrix"]

ROW_SUM_TOL = 1e-12
MAX_WORST_ROWS = 10


class MMatrixError(AssertionError):
    """A matrix violates the M-matrix hypotheses. `report` is the dict from check_m_matrix."""

    def __init__(self, message: str, report: dict[str, Any]) -> None:
        super().__init__(message)
        self.report = report


def _to_clean_csr(A: sp.spmatrix | sp.sparray) -> sp.csr_matrix:
    """A float64 CSR copy of A without explicit zeros; raises on bad input."""
    if not sp.issparse(A):
        raise TypeError(f"expected a scipy sparse matrix, got {type(A).__name__}")
    n_rows, n_cols = A.shape
    if n_rows != n_cols:
        raise ValueError(f"matrix must be square, got shape {A.shape}")
    if n_rows == 0:
        raise ValueError("matrix is empty (0 x 0)")
    B = sp.csr_matrix(A, dtype=np.float64, copy=True)
    B.sum_duplicates()
    bad = ~np.isfinite(B.data)
    if bad.any():
        first = int(np.flatnonzero(bad)[0])
        row = int(np.searchsorted(B.indptr, first, side="right") - 1)
        raise ValueError(f"matrix has {int(bad.sum())} non-finite entries; first one in row {row}")
    B.eliminate_zeros()
    return B


def check_m_matrix(A: sp.spmatrix | sp.sparray) -> dict[str, Any]:
    """Check the M-matrix hypotheses of Lemma 2 on a square sparse matrix.

    Returns {"ok", "n_pos_offdiag", "n_nonpos_diag", "n_neg_rowsum", "n_strict_rows",
    "n_components", "worst_rows"}. `ok` is True iff there is no positive off-diagonal entry, no
    nonpositive diagonal, no row sum below −1e-12·|diag|, at least one row sum above
    1e-12·|diag|, and the coupling graph has one component. `worst_rows` lists up to 10 rows
    as {"row", "reason", "value"}: positive off-diagonals (value = largest one in the row),
    then nonpositive diagonals (value = the diagonal), then negative row sums (value = the sum),
    each worst first.

    Raises TypeError for a dense array, ValueError for a non-square, empty or non-finite
    matrix. The input is not modified.
    """
    B = _to_clean_csr(A)
    n = B.shape[0]
    diag = B.diagonal()
    row_sum = np.asarray(B.sum(axis=1)).ravel()
    tol = ROW_SUM_TOL * np.abs(diag)

    coo = B.tocoo()
    off = coo.row != coo.col
    rows, cols, vals = coo.row[off], coo.col[off], coo.data[off]

    positive = vals > 0
    max_pos = np.zeros(n)
    np.maximum.at(max_pos, rows[positive], vals[positive])
    nonpos_diag = diag <= 0
    neg_row_sum = row_sum < -tol
    strict = row_sum > tol

    coupling = sp.csr_matrix((np.ones(vals.size), (rows, cols)), shape=(n, n))
    n_components, _ = connected_components(coupling, directed=True, connection="strong")

    report: dict[str, Any] = {
        "n_pos_offdiag": int(positive.sum()),
        "n_nonpos_diag": int(nonpos_diag.sum()),
        "n_neg_rowsum": int(neg_row_sum.sum()),
        "n_strict_rows": int(strict.sum()),
        "n_components": int(n_components),
    }
    report["ok"] = bool(
        report["n_pos_offdiag"] == 0
        and report["n_nonpos_diag"] == 0
        and report["n_neg_rowsum"] == 0
        and report["n_strict_rows"] >= 1
        and report["n_components"] == 1
    )
    report["worst_rows"] = _worst_rows(max_pos, diag, row_sum, tol, nonpos_diag, neg_row_sum)
    return {key: report[key] for key in (
        "ok", "n_pos_offdiag", "n_nonpos_diag", "n_neg_rowsum",
        "n_strict_rows", "n_components", "worst_rows",
    )}  # fmt: skip


def _worst_rows(
    max_pos: np.ndarray,
    diag: np.ndarray,
    row_sum: np.ndarray,
    tol: np.ndarray,
    nonpos_diag: np.ndarray,
    neg_row_sum: np.ndarray,
) -> list[dict[str, Any]]:
    worst: list[dict[str, Any]] = []
    pos_rows = np.flatnonzero(max_pos > 0)
    for i in pos_rows[np.argsort(-max_pos[pos_rows], kind="stable")]:
        worst.append({"row": int(i), "reason": "positive_offdiag", "value": float(max_pos[i])})
    bad_diag = np.flatnonzero(nonpos_diag)
    for i in bad_diag[np.argsort(diag[bad_diag], kind="stable")]:
        worst.append({"row": int(i), "reason": "nonpositive_diag", "value": float(diag[i])})
    bad_sum = np.flatnonzero(neg_row_sum)
    relative = row_sum[bad_sum] / np.maximum(np.abs(diag[bad_sum]), np.finfo(float).tiny)
    for i in bad_sum[np.argsort(relative, kind="stable")]:
        worst.append({"row": int(i), "reason": "negative_rowsum", "value": float(row_sum[i])})
    return worst[:MAX_WORST_ROWS]


def assert_m_matrix(A: sp.spmatrix | sp.sparray, context: str | None = None) -> dict[str, Any]:
    """Return the check_m_matrix report if A passes; otherwise raise MMatrixError with it.

    `context` (for example "board 17, 4 layers") is put in the message to name the input.
    """
    report = check_m_matrix(A)
    if not report["ok"]:
        prefix = f"{context}: " if context else ""
        raise MMatrixError(
            f"{prefix}matrix is not a nonsingular M-matrix (hypotheses of Lemma 2 violated): "
            f"{report}",
            report,
        )
    return report
