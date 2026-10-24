"""Tests for solvers.mmatrix_test (invariant B1, CLAUDE.md §5; hypotheses of Lemma 2, §8.4)."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from solvers.mmatrix_test import MMatrixError, assert_m_matrix, check_m_matrix


def laplacian_1d(n: int, shift: float = 0.0, neumann_ends: bool = True) -> sp.csr_matrix:
    """1D Laplacian, free (Neumann) ends so every row sums to `shift`; diagonal + shift."""
    main = np.full(n, 2.0)
    if neumann_ends:
        main[0] = main[-1] = 1.0
    A = sp.diags([-np.ones(n - 1), main + shift, -np.ones(n - 1)], [-1, 0, 1], format="csr")
    return A


def laplacian_2d(nx: int, ny: int, h: float = 0.0, dirichlet_left: bool = False) -> sp.csr_matrix:
    """5-point Laplacian with free edges, plus H = h on the diagonal.

    dirichlet_left adds 2 on the diagonal of the x = 0 column (half-cell Dirichlet face).
    """
    Tx = laplacian_1d(nx)
    Ty = laplacian_1d(ny)
    A = (sp.kron(sp.identity(ny), Tx) + sp.kron(Ty, sp.identity(nx))).tolil()
    for k in range(nx * ny):
        A[k, k] += h
    if dirichlet_left:
        for y in range(ny):
            A[y * nx, y * nx] += 2.0
    return A.tocsr()


def layered_3d(n: int, layers: int, h_top: float, g_vert: float = 0.5) -> sp.csr_matrix:
    """7-point layered operator: in-plane Laplacians, vertical coupling, H only on layer 0."""
    L = laplacian_2d(n, n)
    chain = laplacian_1d(layers) * g_vert
    A = sp.kron(sp.identity(layers), L) + sp.kron(chain, sp.identity(n * n))
    h = sp.diags(np.r_[np.full(n * n, h_top), np.zeros((layers - 1) * n * n)])
    return (A + h).tocsr()


# ------------------------------------------------------------------ the four hand-built cases


def test_laplacian_with_positive_shift_is_m_matrix() -> None:
    r = check_m_matrix(laplacian_1d(10, shift=0.1))
    assert r["ok"] is True
    assert r["n_pos_offdiag"] == 0
    assert r["n_nonpos_diag"] == 0
    assert r["n_neg_rowsum"] == 0
    assert r["n_strict_rows"] == 10
    assert r["n_components"] == 1
    assert r["worst_rows"] == []


def test_one_positive_offdiagonal_fails() -> None:
    A = laplacian_1d(10, shift=0.1).tolil()
    A[3, 4] = 0.2
    r = check_m_matrix(A.tocsr())
    assert r["ok"] is False
    assert r["n_pos_offdiag"] == 1
    assert r["worst_rows"][0]["row"] == 3
    assert r["worst_rows"][0]["reason"] == "positive_offdiag"


def test_block_diagonal_of_two_valid_blocks_is_disconnected() -> None:
    B = laplacian_1d(6, shift=0.1)
    r = check_m_matrix(sp.block_diag([B, B], format="csr"))
    assert r["ok"] is False
    assert r["n_components"] == 2
    assert r["n_pos_offdiag"] == 0 and r["n_neg_rowsum"] == 0  # the other checks pass


def test_pure_neumann_laplacian_has_no_strict_row() -> None:
    r = check_m_matrix(laplacian_1d(10, shift=0.0))
    assert r["ok"] is False
    assert r["n_strict_rows"] == 0
    assert r["n_pos_offdiag"] == 0 and r["n_components"] == 1  # only the strict-row check fails


# ------------------------------------------------------------------ realistic matrices


def test_2d_plate_with_convection_is_m_matrix() -> None:
    assert check_m_matrix(laplacian_2d(12, 9, h=0.05))["ok"]


def test_2d_plate_without_convection_but_with_one_dirichlet_edge_is_m_matrix() -> None:
    r = check_m_matrix(laplacian_2d(12, 9, h=0.0, dirichlet_left=True))
    assert r["ok"]
    assert r["n_strict_rows"] == 9  # only the Dirichlet column is strictly dominant


def test_2d_plate_without_convection_or_dirichlet_edge_fails() -> None:
    assert not check_m_matrix(laplacian_2d(12, 9, h=0.0))["ok"]


def test_layered_operator_with_convection_on_top_layer_only_is_m_matrix() -> None:
    r = check_m_matrix(layered_3d(6, 4, h_top=0.05))
    assert r["ok"]
    assert r["n_strict_rows"] == 36  # only the convective layer


def test_layered_operator_without_any_h_or_dirichlet_fails() -> None:
    assert not check_m_matrix(layered_3d(6, 4, h_top=0.0))["ok"]


def test_nonsymmetric_matrix_is_accepted_when_strongly_connected() -> None:
    A = sp.csr_matrix(np.array([[2.0, -1.0, 0.0], [-0.5, 2.0, -1.0], [0.0, -0.2, 1.0]]))
    assert check_m_matrix(A)["ok"]


def test_triangular_matrix_is_rejected_as_reducible() -> None:
    """A triangular Z-matrix is an M-matrix but not irreducible; the checker is conservative."""
    A = sp.csr_matrix(np.array([[1.0, -0.5, 0.0], [0.0, 1.0, -0.5], [0.0, 0.0, 1.0]]))
    r = check_m_matrix(A)
    assert r["ok"] is False
    assert r["n_components"] == 3


def test_one_by_one_positive_matrix_is_m_matrix() -> None:
    assert check_m_matrix(sp.csr_matrix([[3.0]]))["ok"]


# ------------------------------------------------------------------ individual checks


def test_nonpositive_diagonal_is_counted() -> None:
    A = laplacian_1d(8, shift=0.1).tolil()
    A[2, 2] = 0.0
    A[5, 5] = -1.0
    r = check_m_matrix(A.tocsr())
    assert r["ok"] is False
    assert r["n_nonpos_diag"] == 2
    assert {w["row"] for w in r["worst_rows"] if w["reason"] == "nonpositive_diag"} == {2, 5}


def _with_row2_sum(relative_to_diag: float) -> sp.csr_matrix:
    """1D matrix whose interior row 2 has off-diagonals -1, -1 and row sum relative_to_diag * d."""
    A = laplacian_1d(6, shift=0.5).tolil()
    A[2, 2] = 2.0 / (1.0 - relative_to_diag)  # d - 2 = relative_to_diag * d
    return A.tocsr()


def test_row_sum_tolerance_is_1e_12_relative_to_diagonal() -> None:
    inside = check_m_matrix(_with_row2_sum(-1e-13))  # row sum = -1e-13 * d: within tolerance
    assert inside["n_neg_rowsum"] == 0
    outside = check_m_matrix(_with_row2_sum(-1e-9))  # row sum = -1e-9 * d: violation
    assert outside["n_neg_rowsum"] == 1
    assert outside["ok"] is False
    worst = outside["worst_rows"][0]
    assert worst["row"] == 2 and worst["reason"] == "negative_rowsum" and worst["value"] < 0


def test_explicit_zero_offdiagonals_do_not_count_as_edges() -> None:
    B = laplacian_1d(5, shift=0.1)
    A = sp.block_diag([B, B], format="lil")
    A[4, 5] = 0.0  # stored explicit zero must not connect the two blocks
    A = A.tocsr()
    assert check_m_matrix(A)["n_components"] == 2


def test_worst_rows_is_capped_and_sorted_worst_first() -> None:
    A = laplacian_1d(40, shift=0.1).tolil()
    for i in range(30):
        A[i, i + 1 if i + 1 < 40 else i - 1] = 0.1 * (i + 1)
    r = check_m_matrix(A.tocsr())
    assert len(r["worst_rows"]) <= 10
    values = [w["value"] for w in r["worst_rows"] if w["reason"] == "positive_offdiag"]
    assert values == sorted(values, reverse=True)


def test_input_matrix_is_not_modified() -> None:
    A = laplacian_1d(8, shift=0.1)
    A.data[3] = 0.0  # an explicit stored zero
    before = (A.data.copy(), A.indices.copy(), A.indptr.copy())
    check_m_matrix(A)
    assert np.array_equal(A.data, before[0])
    assert np.array_equal(A.indices, before[1])
    assert np.array_equal(A.indptr, before[2])


@pytest.mark.parametrize("fmt", ["csr", "csc", "coo", "lil"])
def test_accepts_any_scipy_sparse_format(fmt: str) -> None:
    assert check_m_matrix(laplacian_1d(8, shift=0.1).asformat(fmt))["ok"]


def test_float32_and_integer_matrices_work() -> None:
    A = laplacian_1d(8, shift=1.0)
    assert check_m_matrix(A.astype(np.float32))["ok"]
    assert check_m_matrix(sp.csr_matrix(np.array([[2, -1], [-1, 2]], dtype=np.int64)))["ok"]


# ------------------------------------------------------------------ errors fail loudly


def test_dense_array_is_rejected() -> None:
    with pytest.raises(TypeError, match="sparse"):
        check_m_matrix(np.eye(3))


def test_non_square_matrix_is_rejected() -> None:
    with pytest.raises(ValueError, match=r"square.*\(3, 4\)"):
        check_m_matrix(sp.csr_matrix(np.ones((3, 4))))


def test_empty_matrix_is_rejected() -> None:
    with pytest.raises(ValueError, match="empty"):
        check_m_matrix(sp.csr_matrix((0, 0)))


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_non_finite_entries_are_rejected_not_hidden(bad: float) -> None:
    A = laplacian_1d(6, shift=0.1).tolil()
    A[2, 2] = bad
    with pytest.raises(ValueError, match="non-finite.*row 2"):
        check_m_matrix(A.tocsr())


# ------------------------------------------------------------------ assert_m_matrix


def test_assert_m_matrix_passes_silently_and_returns_report() -> None:
    report = assert_m_matrix(laplacian_1d(8, shift=0.1))
    assert report["ok"] is True


def test_assert_m_matrix_raises_with_the_report() -> None:
    A = laplacian_1d(10, shift=0.0)
    with pytest.raises(MMatrixError, match="n_strict_rows") as info:
        assert_m_matrix(A)
    assert info.value.report["n_strict_rows"] == 0
    assert isinstance(info.value, AssertionError)


def test_assert_m_matrix_names_the_context_in_the_message() -> None:
    with pytest.raises(MMatrixError, match="board 17"):
        assert_m_matrix(laplacian_1d(10), context="board 17")


# ------------------------------------------------------------------ B1 on real boards (later)


@pytest.mark.slow
def test_b1_on_50_sampled_boards_per_configuration() -> None:
    """Placeholder: runs once solvers/fvm.py and data/sampler.py exist (CLAUDE.md §5, B1)."""
    pytest.importorskip("solvers.fvm", reason="solvers/fvm.py does not exist yet (task N1.1)")
    pytest.skip("B1 sweep over 50 boards x {1-layer, 4-layer} is written together with fvm.py")
