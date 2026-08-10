"Dynamical matrix builder helper functions"
import numpy as np
import gemmi
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.ndimage import gaussian_filter1d

"""
Fast construction of the rigid-body dynamical matrix D(q) on a commensurate
q-grid, via real-space accumulation of 6x6 blocks followed by an FFT.

ASSUMED INPUTS
--------------
df_symmetry : one row per *unique* spring (no inter-cell double-counting),
    with columns:
        spring_class        - int, symmetry-equivalence class w
        cra1, cra2   - atom identifiers (unused directly here, kept for bookkeeping)
        sym_idx1, sym_idx2 - which symmetry op generates each atom's rigid body
        group_id1, group_id2 - which rigid body in the asymmetric unit (0,1,...)
        pbc_shift    - length-3 int array/tuple, lattice-vector offset from
                       cra1's cell to cra2's cell
        r1, r2       - length-3 Cartesian coordinates of the two spring endpoints

df_atoms : one row per atom in the unit cell (ALL M rigid bodies), needed to
    compute each body's center of mass and generalized mass matrix. Columns:
        sym_idx, group_id, mass, x, y, z
    This is NOT contained in df_symmetry (which only carries the two bonded
    atoms per row) and must be supplied separately.

lattice : 3x3 array, rows = real-space lattice vectors a1, a2, a3 (Cartesian),
    used to convert integer pbc_shift into a Cartesian offset vector.

grid_shape : (N1, N2, N3), the Born-von Karman supercell / q-grid dimensions.
    pbc_shift components must be well within this range (spring cutoff should
    be much smaller than the supercell in every direction, or wraparound will
    alias distinct neighbor images together).

CONVENTIONS
-----------
- Phi0 = kL * n n^T + kT * (I - n n^T) is the *positive* single-spring
  stiffness matrix; L/T amplitudes are refined separately per symmetry class.
- Rigid-body DOF ordering per body: (u_x, u_y, u_z, omega_x, omega_y, omega_z).
- T_i = [I3 | -skew(dr_i)] maps body DOF to atomic displacement.
- The FFT sign convention here is the numpy default (exp(-2*pi*i*h.R/N)).
  Verify this matches how you define q from grid index h elsewhere in your
  pipeline (structure factors, etc.) -- if it's backwards, swap fftn<->ifftn
  or negate q; this is purely a labeling convention and does not affect
  physical content, but must be self-consistent.
"""
# ----------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------

def skew(v):
    """3x3 cross-product matrix: skew(v) @ w == v x w."""
    return np.array([[0.0, -v[2], v[1]],
                      [v[2], 0.0, -v[0]],
                      [-v[1], v[0], 0.0]])


def T_matrix(dr):
    """3x6 projector: atomic displacement = T_matrix(dr) @ (u; omega)."""
    return np.hstack([np.eye(3), -skew(dr)])


# ----------------------------------------------------------------------
# Step 0: rigid-body definitions, CM, generalized mass matrix, Cholesky
# ----------------------------------------------------------------------

def build_rigid_bodies(df_atoms):
    """
    Group atoms by (sym_idx, group_id) -> one rigid body each.

    Returns
    -------
    body_index : dict (sym_idx, group_id) -> mu   (mu = 0..M-1)
    CM         : (M, 3) array, center of mass of each body (reference cell)
    L          : (M, 6, 6) array, Cholesky factors (M_gen = L @ L.T) per body
    """
    keys = sorted(set(zip(df_atoms['sym_idx'], df_atoms['group_id'])))
    body_index = {k: i for i, k in enumerate(keys)}
    M = len(keys)

    CM = np.zeros((M, 3))
    Mgen = np.zeros((M, 6, 6))

    for (sym_idx, group_id), mu in body_index.items():
        sel = (df_atoms['sym_idx'] == sym_idx) & (df_atoms['group_id'] == group_id)
        sub = df_atoms[sel]
        m = sub['mass'].to_numpy(dtype=float)
        pos = sub[['x', 'y', 'z']].to_numpy(dtype=float)

        cm = (m[:, None] * pos).sum(axis=0) / m.sum()
        CM[mu] = cm
        dr = pos - cm

        m_tot = m.sum()
        S = np.zeros((3, 3))      # sum m_i skew(dr_i); ~0 if cm is exact
        Itens = np.zeros((3, 3))  # moment-of-inertia tensor about CM
        for mi, dri in zip(m, dr):
            S += mi * skew(dri)
            Itens += mi * (dri @ dri * np.eye(3) - np.outer(dri, dri))

        Mg = np.zeros((6, 6))
        Mg[:3, :3] = m_tot * np.eye(3)
        Mg[:3, 3:] = -S
        Mg[3:, :3] = S          # S is skew-symmetric, so S.T = -S = (-S).T consistent
        Mg[3:, 3:] = Itens
        Mgen[mu] = Mg

    L = np.zeros((M, 6, 6))
    for mu in range(M):
        L[mu] = np.linalg.cholesky(Mgen[mu])

    return body_index, CM, L


# ----------------------------------------------------------------------
# Step 1: accumulate real-space 6x6 blocks per symmetry class, per (mu,nu,R)
# ----------------------------------------------------------------------

def accumulate_real_space_blocks(df_symmetry, body_index, CM, lattice, grid_shape):
    """
    Returns
    -------
    classes : sorted list of symmetry-class labels w
    real_L, real_T : dense arrays, shape (W, M, M, N1, N2, N3, 6, 6)
        Un-mass-weighted, real-valued, longitudinal/transverse geometric blocks.
    """
    M = len(body_index)
    N1, N2, N3 = grid_shape
    grid = np.array(grid_shape)

    classes = sorted(df_symmetry['spring_class'].unique())
    widx = {w: k for k, w in enumerate(classes)}
    W = len(classes)

    raw_L = defaultdict(lambda: np.zeros((6, 6)))
    raw_T = defaultdict(lambda: np.zeros((6, 6)))

    for row in df_symmetry.itertuples(index=False):
        w = row.spring_class
        mu = body_index[(row.sym_idx1, row.group_id1)]
        nu = body_index[(row.sym_idx2, row.group_id2)]

        Rvec = np.asarray(row.pbc_shift, dtype=int)
        Rcart = Rvec @ lattice

        r1 = np.asarray(row.r1, dtype=float)
        r2 = np.asarray(row.r2, dtype=float)

        dr1 = r1 - CM[mu]
        dr2 = (r2 - Rcart) - CM[nu]   # bring cra2 back to its reference-cell image

        d = r1 - r2
        dist = np.linalg.norm(d)
        n = d / dist
        Phi_L = np.outer(n, n)
        Phi_T = np.eye(3) - Phi_L

        T1 = T_matrix(dr1)
        T2 = T_matrix(dr2)

        blk11_L = T1.T @ Phi_L @ T1
        blk22_L = T2.T @ Phi_L @ T2
        blk12_L = -T1.T @ Phi_L @ T2
        blk21_L = blk12_L.T           # Phi_L symmetric real -> exact transpose

        blk11_T = T1.T @ Phi_T @ T1
        blk22_T = T2.T @ Phi_T @ T2
        blk12_T = -T1.T @ Phi_T @ T2
        blk21_T = blk12_T.T

        R0 = (0, 0, 0)
        Rmod = tuple(Rvec % grid)
        Rneg = tuple((-Rvec) % grid)

        raw_L[(w, mu, mu, R0)] += blk11_L
        raw_L[(w, nu, nu, R0)] += blk22_L
        raw_L[(w, mu, nu, Rmod)] += blk12_L
        raw_L[(w, nu, mu, Rneg)] += blk21_L

        raw_T[(w, mu, mu, R0)] += blk11_T
        raw_T[(w, nu, nu, R0)] += blk22_T
        raw_T[(w, mu, nu, Rmod)] += blk12_T
        raw_T[(w, nu, mu, Rneg)] += blk21_T

    real_L = np.zeros((W, M, M, N1, N2, N3, 6, 6))
    real_T = np.zeros((W, M, M, N1, N2, N3, 6, 6))
    for (w, mu, nu, R), blk in raw_L.items():
        real_L[widx[w], mu, nu, R[0], R[1], R[2]] += blk
    for (w, mu, nu, R), blk in raw_T.items():
        real_T[widx[w], mu, nu, R[0], R[1], R[2]] += blk

    return classes, real_L, real_T


# ----------------------------------------------------------------------
# Step 2: mass-weight (once, in real space) then FFT to get S_L^w(q), S_T^w(q)
# ----------------------------------------------------------------------

def mass_weight_blocks(real_blocks, L):
    """
    Apply Linv[mu] @ block @ Linv[nu].T to every (w, mu, nu, R) block.
    real_blocks: (W, M, M, N1, N2, N3, 6, 6)
    L: (M, 6, 6) Cholesky factors
    """
    M = L.shape[0]
    Linv = np.linalg.inv(L)  # (M,6,6); fine for 6x6, or use solve_triangular
    # out[w,mu,nu,R,a,d] = sum_bc Linv[mu,a,b] * blk[w,mu,nu,R,b,c] * Linv[nu,d,c]
    out = np.einsum('mab,wmnijkbc,ndc->wmnijkad', Linv, real_blocks, Linv,
                     optimize=True)
    return out


def fourier_transform_blocks(mass_weighted_blocks):
    """FFT along the three lattice-offset axes (axes 3,4,5)."""
    return np.fft.fftn(mass_weighted_blocks, axes=(3, 4, 5))


# ----------------------------------------------------------------------
# One-time setup: run once, cache S_L_fft / S_T_fft for the whole refinement
# ----------------------------------------------------------------------

def precompute(df_symmetry, df_atoms, lattice, grid_shape):
    body_index, CM, L = build_rigid_bodies(df_atoms)
    classes, real_L, real_T = accumulate_real_space_blocks(
        df_symmetry, body_index, CM, lattice, grid_shape)
    mw_L = mass_weight_blocks(real_L, L)
    mw_T = mass_weight_blocks(real_T, L)
    S_L_fft = fourier_transform_blocks(mw_L)   # (W,M,M,N1,N2,N3,6,6) complex
    S_T_fft = fourier_transform_blocks(mw_T)
    M = len(body_index)
    return {
        'classes': classes,
        'body_index': body_index,
        'M': M,
        'S_L_fft': S_L_fft,
        'S_T_fft': S_T_fft,
    }


# ----------------------------------------------------------------------
# Per-iteration: cheap weighted sum + reshape to a 6M x 6M matrix
# ----------------------------------------------------------------------

def build_D_grid(precomputed, kL, kT):
    """
    kL, kT: dict {class_label: value}, refined spring constants.
    Returns D_grid: (M, M, N1, N2, N3, 6, 6) complex array (mass-weighted D(q)
    for every q on the grid, for every body pair).
    """
    classes = precomputed['classes']
    S_L_fft = precomputed['S_L_fft']
    S_T_fft = precomputed['S_T_fft']
    D = np.zeros_like(S_L_fft[0])
    for k, w in enumerate(classes):
        D = D + kL[w] * S_L_fft[k] + kT[w] * S_T_fft[k]
    return D


def assemble_Dq(D_grid, h1, h2, h3):
    """Extract the full 6M x 6M D(q) at a single grid index (h1,h2,h3)."""
    M = D_grid.shape[0]
    block = D_grid[:, :, h1, h2, h3, :, :]        # (M,M,6,6)
    Dq = block.transpose(0, 2, 1, 3).reshape(6 * M, 6 * M)
    return Dq

"""
Builds df_atoms from a gemmi structure (space-group expansion of each rigid
body's constituent atoms), and computes/plots a band structure along the
tetragonal path Gamma-X-M-Gamma-Z-R-A-Z | X-R | M-A.

Reuses skew(), T_matrix(), build_rigid_bodies()
"""
# ----------------------------------------------------------------------
# df_atoms construction
# ----------------------------------------------------------------------

def build_df_atoms(st, groups):
    cell = st.cell
    model = st[0]
    images = cell.images  # sym_idx=0 -> identity, sym_idx=i -> images[i-1], matching enm.py

    rows = []
    for sym_idx in range(len(images) + 1):
        transform = gemmi.Transform() if sym_idx == 0 else images[sym_idx - 1]
        for group_id, selection in enumerate(groups):
            sel_model = selection.copy_model_selection(model)
            for chain in sel_model:
                for residue in chain:
                    for atom in residue:
                        v = transform.apply(cell.fractionalize(atom.pos))
                        pos_new = cell.orthogonalize(gemmi.Fractional(v.x, v.y, v.z))
                        rows.append({
                            'sym_idx': sym_idx,
                            'group_id': group_id,
                            'mass': atom.element.weight,
                            'x': pos_new.x,
                            'y': pos_new.y,
                            'z': pos_new.z,
                            'chain': chain.name,
                            'resnum': residue.seqid.num,
                            'atom_name': atom.name,
                        })
    return pd.DataFrame(rows)


def get_lattice(cell):
    """3x3 array, rows = real-space lattice vectors a1,a2,a3 (Cartesian, Angstrom)."""
    a1 = cell.orthogonalize(gemmi.Fractional(1, 0, 0))
    a2 = cell.orthogonalize(gemmi.Fractional(0, 1, 0))
    a3 = cell.orthogonalize(gemmi.Fractional(0, 0, 1))
    return np.array([[a1.x, a1.y, a1.z],
                      [a2.x, a2.y, a2.z],
                      [a3.x, a3.y, a3.z]])


def reciprocal_lattice(lattice):
    """3x3 array, rows = reciprocal lattice vectors b1,b2,b3 (with 2*pi convention)."""
    a1, a2, a3 = lattice
    V = np.dot(a1, np.cross(a2, a3))
    b1 = 2 * np.pi * np.cross(a2, a3) / V
    b2 = 2 * np.pi * np.cross(a3, a1) / V
    b3 = 2 * np.pi * np.cross(a1, a2) / V
    return np.array([b1, b2, b3])


# ----------------------------------------------------------------------
# sparse real-space blocks, suitable for evaluating D(q) at arbitrary q
# ----------------------------------------------------------------------

def accumulate_sparse_blocks(df_symmetry, body_index, CM, lattice):
    """
    Same geometric construction as accumulate_real_space_blocks, but keeps Cartesian R explicitly instead of
    scattering onto a fixed grid -- for evaluating D(q) at off-grid q
    (e.g. a band-structure path) by direct phase summation.

    Returns sparse_L, sparse_T: dict (w, mu, nu) -> list of (Rcart, 6x6 block)
    """
    sparse_L = defaultdict(list)
    sparse_T = defaultdict(list)

    for row in df_symmetry.itertuples(index=False):
        w = row.spring_class
        mu = body_index[(row.sym_idx1, row.group_id1)]
        nu = body_index[(row.sym_idx2, row.group_id2)]

        Rvec = np.asarray(row.pbc_shift, dtype=int)
        Rcart = Rvec @ lattice

        r1 = np.asarray(row.r1, dtype=float)
        r2 = np.asarray(row.r2, dtype=float)
        dr1 = r1 - CM[mu]
        dr2 = (r2 - Rcart) - CM[nu]

        d = r1 - r2
        n = d / np.linalg.norm(d)
        Phi_L = np.outer(n, n)
        Phi_T = np.eye(3) - Phi_L

        T1 = T_matrix(dr1)
        T2 = T_matrix(dr2)

        blk11_L = T1.T @ Phi_L @ T1
        blk22_L = T2.T @ Phi_L @ T2
        blk12_L = -T1.T @ Phi_L @ T2
        blk21_L = blk12_L.T

        blk11_T = T1.T @ Phi_T @ T1
        blk22_T = T2.T @ Phi_T @ T2
        blk12_T = -T1.T @ Phi_T @ T2
        blk21_T = blk12_T.T

        R0 = np.zeros(3)
        sparse_L[(w, mu, mu)].append((R0, blk11_L))
        sparse_L[(w, nu, nu)].append((R0, blk22_L))
        sparse_L[(w, mu, nu)].append((Rcart, blk12_L))
        sparse_L[(w, nu, mu)].append((-Rcart, blk21_L))

        sparse_T[(w, mu, mu)].append((R0, blk11_T))
        sparse_T[(w, nu, nu)].append((R0, blk22_T))
        sparse_T[(w, mu, nu)].append((Rcart, blk12_T))
        sparse_T[(w, nu, mu)].append((-Rcart, blk21_T))

    return sparse_L, sparse_T


def build_Dq(q, sparse_L, sparse_T, kL, kT, Linv, M):
    """Direct evaluation of the mass-weighted 6M x 6M D(q) at one Cartesian q."""
    D = np.zeros((M, M, 6, 6), dtype=complex)

    for (w, mu, nu), entries in sparse_L.items():
        acc = sum(blk * np.exp(1j * np.dot(q, R)) for R, blk in entries)
        D[mu, nu] += kL[w] * acc
    for (w, mu, nu), entries in sparse_T.items():
        acc = sum(blk * np.exp(1j * np.dot(q, R)) for R, blk in entries)
        D[mu, nu] += kT[w] * acc

    Dq = np.zeros((6 * M, 6 * M), dtype=complex)
    for mu in range(M):
        for nu in range(M):
            Dq[6*mu:6*mu+6, 6*nu:6*nu+6] = Linv[mu] @ D[mu, nu] @ Linv[nu].T
    return Dq

def build_path(recip_lattice, HIGH_SYM_POINTS, PATH_SEQ, n_per_segment=40):
    """
    Returns:
        qs   : (Npts,3) Cartesian q along the path
        xs   : (Npts,) path coordinate for plotting (uniform per segment)
        ticks, ticklabels : positions/labels of high-symmetry points
        breaks : indices (into qs/xs, pre-insertion) marking each '|'
                 discontinuity -- pass to plot_bands so the corresponding
                 segments aren't drawn as connected.
    """
    qs, xs, ticks, ticklabels, breaks = [], [], [], [], []
    x = 0.0
    i = 0
    need_start_point = True  # True right after Gamma and right after each '|'
    while i < len(PATH_SEQ):
        if PATH_SEQ[i] == '|':
            breaks.append(len(xs))  # break falls right before the next segment
            need_start_point = True
            i += 1
            continue
        start = PATH_SEQ[i]
        ticks.append(x)
        if i>0:
            if PATH_SEQ[i-1]=='|':
                label = (r'$\Gamma$' if PATH_SEQ[i-2] == 'G' else PATH_SEQ[i-2])+'|'+(r'$\Gamma$' if start == 'G' else start)
            else:
                label = r'$\Gamma$' if start == 'G' else start
        else:
            label = r'$\Gamma$' if start == 'G' else start
        ticklabels.append(label)
        if need_start_point:
            qs.append(HIGH_SYM_POINTS[start] @ recip_lattice)
            xs.append(x)
            need_start_point = False
        if i + 1 >= len(PATH_SEQ) or PATH_SEQ[i + 1] == '|':
            i += 1
            continue
        end = PATH_SEQ[i + 1]
        p0 = HIGH_SYM_POINTS[start] @ recip_lattice
        p1 = HIGH_SYM_POINTS[end] @ recip_lattice
        for n in range(1, n_per_segment + 1):
            t = n / n_per_segment
            qs.append((1 - t) * p0 + t * p1)
            xs.append(x + t)
        x += 1.0
        i += 1
    return np.array(qs), np.array(xs), ticks, ticklabels, breaks


def compute_bands(qs, sparse_L, sparse_T, kL, kT, Linv, M):
    n_modes = 6 * M
    bands = np.zeros((len(qs), n_modes))
    for idx, q in enumerate(qs):
        Dq = build_Dq(q, sparse_L, sparse_T, kL, kT, Linv, M)
        Dq = 0.5 * (Dq + Dq.conj().T)  # clean up numerical Hermiticity
        bands[idx] = np.linalg.eigvalsh(Dq)
    return bands


def _insert_breaks(xs, bands, breaks):
    """Insert a NaN row/x-value at each break index so matplotlib doesn't
    draw a connecting line across it. Processed back-to-front so earlier
    insertion indices stay valid."""
    xs = list(xs)
    bands = list(bands)
    for b in sorted(breaks, reverse=True):
        bands.insert(b, np.full(len(bands[0]), np.nan))
        xs.insert(b, xs[b])  # x-value is irrelevant; NaN in y breaks the line
    return np.array(xs), np.array(bands)


def plot_bands(bands, xs, ticks, ticklabels, ymin, ymax, breaks=()):
    freqs = np.sqrt(np.clip(bands, 0, None))  # eigenvalues of D are omega^2
    xs_plot, freqs_plot = _insert_breaks(xs, freqs, breaks)
    plt.figure(figsize=(8, 5))
    for m in range(freqs_plot.shape[1]):
        plt.plot(xs_plot, freqs_plot[:, m], color='C0', lw=1)
    for t in ticks:
        plt.axvline(t, color='k', lw=0.5)
    plt.xticks(ticks, ticklabels)
    plt.xlim(ticks[0], ticks[-1])
    plt.ylim(ymin, ymax)
    plt.ylabel('Frequency (arb. units)')
    plt.tight_layout()
    plt.show()

# ----------------------------------------------------------------------
# DOS and 2D dispersion images, from the full-BZ FFT grid
# ----------------------------------------------------------------------
#
# The path bands sample only a handful of 1D lines through the BZ -- a
# histogram of those points would badly misrepresent the DOS (it over-weights
# whatever region the path happens to linger near). A proper DOS needs
# eigenvalues from a dense, *uniform* sampling of the full 3D BZ, which is
# exactly what precompute()/build_D_grid() from rigid_body_dynmat.py already
# gives you on a Born-von Karman-commensurate grid -- reused here rather than
# looping build_Dq() by hand over a Monkhorst-Pack mesh.
 
def compute_all_eigs_from_grid(D_grid):
    """
    D_grid: (M,M,N1,N2,N3,6,6) complex, as returned by build_D_grid().
 
    Returns eigs: (N1,N2,N3,6M) real array of eigenvalues (=omega^2) at
    every grid q-point, kept in grid shape so 2D slices are easy to pull out.
    """
    M, _, N1, N2, N3, _, _ = D_grid.shape
    n_modes = 6 * M
    eigs = np.zeros((N1, N2, N3, n_modes))
    for h1 in range(N1):
        for h2 in range(N2):
            for h3 in range(N3):
                Dq = assemble_Dq(D_grid, h1, h2, h3)
                Dq = 0.5 * (Dq + Dq.conj().T)  # clean up numerical Hermiticity
                eigs[h1, h2, h3] = np.linalg.eigvalsh(Dq)
    return eigs
 
 
def compute_dos(eigs, n_bins=400, broaden_bins=2.0):
    """
    eigs: any shape ending in n_modes (e.g. (N1,N2,N3,6M)) of D-eigenvalues.
    Histograms frequencies sqrt(eig) over all grid q-points and bands, then
    applies a light Gaussian smoothing (in bin units) for a continuous curve.
 
    Returns centers (frequency bin centers), dos (counts, un-normalized --
    scale by 1/N_q if you want "states per unit cell per unit frequency").
    """
    freqs = np.sqrt(np.clip(eigs, 0, None)).ravel()
    hist, edges = np.histogram(freqs, bins=n_bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    dos = gaussian_filter1d(hist.astype(float), sigma=broaden_bins)
    return centers, dos
 
 
def plot_bands_and_dos(bands, xs, ticks, ticklabels, ymin, ymax, breaks, centers, dos,
                        outfile=None):
    """Combined panel: path band structure (left) + full-BZ DOS (right),
    sharing the frequency axis."""
    freqs = np.sqrt(np.clip(bands, 0, None))
    xs_plot, freqs_plot = _insert_breaks(xs, freqs, breaks)
 
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(10, 5), sharey=True,
        gridspec_kw={'width_ratios': [3, 1], 'wspace': 0.05})
 
    for m in range(freqs_plot.shape[1]):
        ax1.plot(xs_plot, freqs_plot[:, m], color='C0', lw=1)
    for t in ticks:
        ax1.axvline(t, color='k', lw=0.5)
    ax1.set_xticks(ticks)
    ax1.set_xticklabels(ticklabels)
    ax1.set_xlim(ticks[0], ticks[-1])
    ax1.set_ylim(ymin, ymax)
    ax1.set_ylabel('Frequency (arb. units)')
 
    ax2.plot(dos, centers, color='C1')
    ax2.fill_betweenx(centers, 0, dos, color='C1', alpha=0.3)
    ax2.set_xlabel('DOS')
    ax2.set_xlim(left=0)

    if outfile:
        plt.savefig(outfile, dpi=300)
    plt.show()
 
 
def plot_band_2d_slice(eigs, band_index, h3_fixed=0, outfile=None):
    """
    Image plot of one band's frequency across the (h1,h2) plane at fixed h3
    -- e.g. h3_fixed=0 gives the qz=0 (Gamma-X-M-Gamma-like) plane.
 
    eigs: (N1,N2,N3,6M) array from compute_all_eigs_from_grid(), already
    sorted ascending at each q by eigvalsh. NOTE: sorting per-q means a
    fixed "band index" can jump between physically different branches
    wherever two bands cross/nearly-cross -- fine for a quick visual, but
    don't over-interpret sharp features in a single slice as physical without
    checking neighboring bands too.
    """
    freqs = np.sqrt(np.clip(eigs[:, :, h3_fixed, band_index], 0, None))
    plt.figure(figsize=(5, 5))
    im = plt.imshow(freqs.T, origin='lower', extent=[0, 1, 0, 1],
                     aspect='equal', cmap='viridis')
    plt.colorbar(im, label='Frequency (arb. units)')
    plt.xlabel(r'$q_1$ (fractional)')
    plt.ylabel(r'$q_2$ (fractional)')
    plt.title(f'Band {band_index}, $q_3$ index = {h3_fixed}')
    if outfile:
        plt.savefig(outfile, dpi=300)
    plt.show()