"""Write what the MBAM movie needs from rigid_body_diffuse_6o2h.ipynb.

Run it in that notebook's namespace, after export_mbam_bundle.py:

    %run -i export_mbam_movie.py

It writes OUT_PREFIX + 'movie_bundle.npz': the molecular envelope (the same
Gaussian-blurred surface the notebook draws for its mode animations), the atom
positions and centre of mass, and the footprint of every contact patch (the
midpoints of its contacting atom pairs), all in the notebook's Cartesian frame.
"""
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from skimage.measure import marching_cubes

MOVIE_BUNDLE = OUT_PREFIX + 'movie_bundle.npz'
_ENV_SPACING, _ENV_SIGMA, _ENV_LEVEL = 1.5, 2.0, 0.35       # as in the notebook's envelope cell

def _envelope(P, spacing=_ENV_SPACING, sigma=_ENV_SIGMA, level=_ENV_LEVEL):
    lo, hi = P.min(0) - 3*sigma, P.max(0) + 3*sigma
    axes_ = [np.arange(l, h + spacing, spacing) for l, h in zip(lo, hi)]
    G3 = np.stack(np.meshgrid(*axes_, indexing='ij'), -1).reshape(-1, 3)
    dens = np.zeros(len(G3)); tree = cKDTree(P)
    for i, nb in enumerate(tree.query_ball_point(G3, 3*sigma)):
        if nb:
            dens[i] = np.exp(-((P[nb] - G3[i])**2).sum(1)/(2*sigma**2)).sum()
    dens = dens.reshape([len(a) for a in axes_])
    v, f, _, _ = marching_cubes(dens, level=level*dens.max())
    v = v*spacing + lo
    n = len(v)
    i_ = np.concatenate([f[:, 0], f[:, 1], f[:, 2]]); j_ = np.concatenate([f[:, 1], f[:, 2], f[:, 0]])
    _, lab = connected_components(csr_matrix((np.ones(len(i_), np.int8), (i_, j_)), shape=(n, n)), directed=False)
    keep_lab = np.bincount(lab).argmax(); keep = np.where(lab == keep_lab)[0]
    remap = np.full(n, -1); remap[keep] = np.arange(len(keep))
    return v[keep], remap[f[(lab[f] == keep_lab).all(axis=1)]]

if 'mc_verts_cart' in globals() and 'mc_faces' in globals():
    _verts, _faces = np.asarray(mc_verts_cart), np.asarray(mc_faces)
    print("using the notebook's molecular envelope")
else:
    print("computing the molecular envelope from the atom positions ...")
    _verts, _faces = _envelope(np.asarray(apos))

_mid, _idx = [], []
for _c_i, _c in enumerate(shell_order):
    m = np.asarray(unique[_c]['midpoints'])
    _mid.append(m); _idx.append(np.full(len(m), _c_i))
np.savez_compressed(MOVIE_BUNDLE, mesh_verts=_verts, mesh_faces=_faces, apos=np.asarray(apos),
                    r_cm_at=np.asarray(r_cm_at), patch_mid=np.vstack(_mid), patch_idx=np.concatenate(_idx))
print(f"wrote {MOVIE_BUNDLE}: envelope with {len(_verts)} vertices / {len(_faces)} triangles, "
      f"{len(apos)} atoms, {len(np.vstack(_mid))} contact-pair midpoints")
