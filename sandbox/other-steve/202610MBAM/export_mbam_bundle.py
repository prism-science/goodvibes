"""Write everything the MBAM notebook needs from rigid_body_diffuse_6o2h.ipynb.

Run it LAST in that notebook, in the notebook's own namespace:

    %run -i export_mbam_bundle.py

(-i matters: the script reads the notebook's variables and functions directly.)
It writes OUT_PREFIX + 'mbam_bundle.npz' (default 'synthetic_mbam_bundle.npz').

Besides copying arrays it computes three things only this notebook can, because
they need the electron density: G(q) on a dense independent 'oracle' q-set (for
measuring generalization against the noise-free truth), G on the l = 0 and
l = 0.5 planes (for diffuse maps of reduced models), and the fitted model's
intensities at the data points (so the MBAM notebook can check that its own
forward model reproduces this one to rounding error).
"""
import numpy as np

ORACLE_N_NEAR = 400       # near-Bragg points in the oracle set
ORACLE_N_MID  = 1600      # mid-zone points in the oracle set
ORACLE_SEED   = 101       # differs from the data's seed, so the set is independent
MBAM_BUNDLE   = OUT_PREFIX + 'mbam_bundle.npz'

_need = ['B_recip', 'A_orth', 'cell', 'shell_order', 'unique', 'r_cm_at', 'M_mat',
         'freq_unit', 'K_PRIOR_LOCAL', 'L_PRIOR', 'L_TRUE', 'theta_fit', 'LAMBDA_PRIOR',
         'q_all', 'G_all', 'I_true_all', 'I_obs_all', 'sigma_all', 'train_mask',
         'NOISE_FLOOR', 'NOISE_SCALE', 'q_cart', 'x', 'tick_idx', 'path_labels',
         'sample_q_points', 'well_conditioned_batch', 'G_at_hkl_batch', 'G_plane',
         'diffuse_intensity_batch', 'build_K', 'theta_to_Lmap', 'D_MIN_DIFFUSE', 'P']
_missing = [v for v in _need if v not in globals()]
assert not _missing, f"run this at the end of the notebook; missing: {_missing}"

_nc = len(shell_order)
_contacts = {
    'n':          np.array([list(c) for c in shell_order], int),
    'R_n':        np.array([unique[c]['R_n'] for c in shell_order]),
    'R_frame':    np.array([unique[c]['R_frame'] for c in shell_order]),
    'c_local':    np.array([unique[c]['c_local'] for c in shell_order]),
    'Gamma':      np.array([unique[c]['Gamma'] for c in shell_order]),
    'n_pairs':    np.array([unique[c]['n_contacts'] for c in shell_order]),
    'rho_g':      np.array([unique[c]['rho_g'] for c in shell_order]),
    'contact_pt': np.array([unique[c]['contact_pt'] for c in shell_order]),
}

# --- the fitted model at the data points, for the forward-model check -------
_, _K_LAB_FIT = build_K(theta_to_Lmap(theta_fit))
_I_fit_all = diffuse_intensity_batch(_K_LAB_FIT, q_all[:, 0], q_all[:, 1], q_all[:, 2], G_arr=G_all)

# --- which data points are near-Bragg ---------------------------------------
# sample_q_points returns its near-Bragg points first; recompute that split for
# the points that survived the conditioning screen.
_raw = sample_q_points()
_is_near_raw = np.zeros(len(_raw), bool)
_is_near_raw[:min(60, len(_raw))] = True
try:
    _is_near_all = _is_near_raw[well_conditioned_batch(build_K(L_TRUE)[1], _raw)]
    assert len(_is_near_all) == len(q_all)
except Exception:
    _is_near_all = np.zeros(len(q_all), bool)

# --- oracle set: independent q-points, noise-free truth ----------------------
_K_LAB_TRUE = build_K(L_TRUE)[1]
_qo_raw = sample_q_points(n_near=ORACLE_N_NEAR, n_mid=ORACLE_N_MID, seed=ORACLE_SEED)
_near_o = np.arange(len(_qo_raw)) < ORACLE_N_NEAR
_keep_o = well_conditioned_batch(_K_LAB_TRUE, _qo_raw)
q_oracle, oracle_is_near = _qo_raw[_keep_o], _near_o[_keep_o]
print(f"oracle set: {len(q_oracle)} points ({oracle_is_near.sum()} near-Bragg); computing G ...")
G_oracle = G_at_hkl_batch(q_oracle[:, 0], q_oracle[:, 1], q_oracle[:, 2])
I_true_oracle = diffuse_intensity_batch(_K_LAB_TRUE, q_oracle[:, 0], q_oracle[:, 1],
                                        q_oracle[:, 2], G_arr=G_oracle)
sigma_oracle = NOISE_FLOOR + NOISE_SCALE*np.sqrt(np.clip(I_true_oracle, 0, None))

# --- G on two planes, for diffuse maps of reduced models ---------------------
def _plane(L_layer):
    h_max = int(np.ceil(cell.a/D_MIN_DIFFUSE)) + 1
    k_max = int(np.ceil(cell.b/D_MIN_DIFFUSE)) + 1
    h_pts = np.arange(-h_max*P, h_max*P + 1)/P
    k_pts = np.arange(-k_max*P, k_max*P + 1)/P
    HH, KK = np.meshgrid(h_pts, k_pts, indexing='ij')
    q = HH[..., None]*B_recip[:, 0] + KK[..., None]*B_recip[:, 1] + L_layer*B_recip[:, 2]
    qn = np.linalg.norm(q, axis=-1)
    keep = (qn > 1e-3) & (qn < 2*np.pi/D_MIN_DIFFUSE)
    return h_pts, k_pts, G_plane(h_pts, k_pts, float(L_layer)), keep, h_max, k_max

print("G on the l = 0 and l = 0.5 planes ...")
_p0, _p5 = _plane(0.0), _plane(0.5)

_scan_arr = np.array(_scan, float) if '_scan' in globals() else np.zeros((0, 9))
_gen_spec = np.asarray(gen_spec, float) if 'gen_spec' in globals() else np.zeros(0)
_perm_arr = np.asarray(_perm, int) if '_perm' in globals() else np.arange(len(q_all))
_nfolds = int(N_FOLDS) if 'N_FOLDS' in globals() else 3

np.savez_compressed(
    MBAM_BUNDLE,
    # lattice and contacts
    B_recip=B_recip, A_orth=A_orth,
    cell=np.array([cell.a, cell.b, cell.c, cell.alpha, cell.beta, cell.gamma]),
    **{f'contact_{k}': v for k, v in _contacts.items()},
    r_cm_at=r_cm_at, M_mat=M_mat, freq_unit=float(freq_unit),
    # prior, truth, fit
    K_prior_local=np.array([K_PRIOR_LOCAL[c] for c in shell_order]),
    L_prior=np.array([L_PRIOR[c] for c in shell_order]),
    L_true=np.array([L_TRUE[c] for c in shell_order]),
    theta_fit=np.asarray(theta_fit, float), lambda_prior=float(LAMBDA_PRIOR),
    lambda_scan=_scan_arr, gen_spec=_gen_spec,
    # data
    q_all=q_all, G_all=G_all, I_true_all=I_true_all, I_obs_all=I_obs_all,
    sigma_all=sigma_all, train_mask=train_mask, is_near_all=_is_near_all,
    I_fit_all=_I_fit_all, noise=np.array([NOISE_FLOOR, NOISE_SCALE]),
    cv_perm=_perm_arr, cv_nfolds=_nfolds,
    # oracle set
    q_oracle=q_oracle, G_oracle=G_oracle, I_true_oracle=I_true_oracle,
    sigma_oracle=sigma_oracle, oracle_is_near=oracle_is_near,
    # band-structure path
    bs_q_cart=q_cart, bs_x=x, bs_ticks=np.asarray(tick_idx, int),
    bs_labels=np.array(path_labels),
    # planes
    plane0_h=_p0[0], plane0_k=_p0[1], plane0_G=_p0[2], plane0_keep=_p0[3],
    plane0_hk=np.array(_p0[4:6]),
    plane5_h=_p5[0], plane5_k=_p5[1], plane5_G=_p5[2], plane5_keep=_p5[3],
    plane5_hk=np.array(_p5[4:6]),
)
print(f"wrote {MBAM_BUNDLE}: {_nc} contacts, {len(q_all)} data points "
      f"({int(train_mask.sum())} training), {len(q_oracle)} oracle points")
