"""Elastic Network Model (ENM)"""

from dataclasses import dataclass
from functools import cache, lru_cache
import warnings

import gemmi
import pandas as pd


def _find_contacts(st, distance_cutoff, include_h=False):
    max_radius = distance_cutoff + 1.0  # what is optimal here?
    cs = gemmi.ContactSearch(distance_cutoff)
    cs.ignore = gemmi.ContactSearch.Ignore.Nothing
    cs.twice = True  # True: get both copies for MATLAB-like behavior
    ns = gemmi.NeighborSearch(st[0], st.cell, max_radius).populate(include_h=include_h)
    contacts = cs.find_contacts(ns)
    n = len(contacts)

    # Pre-allocate column storage
    cra1 = [None] * n
    cra2 = [None] * n
    sym_idx1 = [0] * n
    sym_idx2 = [0] * n
    pbc_shift1 = [(0, 0, 0)] * n
    pbc_shift2 = [None] * n

    find_image = st.cell.find_nearest_pbc_image

    for i, res in enumerate(contacts):
        p1 = res.partner1
        p2 = res.partner2
        cra1[i] = str(p1)
        cra2[i] = str(p2)
        sym_idx2[i] = res.image_idx
        pbc_shift2[i] = find_image(p1.atom.pos, p2.atom.pos, res.image_idx).pbc_shift
        # if sym_idx2 is zero and pbc_shift2 is (0, 0, 0), then this is an internal contact,
        # to properly detect duplicates, lets put the atoms in alphabetical order
        if sym_idx2[i] == 0 and pbc_shift2[i] == (0, 0, 0):
            if cra1[i] > cra2[i]:
                cra1[i], cra2[i] = cra2[i], cra1[i]

    df = pd.DataFrame(
        {
            "cra1": cra1,
            "cra2": cra2,
            "sym_idx1": sym_idx1,
            "sym_idx2": sym_idx2,
            "pbc_shift1": pbc_shift1,
            "pbc_shift2": pbc_shift2,
        }
    )

    def num_internal(df):
        is_internal = (df["sym_idx1"] == df["sym_idx2"]) & (df["pbc_shift1"] == df["pbc_shift2"])
        return is_internal.sum()

    num_internal_before = num_internal(df)
    df = df.drop_duplicates(ignore_index=True)
    num_internal_after = num_internal(df)

    if num_internal_before != 2 * num_internal_after:
        warnings.warn(
            (
                "Expected duplicate internal contacts to decrease by half, but "
                f"found {num_internal_before} before and {num_internal_after} after."
                f"This suggests that {num_internal_before - 2 * num_internal_after} "
                "contacts were dropped improperly. This is a known bug for very small"
                "unit cells. Bug your friendly neighborhood developer to fix it!"
            ),
            RuntimeWarning,
        )

    return df


def _assign_interface(df):

    # assign interface labels
    df["interface"] = df.groupby(["sym_idx1", "pbc_shift1", "sym_idx2", "pbc_shift2"]).ngroup()
    return df


def _pack_unit_cell(st, inplace=False):
    """Modify the translation component of the cell images to pack the unit cell"""
    polymer_selection = gemmi.Selection(";polymer")
    polymer_model = polymer_selection.copy_model_selection(st[0])
    asu_com = st.cell.fractionalize(polymer_model.calculate_center_of_mass())
    asu_shift = asu_com.wrap_to_unit() - asu_com
    if not inplace:
        st = st.clone()
    for im in st.cell.images:
        im_com = im.apply(asu_com)
        im_shift = im_com.wrap_to_unit() - im_com
        im.vec.x += im_shift.x - asu_shift.x
        im.vec.y += im_shift.y - asu_shift.y
        im.vec.z += im_shift.z - asu_shift.z
    return st


def _is_identity(t, wrap=True, tol=1e-9):
    """return True if the transform is identity modulo PBC"""
    tf = False
    if t.mat.approx(gemmi.Mat33(), tol):
        fvec = gemmi.Fractional(*t.vec.tolist())
        if wrap:
            fvec = fvec.wrap_to_zero()
        tf = fvec.approx(gemmi.Vec3(0, 0, 0), tol)
    return tf


def _find_inverse(t, images, wrap=True):
    """Find the inverse of a transform (t) from within a list of transforms (images).

    Returns the first matching index.
    """
    for j, im in enumerate(images):
        if _is_identity(t.combine(im), wrap=wrap):
            return j
    raise ValueError(f"Inverse not found for transform: {t}")


def _symop_to_transform(images, sym_idx, pbc_shift):
    shift_transform = gemmi.Transform(gemmi.Mat33(), gemmi.Vec3(*pbc_shift))
    if sym_idx == 0:
        return shift_transform
    return shift_transform @ images[sym_idx - 1]


def _transform_to_symop(images, t):
    if _is_identity(t, wrap=True):
        return 0, (int(t.vec.x), int(t.vec.y), int(t.vec.z))
    for j, im in enumerate(images):
        op = im @ t.inverse()
        if _is_identity(op, wrap=True):
            sym_idx = j + 1
            pbc_shift = (int(op.vec.x), int(op.vec.y), int(op.vec.z))
            return sym_idx, pbc_shift
    raise ValueError(f"Symmetry operation not found for transform: {t}")


@dataclass(frozen=True)
class _SymmetryImageFinder:
    images: tuple[gemmi.Transform, ...]

    @lru_cache(maxsize=None)
    def symop_to_transform(self, sym_idx, pbc_shift):
        shift_transform = gemmi.Transform(gemmi.Mat33(), gemmi.Vec3(*pbc_shift))
        if sym_idx == 0:
            return shift_transform
        return shift_transform @ self.images[sym_idx - 1]

    def transform_to_symop(self, t):
        if _is_identity(t, wrap=True):
            return 0, (int(t.vec.x), int(t.vec.y), int(t.vec.z))
        for j, im in enumerate(self.images):
            op = im @ t.inverse()
            if _is_identity(op, wrap=True):
                sym_idx = j + 1
                pbc_shift = (-1 * int(op.vec.x), -1 * int(op.vec.y), -1 * int(op.vec.z))
                return sym_idx, pbc_shift
        raise ValueError(f"Symmetry operation not found for transform: {t}")

    @lru_cache(maxsize=None)
    def find_symmetry_image(self, im_idx, sym_idx, pbc_shift):
        im = self.images[im_idx]
        t = self.symop_to_transform(sym_idx, pbc_shift)
        new_im = im @ t
        sym_idx2, pbc_shift2 = self.transform_to_symop(new_im)
        return sym_idx2, pbc_shift2


def _symmetry_expand(df, images):
    image_finder = _SymmetryImageFinder(tuple(images))
    rows = list(df.itertuples(index=False, name="Row"))

    def transform_contact(row, im_idx):
        cra1, cra2 = row.cra1, row.cra2
        sym_idx1, pbc_shift1 = image_finder.find_symmetry_image(im_idx, row.sym_idx1, row.pbc_shift1)
        sym_idx2, pbc_shift2 = image_finder.find_symmetry_image(im_idx, row.sym_idx2, row.pbc_shift2)

        if pbc_shift1 == (0, 0, 0) and pbc_shift2 == (0, 0, 0) and sym_idx2 < sym_idx1:
            sym_idx1, sym_idx2 = sym_idx2, sym_idx1
            pbc_shift1, pbc_shift2 = pbc_shift2, pbc_shift1
            cra1, cra2 = cra2, cra1

        return cra1, cra2, sym_idx1, pbc_shift1, sym_idx2, pbc_shift2

    result = [df]
    for im_idx in range(len(images)):
        updates = [transform_contact(row, im_idx) for row in rows]
        cra1, cra2, sym_idx1, pbc_shift1, sym_idx2, pbc_shift2 = zip(*updates)

        tmp = df.copy()
        tmp["cra1"] = cra1
        tmp["cra2"] = cra2
        tmp["sym_idx1"] = sym_idx1
        tmp["pbc_shift1"] = pbc_shift1
        tmp["sym_idx2"] = sym_idx2
        tmp["pbc_shift2"] = pbc_shift2
        result.append(tmp)

    df2 = pd.concat(result, ignore_index=True)
    return df2.drop_duplicates(ignore_index=True)
