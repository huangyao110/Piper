"""Geometry, original-label parity and archive verification for distance exports."""
from io import BytesIO
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest

import gemmi
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.build_ppi_distance_archives import distance_matrix, extract_chain, verify_archive


class DistanceTests(unittest.TestCase):
    def test_strict_boundary_and_float32_rounding(self):
        x = np.array([[0., 0., 0.]])
        y = np.array([[np.nextafter(12., 0.), 0., 0.], [12., 0., 0.], [13., 0., 0.]])
        d, stats = distance_matrix(x, y, np.ones(1, bool), np.ones(3, bool), [[0, 0]])
        self.assertEqual(d.dtype, np.float32)
        np.testing.assert_array_equal(d < 12., [[True, False, False]])
        self.assertEqual(stats["rounding_adjustments"], 1)
        self.assertLess(stats["max_abs_rounding_error_angstrom"], 1.e-6)

    def test_masks_all_unknown_and_contact_mismatch(self):
        x = np.array([[0., 0., 0.], [np.nan, np.nan, np.nan]])
        y = np.array([[3., 4., 0.]])
        d, _ = distance_matrix(x, y, np.array([True, False]), np.array([True]), [[0, 0]])
        self.assertEqual(d[0, 0], 5.)
        self.assertTrue(np.isnan(d[1, 0]))
        with self.assertRaisesRegex(ValueError, "disagree"):
            distance_matrix(x, y, np.array([True, False]), np.array([True]), [])
        d, stats = distance_matrix(x[1:], y, np.array([False]), np.array([True]), [])
        self.assertTrue(np.isnan(d).all())
        self.assertEqual(stats["valid_cells"], 0)

    def test_chain_swap_transposes_and_homomer_diagonal_survives(self):
        x = np.array([[0., 0., 0.], [30., 0., 0.]])
        y = np.array([[0., 3., 0.], [30., 3., 0.], [80., 0., 0.]])
        a, _ = distance_matrix(x, y, np.ones(2, bool), np.ones(3, bool), [[0, 0], [1, 1]])
        b, _ = distance_matrix(y, x, np.ones(3, bool), np.ones(2, bool), [[0, 0], [1, 1]])
        np.testing.assert_array_equal(a.T, b)
        self.assertEqual(a[0, 0], 3.)
        self.assertEqual(a[1, 1], 3.)

    def test_representative_atoms_and_occupancy(self):
        # Construct a polymer: GLY has CA; ALA lacks CB; SER has zero-occupancy CB.
        structure = gemmi.Structure()
        model = gemmi.Model("1")
        chain = gemmi.Chain("A")
        for i, (name, atom_name, occupancy) in enumerate(
                [("GLY", "CA", 1.), ("ALA", "CA", 1.), ("SER", "CB", 0.)], 1):
            residue = gemmi.Residue()
            residue.name = name
            residue.seqid = gemmi.SeqId(i, " ")
            residue.entity_type = gemmi.EntityType.Polymer
            atom = gemmi.Atom()
            atom.name = atom_name
            atom.element = gemmi.Element("C")
            atom.occ = occupancy
            atom.pos = gemmi.Position(i, 0., 0.)
            residue.add_atom(atom)
            chain.add_residue(residue)
        model.add_chain(chain)
        structure.add_model(model)
        sequence, coordinates, valid = extract_chain(structure[0], "A")
        self.assertEqual(sequence, "GAS")
        np.testing.assert_array_equal(valid, [True, False, False])
        self.assertTrue(np.isnan(coordinates[1:]).all())

    def test_independent_archive_verification_detects_bad_contacts(self):
        def write(path, prefix, arrays):
            payload = BytesIO()
            np.savez_compressed(payload, **arrays)
            with tarfile.open(path, "w") as tar:
                member = tarfile.TarInfo(f"{prefix}/ppi/00/00/ppi_example.npz")
                member.size = payload.tell()
                payload.seek(0)
                tar.addfile(member, payload)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            common = dict(pair_id="ppi_example", shape=np.array([1, 2]),
                          valid1=np.array([True]), valid2=np.array([True, False]))
            write(root / "old.tar", "concat", dict(**common, contacts=np.array([[0, 0]])))
            write(root / "new.tar", "distances", dict(**common, distances=np.array([[5., np.nan]], np.float32)))
            result = verify_archive(root / "new.tar", root / "old.tar", 1)
            self.assertTrue(result["contacts_exact"])
            write(root / "bad.tar", "distances", dict(**common, distances=np.array([[15., np.nan]], np.float32)))
            with self.assertRaisesRegex(ValueError, "contact labels"):
                verify_archive(root / "bad.tar", root / "old.tar", 1)


if __name__ == "__main__":
    unittest.main()
