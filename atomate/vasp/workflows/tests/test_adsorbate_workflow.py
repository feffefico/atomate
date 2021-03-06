# coding: utf-8

from __future__ import division, print_function, unicode_literals, absolute_import

import os
import unittest
import numpy as np

from fireworks import FWorker
from fireworks.core.rocket_launcher import rapidfire

from atomate.vasp.powerups import use_fake_vasp
from atomate.vasp.workflows.base.adsorption import get_wf_slab, get_slab_fw, \
    get_slab_trans_params
from atomate.utils.testing import AtomateTest

from pymatgen import Structure, Molecule, Lattice
from pymatgen.util.testing import PymatgenTest
from pymatgen.core.surface import generate_all_slabs
from pymatgen.transformations.advanced_transformations import SlabTransformation

__author__ = 'Kiran Mathew, Joseph Montoya'
__email__ = 'montoyjh@lbl.gov'

module_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
db_dir = os.path.join(module_dir, "..", "..", "..", "common", "test_files")
ref_dir = os.path.join(module_dir, "..", "..", "test_files")

DEBUG_MODE = False  # If true, retains the database and output dirs at the end of the test
VASP_CMD = None  # If None, runs a "fake" VASP. Otherwise, runs VASP with this command...


class TestAdsorptionWorkflow(AtomateTest):

    def setUp(self):
        super(TestAdsorptionWorkflow, self).setUp()

        self.struct_ir = Structure.from_spacegroup(
            "Fm-3m", Lattice.cubic(3.875728), ["Ir"], [[0, 0, 0]])
        sgp = {"max_index": 1, "min_slab_size": 7.0, "min_vacuum_size": 20.0}
        self.slabs = generate_all_slabs(self.struct_ir, **sgp)
        self.slab_100 = [slab for slab in self.slabs
                         if slab.miller_index==(1, 0, 0)][0]
        self.wf_1 = get_wf_slab(self.slab_100, True,
                                [Molecule("H", [[0, 0, 0]])],
                               db_file=os.path.join(db_dir, "db.json"))

    def test_wf_functions(self):
        # Test slab trans params generator
        for slab in self.slabs:
            trans_params = get_slab_trans_params(slab)
            trans = SlabTransformation(**trans_params)
            new_slab = trans.apply_transformation(slab.oriented_unit_cell)
            self.assertTrue(np.allclose(new_slab.cart_coords, slab.cart_coords))
            self.assertTrue(np.allclose(new_slab.lattice.matrix,
                                        slab.lattice.matrix))

        # Try something a bit more complicated
        formulas = ['Si', 'Sn', 'SrTiO3', 'Li2O']
        structs = [PymatgenTest.get_structure(s) for s in formulas]
        for struct in structs:
            slabs = generate_all_slabs(struct, max_index=2, min_slab_size=10,
                                       min_vacuum_size=20)
            for slab in slabs:
                trans_params = get_slab_trans_params(slab)
                trans = SlabTransformation(**trans_params)
                new_slab = trans.apply_transformation(slab.oriented_unit_cell)
                old_coords = np.around(slab.frac_coords, 10) % 1
                new_coords = np.around(new_slab.frac_coords, 10) % 1
                self.assertTrue(np.allclose(old_coords, new_coords))
                self.assertTrue(np.allclose(new_slab.lattice.matrix,
                                            slab.lattice.matrix))

    def _simulate_vasprun(self, wf):
        reference_dir = os.path.abspath(os.path.join(ref_dir, "adsorbate_wf"))
        ir_ref_dirs = {"Ir-structure optimization": os.path.join(reference_dir, "1"),
                       "Ir-Ir_(1, 0, 0) slab optimization": os.path.join(reference_dir, "2"),
                       "Ir-H1-Ir_(1, 0, 0) adsorbate optimization 0": os.path.join(reference_dir, "3"),
                       "Ir-H1-Ir_(1, 0, 0) adsorbate optimization 1": os.path.join(reference_dir, "4"),
                       "Ir-H1-Ir_(1, 0, 0) adsorbate optimization 2": os.path.join(reference_dir, "5")}
        return use_fake_vasp(wf, ir_ref_dirs, params_to_check=["ENCUT", "ISIF", "IBRION"])

    def _check_run(self, d, mode):
        if mode not in ["H1-Ir_(1, 0, 0) adsorbate optimization 1",
                        "oriented_ucell"]:
            raise ValueError("Invalid mode!")

        if "adsorbate" in mode:
            self.assertEqual(d["formula_reduced_abc"], "H1 Ir16")
            self.assertAlmostEqual(d["slab"]["lattice"]["c"], 31.005824)
        # Check relaxation of adsorbate
        # Check slab calculations
        # Check structure optimization
        if mode == "oriented_ucell":
            self.assertAlmostEqual(d["output"]["structure"]["lattice"]["a"], 3.89189644)

    def test_wf(self):
        wf = self._simulate_vasprun(self.wf_1)

        self.assertEqual(len(self.wf_1.fws), 5)
        # check vasp parameters for ionic relaxation
        ads_vis = [fw.tasks[1]['vasp_input_set']
                   for fw in self.wf_1.fws if "adsorbate" in fw.name]
        assert all([vis.incar['EDIFFG']==-0.02 for vis in ads_vis])
        assert all([vis.incar['ISIF']==2 for vis in ads_vis])
        self.lp.add_wf(wf)
        rapidfire(self.lp, fworker=FWorker(
            env={"db_file": os.path.join(db_dir, "db.json")}))

        # check relaxation
        d = self.get_task_collection().find_one(
            {"task_label": "H1-Ir_(1, 0, 0) adsorbate optimization 1"})
        self._check_run(d, mode="H1-Ir_(1, 0, 0) adsorbate optimization 1")

        # Check bulk opt
        d = self.get_task_collection().find_one(
            {"task_label": "structure optimization"})
        self._check_run(d, mode='oriented_ucell')

        wf = self.lp.get_wf_by_fw_id(1)
        self.assertTrue(all([s == 'COMPLETED' for s in wf.fw_states.values()]))


if __name__ == "__main__":
    unittest.main()
