# coding: utf-8

from __future__ import absolute_import, division, print_function, unicode_literals

"""
This module defines workflows for HSE bandstructure and band-
edge finding as part of the JCAP project.
"""

import numpy as np

from fireworks import Workflow, Firework
from fireworks import FiretaskBase, FWAction, explicit_serialize

from atomate.vasp.fireworks.core import OptimizeFW, StaticFW, NonSCFFW, HSEBSFW
from atomate.vasp.workflows.presets.core import wf_bandstructure_plus_hse
from atomate.vasp.workflows.base.adsorption import get_slab_fw
from atomate.vasp.firetasks.parse_outputs import BandedgesToDb
from atomate.utils.utils import get_fws_and_tasks

from custodian.vasp.handlers import *

from pymatgen.core.surface import generate_all_slabs
from pymatgen.io.vasp.sets import MVLSlabSet, MPStaticSet

__author__ = 'Joseph Montoya, Arunima Singh'
__email__ = 'montoyjh@lbl.gov'

def get_hse_bandedge_wf(structure, gap_only=True, vasp_cmd='vasp', 
                        db_file=None, max_index=1, min_slab_size=7.0,
                        min_vacuum_size=15.0, analysis=True):
    """
    Function to return workflow designed to return HSE band edges.
    In development.
    """
    # First part of WF is standard BS + HSE with option for linemode
    wf = wf_bandstructure_plus_hse(structure, gap_only=gap_only)
    
    # Second part of workflow is an HSE run on a minimal slab,
    # generated from the bulk structure
    sgp = {"min_slab_size": min_slab_size,
           "min_vacuum_size": min_vacuum_size}
    slabs = generate_all_slabs(structure, max_index=max_index, **sgp)
    # Get min volume slab
    slabs = sorted(slabs, key=lambda x: x.volume)
    slab = slabs[0]

    # TODO: decide whether we want to optimize slab, slab thickness, etc.
    # Since these are semiconductors, shouldn't need as much k-point density
    kpoints = MVLSlabSet(slab, k_product=30).kpoints 
    vis = MPStaticSet(slab, user_kpoints_settings=kpoints)
    cparams = {"vasp_cmd": vasp_cmd, "db_file": db_file}
    slab_fws = [get_slab_fw(slab, bulk_structure=structure, vasp_input_set=vis,
                            slab_gen_params=sgp, name="slab", **cparams)]
    slab_fws.append(NonSCFFW(slab, parents=slab_fws[-1], name="slab", **cparams))
    slab_fws.append(HSEBSFW(slab, parents=slab_fws[-1], name="slab_hse", **cparams))
    wf_slab = Workflow(slab_fws)
    wf.append_wf(wf_slab, wf.root_fw_ids)
    if analysis:
        wf_analysis = Workflow([
            Firework(BandedgesToDb(structure=structure, db_file=db_file, fw_spec_field='tags'),
                     name="Bandedge analysis")])
        wf.append_wf(wf_analysis, wf.leaf_fw_ids)
    
    # Turn off NPAR for HSE fireworks
    wf = add_modify_incar(wf, modify_incar_params={"incar_update": {"NPAR": 1}}, 
                          fw_name_constraint='hse')

    # Turn on LVTOT in relevant FWs
    wf = add_modify_incar(wf, modify_incar_params={"incar_update": {"LVTOT": True}},
                          fw_name_constraint="slab_hse")

    # Modify handlers (TODO: make a preset handler_group)
    # This is so that the slab fws don't exit on IBZKPT errors
    slab_handlers = [VaspErrorHandler(), UnconvergedErrorHandler(),
                     NonConvergingErrorHandler(), PotimErrorHandler(),
                     PositiveEnergyErrorHandler(), FrozenJobErrorHandler(),
                     StdErrHandler()]
    wf = modify_handlers(wf, slab_handlers, fw_name_constraint='slab')

    return wf


def modify_handlers(wf, handler_group, fw_name_constraint=None):
    """
    Modifies custodian handlers, to allow
    for testing of different combinations
    """
    for idx_fw, idx_t in get_fws_and_tasks(wf, fw_name_constraint=fw_name_constraint,
                                           task_name_constraint="Custodian"):
        wf.fws[idx_fw].tasks[idx_t]['handler_group'] = handler_group
    return wf


if __name__ == "__main__":
    # Here's an example of the usage of the workflow
    from pymatgen import MPRester
    from atomate.vasp.powerups import add_tags, add_modify_incar
    from fireworks import LaunchPad
    mpr = MPRester()
    lpad = LaunchPad.auto_load()
    mp_ids = ['mp-149', 'mp-2657', 'mp-24972', 'mp-19443', 'mp-19142']
    for mp_id in mp_ids:
        structure = mpr.get_structure_by_material_id(mp_id)
        wf = get_hse_bandedge_wf(structure, vasp_cmd=">>vasp_cmd<<",
                                 db_file=">>db_file<<")
        wf = add_modify_incar(wf)
        wf = add_tags(wf, ["Bandedge_benchmark_1", mp_id])
        lpad.add_wf(wf)
