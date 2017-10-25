# coding: utf-8

from __future__ import absolute_import, division, print_function, unicode_literals

"""
This module defines workflows for HSE bandstructure and band-
edge finding as part of the JCAP project.
"""

import numpy as np
from collections import OrderedDict

from fireworks import Workflow, Firework
from fireworks import FiretaskBase, FWAction, explicit_serialize

from atomate.vasp.fireworks.core import OptimizeFW, StaticFW, NonSCFFW, HSEBSFW
from atomate.vasp.workflows.presets.core import wf_bandstructure_plus_hse
from atomate.vasp.workflows.base.adsorption import get_slab_fw, SLAB_HANDLERS
from atomate.vasp.firetasks.parse_outputs import BandedgesToDb
from atomate.vasp.powerups import add_tags, add_modify_incar, modify_handlers,\
        add_additional_fields_to_taskdocs
from atomate.utils.utils import get_fws_and_tasks


from pymatgen.core.surface import generate_all_slabs
from pymatgen.io.vasp.sets import MVLSlabSet, MPStaticSet
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

__author__ = 'Joseph Montoya, Arunima Singh'
__email__ = 'montoyjh@lbl.gov'

def get_hse_bandedge_wf(structure, gap_only=True, vasp_cmd='vasp', 
                        db_file=None, max_index=1, min_slab_size=15.0,
                        min_vacuum_size=15.0, analysis=False, bulk=False):
    """
    Function to return workflow designed to return HSE band edges.
    In development.
    """
    # Get conventional structure
    structure = SpacegroupAnalyzer(structure).get_conventional_standard_structure()
    sgp = {"min_slab_size": min_slab_size,
           "min_vacuum_size": min_vacuum_size}
    slabs = generate_all_slabs(structure, max_index=max_index, **sgp)
    
    # Get min volume slab, try to get symmetric slab
    slabs = sorted(slabs, key=lambda x: (x.volume, not x.is_symmetric(), 
                                         len(SpacegroupAnalyzer(x).get_symmetry_operations())))
    # Find first three entries with distinct miller indices, not the most elegant
    slabs_to_calculate = [slabs[0]]
    slabs_to_calculate += [[slab for slab in slabs if slab.miller_index != slabs[0].miller_index][0]]
    slabs_to_calculate += [[slab for slab in slabs if slab.miller_index 
                            not in [s.miller_index for s in slabs_to_calculate]][0]]
    assert len(slabs_to_calculate) == 3, 'Length of slabs is not 3'

    # TODO: decide whether we want to optimize slab, slab thickness, etc.
    # Since these are semiconductors, shouldn't need as much k-point density
    slab_fws = []
    for slab in slabs_to_calculate:
        vis = MVLSlabSet(slab, k_product=30)
        cparams = {"vasp_cmd": vasp_cmd, "db_file": db_file}
        slab_fws += [get_slab_fw(slab, bulk_structure=structure, vasp_input_set=vis,
                                 slab_gen_params=sgp, name="slab", **cparams)]
        slab_fws.append(NonSCFFW(slab, parents=slab_fws[-1], name="slab", **cparams))
        slab_fws.append(HSEBSFW(slab, parents=slab_fws[-1], name="slab_hse", **cparams))

    wf_slab = Workflow(slab_fws, name = '{} bandedge'.format(structure.formula))

    if bulk:
        # Create bulk workflow and append slab WF
        wf = wf_bandstructure_plus_hse(structure, gap_only=gap_only)

        wf.append_wf(wf_slab, wf.root_fw_ids)
        if analysis:
            wf_analysis = Workflow([
                Firework(BandedgesToDb(structure=structure, db_file=db_file, fw_spec_field='tags'),
                         name="Bandedge analysis")])
            wf.append_wf(wf_analysis, wf.leaf_fw_ids)
    else:
        wf = wf_slab
    
    # Turn off NPAR for HSE fireworks
    # NOTE: remove for now, HSE may need more memory
    # wf = add_modify_incar(wf, modify_incar_params={"incar_update": {"NPAR": 1}}, 
    #                       fw_name_constraint='hse')

    # Turn on LVTOT in relevant FWs
    slab_incar_params = {"incar_update": {"LVTOT": True, "PREC": "High", "ENAUG": 2000}}
    wf = add_modify_incar(wf, modify_incar_params=slab_incar_params, 
                          fw_name_constraint="slab")

    # Modify slab handlers
    wf = modify_handlers(wf, SLAB_HANDLERS, fw_name_constraint='slab')
    wf = add_additional_fields_to_taskdocs(wf, {"parent_structure": structure})
    return wf


if __name__ == "__main__":
    # Here's an example of the usage of the workflow
    from pymatgen import MPRester
    from fireworks import LaunchPad
    mpr = MPRester()
    lpad = LaunchPad.auto_load()
    mp_ids = ['mp-149', 'mp-2657', 'mp-24972', 'mp-19443', 'mp-19142']
    for mp_id in mp_ids:
        structure = mpr.get_structure_by_material_id(mp_id)
        for prop in structure.site_properties:
            structure.remove_site_property(prop)
        wf = get_hse_bandedge_wf(structure, vasp_cmd=">>vasp_cmd<<",
                                 db_file=">>db_file<<")
        wf = add_modify_incar(wf)
        wf = add_tags(wf, ["Bandedge_benchmark_2", mp_id])
    lpad.add_wf(wf)
