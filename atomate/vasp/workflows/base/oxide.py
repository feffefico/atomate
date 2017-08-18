"""
This module defines a workflow for OER on O-terminated oxide surfaces
"""

import numpy as np
from copy import copy
from fireworks import Workflow, Firework, FiretaskBase, explicit_serialize
from atomate.utils.utils import get_logger
from atomate.vasp.fireworks.core import OptimizeFW, TransmuterFW, StaticFW
from atomate.vasp.powerups import add_tags
from atomate.vasp.firetasks.parse_outputs import OERAnalysisTask

from pymatgen.core.sites import PeriodicSite
from pymatgen.core.operations import SymmOp
from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from pymatgen.core.surface import generate_all_slabs
from pymatgen.transformations.advanced_transformations import SlabTransformation
from pymatgen.transformations.standard_transformations import SupercellTransformation
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.vasp.sets import MVLSlabSet, MPRelaxSet
from pymatgen import Structure, Lattice, Molecule
from pymatgen.util.coord_utils import find_in_coord_list

__author__ = 'Joseph Montoya'
__email__ = 'montoyjh@lbl.gov'

logger = get_logger(__name__)

# Default parameters for slabs (sip) and adsorbates (aip)
# Note: turned off symmetry
#default_sip = {"ISIF": 0, "EDIFFG": -0.05, "ISYM":0}
default_aip = {"ISIF": 0, "AMIX": 0.1, "AMIX_MAG": 0.4, "ENCUT":500,
               "BMIX": 0.0001, "ISYM":0, "BMIX_MAG": 0.0001, 
               "POTIM": 0.6, "EDIFFG": -0.05, "IBRION": 2, "EDIFF":1e-6}
default_sip = default_aip
default_slab_gen_params = {"max_index": 1, "min_slab_size": 12.0, "min_vacuum_size": 20.0,
                           "center_slab": True}


def get_wfs_oxide_from_bulk(structure, gen_slab_params={}, vasp_input_set=None, 
                            vasp_cmd=">>vasp_cmd<<", db_file=">>db_file<<", 
                            auto_dipole=False, slab_incar_params=None, ads_incar_params=None, 
                            optimize_slab=False, name="", ads_structures_params={},
                            output=False, symmetrize_structs=True):
    """
    This workflow is intended to construct a workflow for
    OER intermediates on an oxide for OER.
    """
    gsp = default_slab_gen_params.copy()
    gsp.update(gen_slab_params)
    slabs = generate_all_slabs(structure, **gsp)
    # TODO: deal with polar slabs... this should probably be dealt with already.
    for slab in copy(slabs):
        if not slab.is_symmetric():
            so = SymmOp.inversion(origin=np.sum(slab.lattice.matrix*0.5, axis=0))
            new_slab = slab.copy()
            new_slab.apply_operation(so)
            slabs.append(new_slab)
    o_term = []
    slab_incar_params = slab_incar_params or default_sip
    ads_incar_params = ads_incar_params or default_aip
    wfs = []
    # Refine slabs:
    ref_slabs = []
    for slab in slabs:
        asf_narrow = AdsorbateSiteFinder(slab, height=0.02)
        surf = [site.species_string for site in asf_narrow.surface_sites]
        if 'O' in surf:
            ref_slabs.append(slab)
            if not all([i=='O' for i in surf]):
                # Add a "depleted slab" if topmost layer
                # contains planar coordinated metal
                depleted = slab.copy()
                nonO_surf_sites = [site for site in asf_narrow.surface_sites
                                   if site.species_string != "O"]
                indices = [find_in_coord_list(slab.cart_coords, site.coords) 
                           for site in nonO_surf_sites]
                depleted.remove_sites(indices)
                ref_slabs.append(depleted)

    for slab in ref_slabs:
        mi_string = ''.join([str(i) for i in slab.miller_index])
        sd = [[True, True, True] if site.frac_coords[2] >= 0.5
              else [False, False, False] for site in slab]
        slab.add_site_property("selective_dynamics", sd)
        asf = AdsorbateSiteFinder(slab, height=1.0, ptol=0.2)
        asf_slab = asf.slab.copy()
        site_props = asf_slab.site_properties
        for n, specie in enumerate(asf_slab.species):
            if specie.name != "O":
                site_props["surface_properties"][n] = "subsurface"
        for key, value in site_props.items():
            asf_slab.add_site_property(key, value)
        if auto_dipole:
            weights = np.array([site.species_and_occu.weight for site in slab])
            dipole_center = np.sum(weights*np.transpose(slab.frac_coords), axis=1)
            dipole_center /= np.sum(weights)
            dipole_dict = {"LDIPOL": "True", "IDIPOL": 3, "DIPOL": dipole_center}
            slab_incar_params.update(dipole_dict)
            ads_incar_params.update(dipole_dict)
        molecules = [Molecule("H", [[ -8.15221440e-01,   4.35023640e-01,   3.23265180e-01]]),
                     Molecule("OH", [[-1.16199703, -0.33129907, 0.78534692],
                                     [ -0.77836053, -0.2210624, 1.68471111]])]
        asf_new = AdsorbateSiteFinder(asf_slab)
        asp = {"min_lw":5.0, "find_args":{"distance":0.0, "positions":['ontop']}}
        ads_sites = asf_new.find_adsorption_sites(**asp["find_args"])["ontop"]
        h_structs = asf_new.generate_adsorption_structures(molecules[0], **asp)
        vac_structs = []
        for n, struct in enumerate(h_structs):
            this = struct.copy()
            o_index = find_in_coord_list(this.cart_coords, ads_sites[n])
            this.pop(o_index)
            this.pop(-1)
            vac_structs += [this]
        oh_structs = asf_new.generate_adsorption_structures(molecules[1], **asp)
        # Add slab metadata
        name = slab.composition.reduced_formula
        name += "_"+"".join([str(i) for i in slab.miller_index])
        if symmetrize_structs:
            slab = symmetrize_slab_by_addition(slab)
            h_structs = [symmetrize_slab_by_addition(h_struct) for h_struct in h_structs]
            oh_structs = [symmetrize_slab_by_addition(oh_struct) for oh_struct in oh_structs]
            vac_structs = [symmetrize_slab_by_addition(vac_struct) for vac_struct in vac_structs]

        vis_slab = MVLSlabSet(slab, user_incar_settings=slab_incar_params)
        fws = [StaticFW(structure=slab, vasp_input_set=vis_slab, vasp_cmd=vasp_cmd,
                        db_file=db_file, name="{} slab".format(name))]
        for n in range(len(oh_structs)):
            # Warning: this is sloppy
            vis_slab = MVLSlabSet(h_structs[n], user_incar_settings=slab_incar_params)
            fws.append(StaticFW(structure=h_structs[n], vasp_input_set=vis_slab, vasp_cmd=vasp_cmd,
                                db_file=db_file, name="{} H ads {}".format(name, n)))
            vis_slab = MVLSlabSet(oh_structs[n], user_incar_settings=slab_incar_params)
            fws.append(StaticFW(structure=oh_structs[n], vasp_input_set=vis_slab, vasp_cmd=vasp_cmd,
                                db_file=db_file, name="{} OH ads {}".format(name, n)))
            vis_slab = MVLSlabSet(vac_structs[n], user_incar_settings=slab_incar_params)
            fws.append(StaticFW(structure=vac_structs[n], vasp_input_set=vis_slab, vasp_cmd=vasp_cmd,
                                db_file=db_file, name="{} vac {}".format(name, n)))
        """
        fws.append(Firework(OERAnalysisTask(slab=slab, db_file=">>db_file<<"), name="OER Analysis", 
                            parents=fws))
        """
        termination = get_termination(slab)
        name += '_{}_terminated'.format(termination)
        wfname = "{}:{}".format(name, " OER calculations")
        wf = Workflow(fws, name=wfname)
        if output:
            filename = "{}_{}_term_{}.cif".format(slab.composition.reduced_formula,
                                                  termination, mi_string)
            print "Writing {}".format(filename)
            slab.to(filename=filename)
        wf = add_tags(wf, ["{} terminated".format(termination),
                           "{} index".format(mi_string),
                           "oxide v0.1"])
        wfs.append(wf)
    return wfs


def get_termination(slab, start=0.2):
    temp_slab = Structure(slab.lattice, slab.species, slab.frac_coords)
    asf = AdsorbateSiteFinder(temp_slab, height=start)
    surf = [site.species_string for site in asf.surface_sites]
    while all([i=='O' for i in surf]):
        start += 0.1
        asf = AdsorbateSiteFinder(temp_slab, height=start)
        surf = [site.species_string for site in asf.surface_sites]
    return Structure.from_sites(asf.surface_sites).composition.reduced_formula

def symmetrize_slab_by_addition(slab, sga_params={}, recenter=True, sd_height=3.0,
                                direction='top'):
    """
    This method checks whether or not the two surfaces of the slab are
    equivalent. If the point group of the slab has an inversion symmetry (
    ie. belong to one of the Laue groups), then it is assumed that the
    surfaces should be equivalent. Otherwise, sites at the bottom of the
    slab will be removed until the slab is symmetric. Note that this method
    should only be limited to elemental structures as the removal of sites
    can destroy the stoichiometry of the slab. For non-elemental
    structures, use is_polar().

    Arg:
        slab (Structure): A single slab structure
        sga_params (dict): kwargs for SpacegroupAnalyzer
        sd (float): selective dynamics flag, freezes N of the central atoms

    Returns:
        Slab (structure): A symmetrized Slab object.
    """
    laue = ["-1", "2/m", "mmm", "4/m", "4/mmm",
            "-3", "-3m", "6/m", "6/mmm", "m-3", "m-3m"]
    if direction=='top':
        index_fn = np.argmax
    elif direction=='bottom':
        index_fn = np.argmin
    else:
        raise ValueError('direction must be top or bottom')

    # TODO: this might rely on having a centered slab
    slab_old = slab.copy()
    slab = slab.copy()
    sg = SpacegroupAnalyzer(slab, **sga_params)
    pg = sg.get_point_group_symbol()

    if str(pg) in laue:
        return slab
    else:
        asym = True
        removed_sites = []
        while asym or len(slab) < 2:
            temp_slab = Structure(slab.lattice, slab.species, slab.frac_coords)
            # Keep removing sites from the top one by one until both
            # surfaces are symmetric or the number of sites removed has
            # exceeded 10 percent of the original slab

            index = index_fn(slab.frac_coords[:, 2])
            removed_sites.append(slab.pop(index))

            # Check if the altered surface is symmetric

            sg = SpacegroupAnalyzer(slab, **sga_params)
            pg = sg.get_point_group_symbol()

            if str(pg) in laue:
                asym = False

    # Find inversion center
    center = np.average(slab.frac_coords, axis=0)
    symmops = sg.get_symmetry_operations()
    inv_symmops = [symmop for symmop in symmops
                   if (symmop.rotation_matrix==-np.eye(3)).all()
                   and (symmop.translation_vector[:2]==0).all()]
    assert len(inv_symmops)==1, "More than one inv_symmop"
    inv_so = inv_symmops[0]
    for site in removed_sites:
        inv_fcoords = inv_so.operate(site.frac_coords)
        slab.append(site.specie, site.frac_coords, properties=site.properties)
        slab.append(site.specie, inv_fcoords, properties=site.properties)
    if recenter:
        slab.translate_sites(list(range(len(slab))), [0, 0, 0.5 - np.average(slab.frac_coords[:, 2])])

    if restoich:
        # copy in smallest direction
        if slab.lattice.a > slab.lattice.b:
            slab.make_supercell([1, 2, 1])
        else:
            slab.make_supercell([2, 1, 1])
        indices = range(slab.num_sites - 2*removed_sites, slab.num_sites)
        slab.remove_sites(indices=indices)
        assert slab_old.composition == slab.composition, "Restoich failed"

    if sd_height:
        mvec = AdsorbateSiteFinder(slab).mvec
        projs = [np.dot(coord, mvec) for coord in slab.cart_coords]
        mx, mn = max(projs), min(projs)
        sd = [[False]*3 if mn+sd_height <= proj <= mx-sd_height 
              else [True]*3 for proj in projs]
        slab.add_site_property("selective_dynamics", sd)
    assert slab.is_symmetric, "resultant slab not symmetric"
    return slab


def decorate_bulk_coord(structure, radius=2.5):
    """
    Assigns bulk coordination dictionary that can help
    identify undercoordinated sites and their missing
    coordinated atoms
    """
    # TODO: figure out sensible radius 
    # TODO: referencing, rotations?
    neighbor_sets = structure.get_all_neighbors(r=radius)
    props = []
    for n, neighbor_set in enumerate(neighbor_sets):
        sites_by_species = groupby(neighbor_set, lambda x: x[0].species_string)
        props.append({k: [site[0].coords - structure[n].coords for site in g]
                      for k, g in sites_by_species})
    new_struct = structure.copy()
    new_struct.add_site_property("bulk_neighbors", props)
    return new_struct


@explicit_serialize
def PassEnergyStructureTask(FireTaskBase):
    """
    Pass energy/structure?
    """
    def run_task(self):
        pass

if __name__=="__main__":
    from pymatgen import MPRester
    from personal.functions import pdb_function
    from pymatgen.core.surface import SlabGenerator
    from atomate.vasp.powerups import add_modify_incar
    mpr = MPRester()
    structure = mpr.get_structures("mp-5229")[0]
    wfs = pdb_function(get_wfs_oxide_from_bulk, structure, ads_structures_params={"repeat":[1, 1, 1]})
    slabs = generate_all_slabs(structure, **default_slab_gen_params)
    so = SymmOp.inversion()
    new_slab = slabs[0].copy()
    new_slab.apply_operation(so, fractional=True)
    symm_slab = pdb_function(symmetrize_slab_by_addition, slabs[0])
    """
    slabs[0].to(filename="111.cif")
    new_slab.to(filename="111_inv.cif")
    """
    #symm_slab.to(filename="POSCAR")
    """
    for n, fw in enumerate(wfs[0].fws[1:]):
        fw.tasks[0]['structure'].to(filename="ads_{}.cif".format(n))
    """
    from fireworks import LaunchPad
    wfs_mod = [add_modify_incar(wf) for wf in wfs]
    wfs_mod = [add_modify_incar(wf, {"incar_update":{"IBRION":2, "NSW":"25"}})]
    lpad = LaunchPad.auto_load()
    lpad.add_wf(wfs_mod[0])

