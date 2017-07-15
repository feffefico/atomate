# coding: utf-8

from __future__ import absolute_import, division, print_function, unicode_literals

"""
This module defines a workflow for adsorption on surfaces
"""

import numpy as np

from fireworks import Workflow, Firework

from atomate.vasp.fireworks.core import OptimizeFW, TransmuterFW, StaticFW
from atomate.vasp.firetasks.glue_tasks import pass_vasp_result
from atomate.vasp.firetasks.parse_outputs import SlabToDb

from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from pymatgen.core.surface import generate_all_slabs, Slab
from pymatgen.transformations.advanced_transformations import SlabTransformation
from pymatgen.transformations.standard_transformations import SupercellTransformation
from pymatgen.io.vasp.sets import MVLSlabSet
from pymatgen.io.vasp.inputs import Kpoints
from pymatgen import Structure, Lattice, Molecule
from pymatgen.entries.computed_entries import ComputedEntry

__author__ = 'Joseph Montoya'
__email__ = 'montoyjh@lbl.gov'

se_pass_dict = {'energy': '>>output.final_energy',
                'structure': '>>output.crystal'}

def get_slab_fw(slab, bulk_structure=None, slab_gen_params={}, db_file=None, vasp_input_set=None,
                copy_vasp_outputs=False, vasp_cmd="vasp", name="", **kwargs):
    """
    Function to generate a a slab firework.  Returns a TransmuterFW if bulk_structure is specified,
    constructing the necessary transformations from the slab and slab generator parameters,
    or an OptimizeFW if only a slab is specified.

    Args:
        slab (Slab or Structure): structure or slab corresponding
            to the slab to be calculated
        bulk_structure (Structure): bulk structure corresponding to slab, if
            provided, slab firework is constructed as a TransmuterFW using
            the necessary transformations to get the slab from the bulk
        slab_gen_params (dict): dictionary of slab generation parameters
            used to generate the slab, necessary to get the slab
            that corresponds to the bulk structure
        vasp_input_set (VaspInputSet): vasp_input_set corresponding to
            the slab calculation
        parents (Fireworks or list of ints): parent FWs
        db_file (string): path to database file
        vasp_cmd (string): vasp command
        **kwargs (kwargs): keyword arguments for Firework

    Returns:
        Firework
    """
    vasp_input_set = vasp_input_set or MVLSlabSet(
            slab, user_incar_settings={"EDIFFG": -0.05}, k_product=30)

    # If a bulk_structure is specified, generate the set of transformations, else
    # just create an optimize FW with the slab
    if bulk_structure:
        if not isinstance(slab, Slab):
            raise ValueError("structure input to get_slab_fw requires slab to be a slab object!")
        slab_trans_params = {"miller_index": slab.miller_index, "shift":slab.shift}
        slab_trans_params.update(slab_gen_params)

        # Get supercell parameters
        trans_struct = SlabTransformation(**slab_trans_params)
        slab_from_bulk = trans_struct.apply_transformation(bulk_structure)
        supercell_trans = SupercellTransformation.from_scaling_factors(
                round(slab.lattice.a / slab_from_bulk.lattice.a),
                round(slab.lattice.b / slab_from_bulk.lattice.b))

        # Get adsorbates for InsertSitesTransformation
        if "adsorbate" in slab.site_properties.get("surface_properties", [None]):
            ads_sites = [site for site in slab if site.properties["surface_properties"]=="adsorbate"]
        else:
            ads_sites = []
        transformations = ["SlabTransformation", "SupercellTransformation",
                           "InsertSitesTransformation", "AddSitePropertyTransformation"]
        trans_params = [slab_trans_params, {"scaling_matrix":supercell_trans.scaling_matrix},
                        {"species": [site.species_string for site in ads_sites],
                         "coords": [site.frac_coords for site in ads_sites]},
                        {"site_properties": slab.site_properties}]
        return TransmuterFW(name=name, structure=bulk_structure, transformations=transformations,
                            transformation_params=trans_params, copy_vasp_outputs=copy_vasp_outputs, 
                            db_file=db_file, vasp_cmd=vasp_cmd, vasp_input_set=vasp_input_set, **kwargs)
    else:
        return OptimizeFW(name=name, structure=slab, vasp_input_set=vasp_input_set, vasp_cmd=vasp_cmd,
                          db_file=db_file, job_type="normal", **kwargs)


def get_wf_surface(slab, adsorbates=[], bulk=None, slab_gen_params=None, 
                   vasp_cmd="vasp", db_file=None, ads_structures_params={}, 
                   asf_params={}, reference_molecules=[], analysis=True):
    """

    Args:
        slab (Slab or Structure): slab to calculate
        adsorbates ([Molecule]): molecules to place as adsorbates
        bulk_structure (Structure or ComputedEntry): bulk structure from which generate slabs
            after reoptimization.  If supplied, workflow will begin with
            bulk structure optimization.
        slab_gen_params (dict): dictionary of slab generation parameters
            used to generate the slab, necessary to get the slab
            that corresponds to the bulk structure if in that mode
        asf_params (dict): parameters to be supplied as kwargs to AdsorbateSiteFinder
            __init__ constructor
        ads_structures_params (dict): parameters to be supplied as
            kwargs to AdsorbateSiteFinder.generate_adsorption_structures
        db_file (string): path to database file
        vasp_cmd (string): vasp command
        reference_molecules ([Molecule or ComputedEntry]): list of molecular
            references.  If molecules are Molecule objects, vasp FWs to calculate
            their energies are appended.  If ComputedEntries, the computed
            entries are passed.
    Returns:
        Workflow
    """
    # Get name from slab info 
    name = slab.composition.reduced_formula
    if getattr(slab, "miller_index", None):
        name += "_{}".format(slab.miller_index)

    wf = None
    # Add bulk
    if isinstance(bulk, Structure):
        vis = MVLSlabSet(bulk, bulk=True)
        bulk_fw = OptimizeFW(bulk, vasp_input_set=vis, vasp_cmd=vasp_cmd, db_file=db_file)
        bulk_fw.tasks.append(pass_vasp_result(filename='vasprun.xml.relax2.gz',
                                              mod_spec_key="bulk"))
        bulk_structure = bulk
        wf = Workflow([bulk_fw], name=name+" surface")
    elif isinstance(bulk, ComputedEntry):
        pass_dict = {"computed_entry": bulk}
        bulk_fw = Firework([pass_vasp_result({"computed_entry": bulk},
                                             mod_spec_key="bulk")])
        wf = Workflow([bulk_fw], name=name+" surface")
        bulk_structure = getattr(bulk, "structure", None)
    elif bulk is None:
        bulk_structure = None
    else:
        raise ValueError("bulk must be Structure or ComputedEntry")


    # Add slab, copy vasp outputs if bulk is a Structure
    fws = []
    copy_vasp_outputs = isinstance(bulk, Structure)

    slab_fw = get_slab_fw(slab, bulk_structure, slab_gen_params, db_file=db_file, 
                          vasp_cmd=vasp_cmd, name=name+" slab optimization",
                          copy_vasp_outputs=copy_vasp_outputs)
    slab_fw.tasks.append(pass_vasp_result(mod_spec_key="slab"))

    fws.append(slab_fw)

    # Add adsorbates
    for adsorbate in adsorbates:
        ads_slabs = AdsorbateSiteFinder(slab, **asf_params).generate_adsorption_structures(
            adsorbate, **ads_structures_params)
        for n, ads_slab in enumerate(ads_slabs):
            ads_name = "{}-{} adsorbate optimization {}".format(adsorbate.composition.formula, name, n)
            ads_fw = get_slab_fw(ads_slab, bulk_structure, slab_gen_params, db_file=db_file,
                                 vasp_cmd=vasp_cmd, name=ads_name)
            ads_fw.tasks.append(pass_vasp_result(
                mod_spec_key="adsorbates->{}->{}".format(adsorbate.composition.formula, n)))
            fws.append(ads_fw)
    
    # Join optimize
    if wf:
        wf_surface = Workflow(fws)
        wf.append_wf(wf_surface, wf.leaf_fw_ids)
    else:
        wf = Workflow(fws, name=name + " surface")

    # Add molecules
    if reference_molecules:
        wf_molecules = get_wf_molecules(reference_molecules, vasp_cmd=vasp_cmd, db_file=db_file)
        wf.append_wf(wf_molecules, [])

    # Add analysis task
    if analysis:
        analysis_fw = Firework([SlabToDb(slab=slab, db_file=db_file)], name="Analyze Slab")
        wf.append_wf(Workflow([analysis_fw]), wf.id_fw.keys())
    return wf


def get_wf_molecules(molecules, vasp_cmd='vasp', db_file=None, box_size=10, **kwargs):
    """
    Helper function that gets a workflow for molecules in VASP,
    can be used in conjunction with the surface workflow or
    independently.
    """
    mol_fws = []

    for n, molecule in enumerate(molecules):
        if isinstance(molecule, Molecule):
            # molecule in box
            m_struct = molecule.get_boxed_structure(*[box_size]*3)
            vis = MVLSlabSet(m_struct, k_product=box_size / 2)
            mol_fw = OptimizeFW(m_struct, job_type="normal", vasp_input_set=vis,
                                db_file=db_file, vasp_cmd=vasp_cmd, handler_group="md")
            mol_fw.tasks.append(pass_vasp_result(mod_spec_key="references->{}".format(n)))
            mol_fws.append(mol_fw)
        elif isinstance(molecule, ComputedEntry):
            pass_dict = {"computed_entry": molecule}
            mol_fw = Firework([pass_vasp_result({"computed_entry": molecule}, 
                                                mod_spec_key="references->{}".format(n))])
            fws.append(mol_fw)
    return Workflow(mol_fws, **kwargs)


def get_wf_surface_all_slabs(bulk_structure, molecules, max_index=1, slab_gen_params=None, **kwargs):
    """
    Convenience constructor that allows a user to construct a workflow
    that finds all adsorption configurations (or slabs) for a given
    max miller index.

    Args:
        bulk_structure (Structure): bulk structure from which to construct slabs
        molecules (list of Molecules): adsorbates to place on surfaces
        max_index (int): max miller index
        slab_gen_params (dict): dictionary of kwargs for generate_all_slabs

    Returns:
        Workflow
    """
    sgp = slab_gen_params or {"min_slab_size": 7.0, "min_vacuum_size": 20.0}
    slabs = generate_all_slabs(bulk_structure, max_index=max_index, **sgp)
    return get_wf_surface(slabs, molecules, bulk_structure, sgp, **kwargs)
