from atomate.vasp.fireworks.core import OptimizeFW, StaticFW
from fireworks import Workflow
from pymatgen.io.vasp.sets import MPRelaxSet, MPStaticSet
from atomate.vasp.powerups import add_tags
from atomate.utils.utils import get_fws_and_tasks

def get_wf_magnetism(structure, vasp_input_set=None, vasp_cmd="vasp", db_file=None):
    """
    Quick implementation of a workflow that 
    1. Calculate non-spin-polarized structure optimization and static
    2. Calculate spin-polarized AFM structure optimization and static
    3. Calculate spin-polarized FM structure optimization and static

    Notes:
        The way this is constructed isn't ideal, but I've added tags
        so that it's easier to query for the AFM, FM, and non-spin cases
    """
    # TODO: enable custom vasp input sets
    # TODO: this assumes that the input structure has an AFM setting,
    magmoms = structure.site_properties.get("magmom", None)
    
    # Test for has magmoms
    assert magmoms, "Structure must have magnetic moments in site properties"
    # Quick test for AFM (in this case presence of negative values)
    assert abs(sum(magmoms)) < 1e-10, "Structure must be AFM"

    fws = []
    formula = structure.composition.reduced_formula
    # ISPIN 1
    vis_nospin = MPRelaxSet(structure, user_incar_settings={"ISPIN":1})
    fws.append(OptimizeFW(structure, vasp_input_set=vis_nospin, vasp_cmd=vasp_cmd,
                          db_file=db_file, name="Non-spin opt".format(formula)))
    fws.append(StaticFW(structure, vasp_input_set=vis_nospin, parents=[fws[-1]],
                        vasp_cmd=vasp_cmd, db_file=db_file,
                        name = "Non-spin static".format(formula)))

    # ISPIN 2, magmoms from JSON
    vis_afm = MPRelaxSet(structure)
    fws.append(OptimizeFW(structure, vasp_input_set=vis_afm, vasp_cmd=vasp_cmd, db_file=db_file,
                          name="AFM opt".format(formula)))
    fws.append(StaticFW(structure, vasp_input_set=vis_afm, parents=[fws[-1]], vasp_cmd=vasp_cmd, db_file=db_file,
                        name = "AFM static".format(formula)))

    # ISPIN 2, magmoms from JSON all positive
    fm_structure = structure.copy()
    fm_structure.add_site_property("magmom", [abs(m) for m in magmoms])
    vis_fm = MPRelaxSet(fm_structure)
    fws.append(OptimizeFW(fm_structure, vasp_input_set=vis_fm, vasp_cmd=vasp_cmd, db_file=db_file,
                     name="FM opt".format(formula)))
    fws.append(StaticFW(fm_structure, vasp_input_set=vis_fm, parents=fws[-1], vasp_cmd=vasp_cmd, db_file=db_file,
                        name = "FM static".format(formula)))

    # Add tags to make querying a little easier
    return Workflow(fws, name="{} - magnetic calculations".format(formula))

if __name__=="__main__":
    from pymatgen.util.testing import PymatgenTest
    from fireworks import LaunchPad
    si = PymatgenTest.get_structure("Si")
    si.add_site_property("magmom", [-1, 1])
    wf = get_wf_magnetism(si)
    lpad = LaunchPad.auto_load()
    lpad.add_wf(wf)
