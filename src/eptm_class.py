#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Author  : Alexander Nestor-Bergmann
# Released: 08/03/2021
# =============================================================================
"""Implementation of a class to represent an epithelial tissue comprised of cell cortices and adhesions."""

import copy
import itertools
import warnings
import os
from collections import Counter, deque
from random import shuffle

import dill
import matplotlib
import numpy as np
import shapely.geometry as geom
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from more_itertools import consecutive_groups
from numpy.core.umath_tests import inner1d
from scipy.interpolate import splev, splrep
from scipy.optimize import minimize
from scipy.signal import decimate, savgol_filter
from scipy.spatial import ConvexHull, cKDTree, distance
from shapely.affinity import scale
from sklearn.neighbors import NearestNeighbors

from src.adhesion_class import Adhesion
from src.cell_class import Cell
from src.concave_hull import ConcaveHull

matplotlib.rcParams.update({'font.size': 22})
plt.rc('font', family='serif', serif='Times')
plt.rc('text', usetex=True)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


class Epithelium(object):
    """
    Class to hold and manipulate all objects that constitute a tissue: cells, adhesion and boundaries.

    The adhesion objects, with their cell conection information, are stored here.  This information is passed to each
    cell as a list, rather than adhesion itself, for memory.
    """

    def __init__(self, radius: float = 35, n: int = 2000, param_dict: dict = None, tissue_type: str = 'within_hexagon',
                 disordered_tissue_params: dict = None, cell_kwargs: dict = None, verbose: bool = False, **eptm_kwargs):
        """
        Initialiser which sets up the class properties.

        :param radius: Radius of the initially circular cell.
        :param n: Number of nodes to discretise the cortex.
        :param param_dict: Dictionary with keys ``kappa``, ``omega0`` and ``delta`` with theit starting values.
        :param tissue_type: Can be ``'within_hexagon'`` or ``'disordered'``.
        :param verbose: Boolean, whether to print information to console.
        :param disordered_tissue_params: Dict of params to pass to ``create_disordered_tissue_in_ellipse``
        :param cell_kwargs: Dict of params to pass to ``_build_single_cell`` to initialise cell properties.
        :param eptm_kwargs: Any other parameters for initialising the epithelium.
        """

        self.verbose: bool = verbose
        self.verboseprint("Initialising epithelium", object(), 1)

        #######

        # Initial radius of circular cell
        self.radius: float = radius
        # Mechanical params for cell
        param_dict: dict = {} if param_dict is None else param_dict
        param_dict['delta'] = param_dict.get('delta', 1)
        param_dict['kappa'] = param_dict.get('kappa', 1e-4)
        param_dict['omega0'] = param_dict.get('omega0', 0)
        self.delta: float = param_dict['delta']
        self.n: int = n

        self.boundary_adhesions = [[], []]

        if tissue_type == 'within_hexagon':
            self.within_hexagons: bool = True
            cell_kwargs: dict = {} if cell_kwargs is None else cell_kwargs
            self._build_single_cell(radius=radius, n=n, param_dict=param_dict, verbose=verbose,
                                    identifier='A', cell_kwargs=cell_kwargs)

        elif tissue_type == 'disordered':
            self.within_hexagons: bool = False
            disordered_tissue_params: dict = {} if disordered_tissue_params is None else disordered_tissue_params
            self.create_disordered_tissue_in_ellipse(param_dict=param_dict, **disordered_tissue_params)
        else:
            raise NotImplementedError('Can have only "within_hexagon" or "disordered" tissues')

        # Concave hull
        self.concave_hull_tolerance: float = eptm_kwargs.get('concave_hull_tolerance', 4 * (radius / 1000))
        # if radius > 25:
        #     self.concave_hull_tolerance *= .1

        ############### Solving parameters

        self.age: float = 0

        self.last_num_internal_relaxes: int = eptm_kwargs.get('last_num_internal_relaxes', 0)
        self.total_elastic_relaxes: int = 0
        self.relax_dist_threshold: float = eptm_kwargs.get('relax_dist_threshold', .1)
        self.max_elastic_relax_steps: int = eptm_kwargs.get('max_elastic_relax_steps', 10)
        self.elastic_relax_tol: float = eptm_kwargs.get('elastic_relax_tol', 2e-2)
        self.in_equilibrium: bool = False
        self.bulk_turnover_time: int = eptm_kwargs.get('bulk_turnover_time', 0)
        self.last_solve_success: bool = True

        self.use_mesh_coarsening: bool = eptm_kwargs.get('use_mesh_coarsening', False)
        self.use_mesh_refinement: bool = eptm_kwargs.get('use_mesh_refinement', False)

        ############## Adhesion properties

        self.adhesion_timescale: str = eptm_kwargs.get('adhesion_timescale', 'slow')
        assert self.adhesion_timescale in ["fast",
                                           "slow"], "Error, adhesion timescale %s not valid." % self.adhesion_timescale

        # Sidekick vertices
        self.sidekick_active: bool = eptm_kwargs.get('sidekick_active', False)
        self.sidekick_adhesions: list = []

        # Second population of adhesions with different rates of turnover
        self.slow_adhesions: list = []
        self.slow_adhesion_lifespan: int = eptm_kwargs.get('slow_adhesion_lifespan', 0)

        ############### Initialise the boundary properties

        self.reference_boundary_adhesions: list = []
        self.boundary_bc: str = eptm_kwargs.get('boundary_bc', 'fixed')
        self.boundary_stiffness_x: float = eptm_kwargs.get('boundary_stiffness_x', 2.5e-2)
        self.boundary_stiffness_y: float = eptm_kwargs.get('boundary_stiffness_y', 2.5e-2)
        self.posterior_pull_shift: float = eptm_kwargs.get('posterior_pull_shift', .1)  # Stretch from posterior midgut.

        # Finish up
        if self.within_hexagons:
            self.set_adhesion_to_fixed_line_bool(True)
            self.update_adhesion_points_between_all_cortices()
            self.update_all_rest_lengths_and_areas(dt=0)

    def __getstate__(self):
        return vars(self)

    def __setstate__(self, state):
        vars(self).update(state)

    def __getattr__(self, name):
        """
        Incase an old eptm is being loaded and doesn't have an attribute. Warning, attribute type may be wrong.
        :param name: the name of the attribute
        :return:
        """
        if name.startswith('__') and name.endswith('__'):
            raise AttributeError
        else:
            warnings.warn(f"{name} does not exist. Creating attribute with value None")
            setattr(self, name, None)

    ### Property decorators ###

    @property
    def verbose(self):
        return self._verbose

    @verbose.setter
    def verbose(self, verbose):
        """Turn print statements on or off.

        :param verbose: Whether to print information to console.
        :type verbose:  bool

        """
        if hasattr(self, 'cells'):
            if self.cells is not None:
                for cell in self.cells:
                    cell.verbose = verbose
                self._verbose = verbose

    @property
    def adhesion_timescale(self):
        return self._adhesion_timescale

    @adhesion_timescale.setter
    def adhesion_timescale(self, adhesion_timescale):
        self._adhesion_timescale = adhesion_timescale

        if adhesion_timescale == 'fast':
            self.activate_fast_adhesions(True)
            self.activate_slow_adhesions(False)
        elif adhesion_timescale == 'slow':
            self.activate_fast_adhesions(False)
            self.activate_slow_adhesions(True)

    @property
    def slow_adhesion_lifespan(self):
        return self._slow_adhesion_lifespan

    @slow_adhesion_lifespan.setter
    def slow_adhesion_lifespan(self, slow_adhesion_lifespan):
        self._slow_adhesion_lifespan = slow_adhesion_lifespan

        if slow_adhesion_lifespan > 0:
            self.activate_slow_adhesions(True)

            if any([c.fast_adhesions_active for c in self.cells]):
                warnings.warn("Using slow adhesion while fast adhesions are active. Maybe you want to set "
                              "eptm.activate_fast_adhesions(False)")

    ### Class methods ###

    def verboseprint(self, *args):
        """Function to print out details as code runs

        :param args:  Information to be printed to console.

        """
        try:
            self.verbose
        except AttributeError:
            self.verbose = True
        print(args) if self.verbose else None
        # print if self.verbose else lambda *args, **k: None

    def set_mesh_coarsening_fraction(self, max_fraction_to_remove):
        """Specify what fraction of total nodes can be removed in coarsening

        :param max_fraction_to_remove: Maxiumum fraction of mesh nodes to remove in one pass, when adaptive.
        :type max_fraction_to_remove: float

        """
        for cell in self.cells:
            cell.max_mesh_coarsen_fraction = max_fraction_to_remove

    def double_mesh(self):
        """ Puts new mesh points at mean locations of current.
        """

        for cell in self.cells:
            cell.double_mesh()

    def remesh_all_cortices(self, coarsen=True, refine=True):
        """Add/remove mesh nodes if spacing too big/small

        :param coarsen:  (Default value = True).  Boolean, whether to adaptively remove mesh nodes.
        :type coarsen:  bool
        :param refine:  (Default value = True).  Boolean, whether to adaptively add mesh nodes.
        :type refine:  bool

        """

        # If sdk is on, we need to store the positions as s values, not indices in s, so we can get them back
        # after the mesh changes
        # initialise a dict of empty sets to store vertices for every sdk point so we don't remove them in
        # coarsening
        indices_to_keep = {identifier: set() for identifier in self.cellDict}
        if self.sidekick_active:
            for ad in self.sidekick_adhesions:
                indices_to_keep[ad.cell_1.identifier].add(ad.cell_1_index)
                indices_to_keep[ad.cell_2.identifier].add(ad.cell_2_index)

        for cell in self.cells:
            cell.adaptive_mesh_update(coarsen=coarsen, refine=refine, nodes_to_keep=indices_to_keep[cell.identifier])

        # If we had periodic bcs, we need to re-do them if the active cell changed \todo hardcoded to have a cell B
        if self.boundary_bc == 'periodic' and self.cells[0].s.size != self.cellDict['B'].s.size:
            self._rebuild_periodic_boundary()

        # Get the local indices back from the curvilinear coords.
        if self.sidekick_active:
            for ad in self.sidekick_adhesions:
                ad.update_local_cell_indices_with_s()

        # If we have slow adhesions, remove the ones where the node no longer exists and update the local index.
        if len(self.slow_adhesions) > 0:
            # Create a dict of sets for cell s coords for fast lookup
            cell_s_set_dict = {c: set(self.cellDict[c].s) for c in self.cellDict.keys()}
            remaining_slows = []  # Store the adhesions with coords that remain
            for ad in self.slow_adhesions:
                # If both sides of the adhesion still exist, keep it.
                if ad.cell_1_s in cell_s_set_dict[ad.cell_1.identifier] and \
                        ad.cell_2_s in cell_s_set_dict[ad.cell_2.identifier]:
                    remaining_slows.append(ad)
                    ad.update_local_cell_indices_with_s()
            self.slow_adhesions = remaining_slows

            # self.update_slow_adhesions()

    def adjust_intersections(self):
        """Applies a hand-of-god correction to prevent intersections of cortices.

        """

        self.verboseprint("Applying hand-of-god hard-body correction", object(), 1)

        for cell in self.cells:
            cell.apply_hand_of_god_hard_body_correction()

    def set_adhesion_to_fixed_line_bool(self, yes_no):
        """Define whether the elements are adhering to fixed line (a fixed boundary, rather than other cells).

        :param yes_no:  Set to fixed line or not.
        :type yes_no: bool

        """

        self.verboseprint("Setting cells.fixed_adhesion to %s" % yes_no, object(), 1)

        for cell in self.cells:
            cell.fixedAdhesion = yes_no

    def set_adhesion_beta_scale_factor(self, adhesion_beta_scale_factor):
        """ Set the scale factor to normalise fast adhesion forces in the meanfield.

        :param adhesion_beta_scale_factor:  Exponential scaling factor.
        :type adhesion_beta_scale_factor:  float

        """

        self.verboseprint("Setting cells.adhesion_beta_scale_factor to %s" % adhesion_beta_scale_factor, object(), 1)

        for cell in self.cells:
            cell.adhesion_beta_scale_factor = adhesion_beta_scale_factor

    def activate_fast_adhesions(self, on_off):
        """Turn forces from fast adhesions on or off.  Even when off, fast adhesions are used to calculate active contractility.

        :param on_off: Turn forces from fast adhesions on or off.
        :type on_off: bool

        """
        for cell in self.cells:
            cell.fast_adhesions_active = on_off

    def activate_slow_adhesions(self, on_off):
        """Turn slow adhesion forces on or off

        :param on_off: Activate slow adhesions, or not.
        :type on_off: bool

        """
        self.slow_adhesions_active = on_off

    def activate_sidekick_adhesions(self, on_off):
        """Turn sdk vertex forces on or off

        :param on_off:  Activate sidekick forces or not.
        :type on_off: bool

        """
        self.sidekick_active = on_off

    def set_sdk_stiffness(self, stiffness):
        """Set the stiffness of an sdk bond

        :param stiffness:  The bond stiffness.
        :type stiffness: float

        """
        for cell in self.cells:
            cell.sdk_stiffness = stiffness

    def set_sdk_restlen(self, rest_len):
        """Set the stiffness of an sdk bond

        :param rest_len:  The bond rest length.
        :type rest_len: float

        """
        for cell in self.cells:
            cell.sdk_restlen = rest_len

    def update_all_rest_lengths_and_areas(self, apply_to: str = 'all', dt: float = None):
        """Update the rest lengths, S_0 <- s, and areas of specified cortices under viscous model

        :param apply_to:  (Default value = 'all')  If not ``'all'``, a list of cell identifiers that the function will be applied to.
        :type apply_to: list
        :param dt:  (Default value = cortex_timestep)  Discretised time to move forward.
        :type dt: float

        """

        self.verboseprint("Updating all rest lengths", object(), 1)

        apply_to = [c.identifier for c in self.cells] if apply_to == "all" else apply_to

        self.update_all_rest_lengths(apply_to=apply_to, dt=dt)
        self.update_pref_areas(apply_to=apply_to)

        # If we have slow or sidekick adhesions, the s coordinates need to be updated to their new values after
        # the viscous update.
        for ad in self.slow_adhesions:
            ad.update_s_by_local_cell_indices()
        for ad in self.sidekick_adhesions:
            ad.update_s_by_local_cell_indices()

        if self.use_mesh_coarsening or self.use_mesh_refinement:
            self.remesh_all_cortices(coarsen=self.use_mesh_coarsening, refine=self.use_mesh_refinement)

    def update_all_rest_lengths(self, apply_to: str = 'all', dt: float = None):
        """Update the rest lengths, S_0 <- s, of specified cortices under viscous model

        :param apply_to:  (Default value = 'all')  If not ``'all'``, a list of cell identifiers that the function will be applied to.
        :type apply_to: list
        :param dt:  (Default value = cortex_timestep)  Discretised time to move forward.
        :type dt: float

        """
        self.verboseprint("Updating all rest lengths", object(), 1)

        apply_to = [c.identifier for c in self.cells] if apply_to == "all" else apply_to

        # Reset lengths
        for cell_ref in apply_to:
            self.cellDict[cell_ref].update_reference_configuration(dt=dt)

    def update_pref_areas(self, area=None, apply_to='all'):
        """Update the preferred area of specified cells \todo make this a cell method.

        :param area:  (Default value = None)  New preferred area.
        :type area: float
        :param apply_to:  (Default value = 'all')  If not ``'all'``, a list of cell identifiers that the function will be applied to.
        :type apply_to: list

        """
        apply_to = [c.identifier for c in self.cells] if apply_to == "all" else apply_to

        for ref in apply_to:
            cell = self.cellDict[ref]
            if self.bulk_turnover_time is None or self.bulk_turnover_time == 0:
                new_pref_area = area if area is not None else cell.get_area()
                cell.pref_area = new_pref_area
            else:
                cell_area = cell.get_area()
                cell.pref_area = cell.pref_area + (cell_area - cell.pref_area) / self.bulk_turnover_time

    def set_cortical_timescale(self, timescale):
        """Specify the turnover time of the actin cortex.

        :param timescale: New cortical timescale
        :type timescale: float

        """

        for cell in self.cells:
            cell.cortical_turnover_time = timescale

    def update_pressures(self, pressure, apply_to='all'):
        """Update the pressure acting on the cortex, from the cell bulk.

        :param pressure:  Magnitude of pressure force in cell bulk.
        :type pressure: float
        :param apply_to:  (Default value = 'all')  If not ``'all'``, a list of cell identifiers that the function will be applied to.
        :type apply_to: list

        """
        apply_to = [c.identifier for c in self.cells] if apply_to == "all" else apply_to

        for ref in apply_to:
            cell = self.cellDict[ref]
            cell.pressure = pressure

    def update_all_max_adhesion_lengths(self, new_length, update_adhesion=False):
        """Update the maximum adhesion lengths.  This is the maximum length for force calculation, not prestretch
        application

        :param new_length:  New rest length for adhesions.
        :type new_length: float
        :param update_adhesion:  (Default value = False)  Whether to refresh the adhesion connection lists.
        :type update_adhesion: bool

        """
        for cell in self.cells:
            cell.max_adhesion_length = new_length

        if update_adhesion:
            self.update_adhesion_points_between_all_cortices()

    def set_adhesion_type(self, new_type):
        """Set how the force for fast adhesions is calculated
        "meanfield": update max search length to always find a neighbour, regarless of how far, and calulate the
        meanfield force.  But the force is set to zero for adhesions longer than max_adhesion_length anyway.
        "nearest": Use only the nearest neighbouring node.
        "fixed_radius":  Attach to all neighbours within 'adhesion_search_radius' and calculate meanfield force.

        :param new_type:  The rule for calculating adhesion forces.
        :type new_type: string

        """
        assert new_type in ["meanfield", "nearest", "fixed_radius"], "Error, adhesion type %s not valid." % new_type

        for cell in self.cells:
            cell.adhesion_type = new_type

    def set_fast_adhesion_force_law(self, new_type):
        """Constitutive properties of fast adhesions

        :param new_type:  The new rule for their constitutive properties.
        :type new_type: string

        """
        assert new_type in ["spring"], "Error, adhesion force law, %s, not valid." % new_type

        for cell in self.cells:
            cell.adhesion_force_law = new_type

    def set_max_num_adhesions_for_fast_force_calculation(self, num_ads):
        """Number of adhesions to use when calculating the force for fast adhesions.

        :param num_ads: The max num adhesions.
        :type num_ads: int

        """

        for cell in self.cells:
            cell.max_num_adhesions_for_force = num_ads

    def set_adhesion_search_radius(self, radius):
        """Sets the range within which to look for fast adhesion binding pairs. This is used for applying prestretch.
        For fast adhesion forces, the force is set to zero for lengths above 'max_adhesion_length', such that
        max_adhesion_length must be <= adhesion_search_radius to work properly
        It is redundant if adhesion_type=meanfield, which adjusts the value to make sure all are connected.

        :param radius:  The new search radius.
        :type num_ads: float

        """

        for cell in self.cells:
            cell.adhesion_search_radius = radius

    def set_constitutive_model(self, model):
        """Updates the constitutive model on all the cortices

        :param model:  The new constitutive model.
        :type model: string

        """

        assert model in ["linear"], "Error, %s constitutive model is not valid." % model

        self.verboseprint("Setting constitutive model to %s" % model, object(), 1)

        for cell in self.cells:
            cell.constitutive_model = model
            cell.possible_constitutive_models = ["linear"]

    def set_prestrain_type(self, prestrain_type):
        """Updates the way the prestretch is calculated for all cells.

        :param prestrain_type:  The new prestrain type.
        :type prestrain_type: string

        """

        assert prestrain_type in ["most_common", "average", "min", "nearest"], "Error self.prestrain_type " \
                                                                               "not in [most_common, average, min, nearest]."

        for cell in self.cells:
            cell.prestrain_type = prestrain_type

    def set_area_stiffness(self, stiffness):
        """Set area stiffness.

        :param stiffness:  The new bulk stiffness.
        :type stiffness: float

        """
        for cell in self.cells:
            cell.area_stiffness = stiffness

    def set_relax_tolerance_for_cells(self, tol):
        """Tolerance for getting the tissue to elastic quilibrium

        :param tol:  New relax tolerance.
        :type tol: float

        """

        for cell in self.cells:
            cell.relax_tol = tol

    def scale_cells_to_fit_adhesion_to_delta(self):
        """Scale all cells such that their adhesions are within delta apart
        Note, we do them in sequence, rather than parallel, so no cells end up overlapping

        """

        self.verboseprint("Scaling cells to fit min adhesion spacing to delta", object(), 1)

        # Choose the largest (length) cell to start.
        cell_list = [c for c in self.cells]
        area_order = np.argsort([-cell.get_length() for cell in cell_list])
        cell_list = [cell_list[idx] for idx in area_order]
        # shuffle(cell_list)
        for cell in cell_list:
            cell.scale_whole_cell_to_fit_adhesion_to_delta()
            if not self.within_hexagons:
                self.update_adhesion_points_between_all_cortices()

    def hand_of_god_distances_to_delta(self, initial_adhesion_update=True):
        """Moves any cortices closer than delta away from each other

        :param initial_adhesion_update:  (Default value = True) Whether to re-calculate the adhesions for all cells
        :type initial_adhesion_update: bool

        """

        if initial_adhesion_update:
            self.update_adhesion_points_between_all_cortices(build_trees=True)
        moved_any = False
        for cell in self.cells:
            this_move = cell.move_cortex_nodes_to_equilibrium_dist(scaling=1)
            moved_any = this_move if this_move else moved_any
        if moved_any:
            self.update_adhesion_points_between_all_cortices(build_trees=True)

    def set_omega_for_cells(self, new_omega, apply_to='all'):
        """ Updates the adhesion strength for specified cells.

        :param new_omega:  New adhesion strength.
        :type new_omega: float
        :param apply_to:  (Default value = 'all')  List of cells to apply function to.
        :type apply_to: list

        """
        apply_to = [c.identifier for c in self.cells] if apply_to == "all" else apply_to

        for cell_ref in apply_to:
            self.cellDict[cell_ref].omega = new_omega

    def apply_prestretch_to_whole_cells(self, prestretch, cells):
        """Apply a prestretch to the whole cortex of specified cells.

        :param prestretch:  Magnitude of of prestretch.
        :type prestretch: float
        :param cells:  List of cell identities that this will be applied to.
        :type cells: list

        """
        for cell_ref in cells:
            cell = self.cellDict[cell_ref]
            cell.prestrain_dict = dict.fromkeys(cell.prestrain_dict, prestretch)

    def apply_prestretch_to_cell_identity_pairs(self, prestretch, cell_pair, unipolar=False):
        """Add active contractility to cell_pairs that share adhesions.

        :param prestretch:   Magnitude of prestretch
        :type prestretch: float
        :param cell_pair:  A list of two cell identites that will localise contractility on shared junctions e.g. ``['A', 'B']```.
        :type cell_pair: list
        :param unipolar:  (Default value = False)  Apply on only the first cell.
        :type unipolar: bool

        """

        self.cellDict[cell_pair[0]].prestrain_dict[cell_pair[1]] = prestretch
        if not unipolar:
            self.cellDict[cell_pair[1]].prestrain_dict[cell_pair[0]] = prestretch

    def run_simulation_timestep(self, apply_to: list = None, viscous_cells: list = None, dt: float = None):
        """Run a full simulation timestep, including viscous length updates and solving to equilibrium.

        :param apply_to:  (Default value = 'all')  List of cells to apply function to.
        :type apply_to: list
        :param viscous_cells:  (Default value = 'all')  List of cells that will have length and area updates.
        :type viscous_cells: list
        :param dt:  (Default value = None)  length of time to step forward, if viscoelastic cortex
                    (active when cortical_timescale > 0)
        :type dt: float

        """
        apply_to = 'all' if apply_to is None else apply_to
        viscous_cells = 'all' if viscous_cells is None else viscous_cells

        # Update adhesions
        self.update_adhesion_points_between_all_cortices(apply_to=apply_to)
        # Update slow adhesions
        self.update_slow_adhesions(prune=True)
        # Viscous update
        self.update_all_rest_lengths_and_areas(apply_to=viscous_cells, dt=dt)
        # Relax
        if self.boundary_bc == 'periodic':
            self.relax_periodic_tissue()
        else:
            self.solve_bvps_in_parallel(applyTo=apply_to)

        # Increase tissue age
        self.age = self.age + dt if (hasattr(self, 'age') and self.age is not None) else dt
        # And age of adhesions
        for ad in self.slow_adhesions:
            ad.age += dt

    def solve_bvps_in_series(self, applyTo='all'):
        """Solve the bvp for each cell in series to reach elastic equilibrium. \todo haven't tested this functionality
        in a while

        :param applyTo:  (Default value = 'all')  Which cells to apply the function to.
        :type applyTo: list

        """

        self.verboseprint(" ".join(["Solving BVPs in series on", ", ".join(applyTo)]), object(), 1)

        # Randomise the order in which we apply the solving
        applyTo = [c.identifier for c in self.cells] if applyTo == "all" else applyTo
        shuffle(applyTo)

        # Update the adhesion points
        self.update_adhesion_points_between_all_cortices(build_trees=True)

        for cell_label in applyTo:
            cell = self.cellDict[cell_label]
            # cell.move_cortex_nodes_to_equilibrium_dist(scaling=1)

            cell.solve_bvp()

            # Get the neighbour cells
            naboer = cell.get_neighbours()
            naboer = [nabo for nabo in naboer if 'boundary' not in nabo]
            naboer.append(cell.identifier)

            # Update the adhesion points
            self.update_adhesion_points_between_all_cortices(build_trees=True, apply_to=naboer)

        self.total_elastic_relaxes += 1

    def solve_bvps_in_parallel(self, applyTo="all", smoothing=0, hand_of_god=True):
        """Use joblib to solve the bvp for all cortices and reach tissue elastic equilibrium

        :param applyTo:  (Default value = "all")  Which cells to apply the function to.
        :type applyTo: list
        :param smoothing:  (Default value = 0)  How much to smooth cortex variables between solving.
        :type smoothing: float
        :param hand_of_god:  (Default value = True)  Whether to manually move cortices to at least ``delta`` apart before solving.  This helps with stability.
        :type hand_of_god: bool
        :return success: Whether solving ran without errors and converged.
        :rtype: bool

        """

        self.verboseprint(" ".join(["Solving BVPs in parallel on cells", ", ".join(applyTo)]), object(), 1)

        # Which cells will relax
        applyTo = [c.identifier for c in self.cells] if applyTo == "all" else applyTo
        cells_to_relax = [self.cellDict[c] for c in applyTo]

        done = False  # This will only pass when the number of nodes has stayed constant.
        success = True  # Returns this bools as a check of whether we were successful within max_relaxes

        counter = 0
        while not done:

            # Get old positions before we do anything
            old_xs = np.array([c.x[i] for c in cells_to_relax for i in range(c.x.size)])
            old_ys = np.array([c.y[i] for c in cells_to_relax for i in range(c.y.size)])

            self.update_adhesion_points_between_all_cortices(only_fast=False, build_trees=True)
            if self.boundary_bc in ['elastic', 'viscous']:
                boundary_success = self.relax_deformable_boundary(update_adhesions=False)
                self.update_adhesion_points_between_all_cortices(only_fast=False, build_trees=True)
            else:
                boundary_success = True

            # Fix min distances to delta
            if hand_of_god:
                self.hand_of_god_distances_to_delta(initial_adhesion_update=False)

            # Run solver in parallel
            solve_data = Parallel(n_jobs=-1)(delayed(cell.solve_bvp)()
                                             for cell in cells_to_relax)
            sols_fail_check = any([r[0] != 0 for r in solve_data])
            self.last_solve_success = not sols_fail_check
            results = np.array([r[1] for r in solve_data])

            # Check if we are moving too far.
            # Calculate the distances that the cortex has moved.
            distances = np.linalg.norm(
                [np.concatenate(results[:, 3]).ravel() - old_xs, np.concatenate(results[:, 4]).ravel() - old_ys],
                axis=0)
            max_dist = np.max(distances)
            dist_norm = np.linalg.norm(distances) / len(self.cells)
            # If any part of the cortex moved too far, slow it down.
            cutoff = self.delta * self.relax_dist_threshold

            # If we moved too far AND failed then retry. If we failed but didn't move much then just let it pass
            if max_dist > cutoff and sols_fail_check:

                # Make sure we loop again
                done = False
                if self.verbose:
                    print('Moved too far (%s); decreasing step. sols_fail_check = %s (max %s). Num tries = %s'
                          % (max_dist, sols_fail_check, max([r[0] != 0 for r in solve_data]), counter))

                # Get the old variables
                old_vars = [[cell.s, cell.theta, cell.gamma, cell.x,
                             cell.y, cell.D, cell.C] for cell in self.cells]
                old_vars = np.array(old_vars)

                # update the results to not move as far.
                ratio_to_scale = cutoff / max_dist
                total_change = results - old_vars
                results = old_vars + ratio_to_scale * total_change

                # Update the solving tolerance
                sol_status_list = [r[0] for r in solve_data]
                print(f'Solving failed: {sol_status_list}')

                # Update the relaxation tolerance to a softer value for the solver
                for cell_index in range(len(sol_status_list)):
                    if sol_status_list[cell_index] != 0:
                        self.cells[cell_index].relax_tol *= 2

            # Or if we haven't reached elastic equilibrium (no more movement) relax again
            elif (
                    dist_norm > self.elastic_relax_tol or not boundary_success) and counter < self.max_elastic_relax_steps:
                done = False
                # if self.verbose:
                print('Not yet in equilibrium (d_norm = %s). Have relaxed %s times' % (dist_norm, counter))

            # Or, if passed all checks, we are done.
            else:
                done = True if not sols_fail_check else False
                self.in_equilibrium = 1
            # If we have relaxed too many times, break out
            if counter > self.max_elastic_relax_steps:
                done = True
                success = False
                self.in_equilibrium = 0
                if self.verbose:
                    print('Relaxed more than max_elastic_relax_steps; ending now.')

            # Update the cortex variables
            for cell_id in range(len(cells_to_relax)):
                cells_to_relax[cell_id].s = results[cell_id, 0]
                cells_to_relax[cell_id].theta = results[cell_id, 1]
                cells_to_relax[cell_id].gamma = results[cell_id, 2]
                cells_to_relax[cell_id].x = results[cell_id, 3]
                cells_to_relax[cell_id].y = results[cell_id, 4]
                cells_to_relax[cell_id].D = results[cell_id, 5]
                cells_to_relax[cell_id].C = results[cell_id, 6]
                cells_to_relax[cell_id].n = cells_to_relax[cell_id].x.size

                # Update spacing
                cells_to_relax[cell_id].update_deformed_mesh_spacing()

            if smoothing != 0:
                self.smooth_all_variables_with_spline(smoothing=smoothing)

            counter += 1

        self.last_num_internal_relaxes = counter
        self.total_elastic_relaxes += 1

        return success

    def get_unbalanced_force_residual(self, norm='L2'):
        """Calculate the unbalanced force at centre of every bicellular junction.
        Can take L1 or L2 norm.

        :param norm:  (Default value = 'L2')
        :type norm: string

        """

        raise NotImplementedError("Need to implement force calculation in unbalanced residual")

        assert norm in ['L1', 'L2', 'max'], "Error, norm must be L1 or L2 or max"

        # Get cell pairs sharing juncs
        junc_pairs = self.get_junction_cell_pairs()

        # Loop over each and get force
        residuals = []
        for cell_pair in junc_pairs:
            c1, c2 = self.cellDict[cell_pair[0]], self.cellDict[cell_pair[1]]
            # Get midpoint
            c1_idx, c2_idx = self.get_coordinates_of_junction_midpoint(cell_pair[0], cell_pair[1])

            # If not connected:
            if c1_idx is None or c2_idx is None:
                continue

            # Get normal force
            diffs = []
            for ixd_shift in range(-2, 3):
                shifted_idx1 = c1_idx + ixd_shift
                shifted_idx2 = c2_idx + ixd_shift
                shifted_idx1 = shifted_idx1 if shifted_idx1 < c1.C.size else shifted_idx1 - c1.C.size
                shifted_idx2 = shifted_idx2 if shifted_idx2 < c2.C.size else shifted_idx2 - c2.C.size

                raise NotImplementedError("Need to implement force calculation in unbalanced residual")

                # ctau_1 = c1.kappa * c1.C[shifted_idx1] ** 3 + \
                #          (1 - (c1.get_prestrains()[shifted_idx1] / c1.gamma[shifted_idx1])) * c1.C[shifted_idx1]
                # ctau_1 /= c1.kappa
                # # For other cell
                # ctau_2 = c2.kappa * c2.C[shifted_idx2] ** 3 + \
                #          (1 - (c2.get_prestrains()[shifted_idx2] / c2.gamma[shifted_idx2])) * c2.C[shifted_idx2]
                # ctau_2 /= c2.kappa
                #
                # diffs.append(ctau_1 - ctau_2)

            # Store mean
            residuals.append(np.mean(diffs))

        if norm == 'L1':
            total_residual = np.abs(residuals).mean()
        elif norm == 'L2':
            total_residual = np.power(residuals, 2).mean()
        elif norm == 'max':
            total_residual = np.max(np.abs(residuals))

        return total_residual

    def get_junction_cell_pairs(self):
        """Returns a set of the pairs of cells belonging to a junction, i.e. [(c1, c2),...], for all bicellular
        junctions in the tissue.

        :return junc_pairs:  A list of the bicellualr junctions in the tissue.
        :rtype: list
        """

        cell_ref_list = [c.identifier for c in self.cells]
        junc_pairs = set()
        for cell_ref in cell_ref_list:
            cell = self.cellDict[cell_ref]

            # Get cell neighbours
            cell_junc_pairs = [frozenset([cell_ref, nabo])
                               for nabo in
                               set([item for sublist in cell.adhesion_connections_identities for item in sublist])
                               if nabo in cell_ref_list]

            # Update the global list
            junc_pairs.update(cell_junc_pairs)

        # Store the pairs as lists
        junc_pairs = [list(p) for p in junc_pairs]

        return junc_pairs

    def get_xy_segment_lengths(self, x, y):
        """Get the spacing between nodes on a curve specified by (array(x), array(y))

        :param x:  The x-coords of the nodes
        :type x: np.array
        :param y:  The y-coords of the nodes
        :type y: np.array
        :return spacing:  The spacing.
        :rtype: np.array
        """
        spacing = [0.5 * (np.sqrt((x[i] - x[i - 1]) ** 2 + (y[i] - y[i - 1]) ** 2) +
                          np.sqrt((x[i + 1] - x[i]) ** 2 + (y[i + 1] - y[i]) ** 2))
                   for i in range(-1, x.size - 1)]
        spacing = np.roll(spacing, -1)

        return spacing

    def get_shortest_distance_between_two_cells(self, cell_1_ref, cell_2_ref):
        """Find the shortest distance between two cells. This is a proxy for the length of
        the shrinking junction when cell_1_ref = "C" and cell_2_ref = "D"

        :param cell_1_ref: Reference of first cell id.
        :type cell_1_ref: string
        :param cell_2_ref: Reference of first cell id.
        :type cell_2_ref: string
        :return min_dist:  The distance between the cortices.
        :rtype: float
        """

        cell_1 = self.cellDict[cell_1_ref]
        cell_2 = self.cellDict[cell_2_ref]

        # Get coords.
        cell_1_coords = np.dstack([cell_1.x, cell_1.y])[0]
        cell_2_coords = np.dstack([cell_2.x, cell_2.y])[0]

        # Get shortest distance
        min_dist = distance.cdist(cell_1_coords, cell_2_coords).min()

        return min_dist

    def get_coordinates_of_nearest_points_between_cells(self, cell_1_ref, cell_2_ref):
        """Get the local idxs for the nearest points between two cell cortices

        :param cell_1_ref:  Identity of one cell.
        :type cell_1_ref: string
        :param cell_2_ref:  Identity of another cell.
        :type cell_2_ref: string
        :return coords: Local indices of cells where the distance between them is minimised.
        :rtype: (int, int)
        """
        cell_1 = self.cellDict[cell_1_ref]
        cell_2 = self.cellDict[cell_2_ref]

        # Get coords.
        cell_1_coords = np.dstack([cell_1.x, cell_1.y])[0]
        cell_2_coords = np.dstack([cell_2.x, cell_2.y])[0]

        # Get shortest distance
        distances = distance.cdist(cell_1_coords, cell_2_coords)

        # Return (idx of Cell1, idx of Cell2)
        return np.unravel_index(distances.argmin(), distances.shape)

    def get_bicellular_junctions_for_cell(self, cell_ref, smooth=True, d_threshold=0.015):
        """Get the local indices (i,j) for the coordinates of the endpoints of bicellular junctions in a cell

        :param cell_ref: identity of cell.
        :type cell_ref: string
        :param smooth:  (Default value = True)  Whether to smooth the cortex curvature to find the straight segments.
        :type smooth: bool
        :param d_threshold:  (Default value = 0.015)  Threshold below which cortex is classed as stright.
        :type d_threshold: float
        :return bi_junctions:  (list([id1, id2],...))  A list of the paired local ids for where the bicellular junctions start and end.
        :rtype: string
        """

        cell = self.cellDict[cell_ref]
        filtered_d = cell.D
        if smooth:
            filtered_d = savgol_filter(filtered_d, 81, 3)
        # Find where curvature gradient is small
        idxs_to_plot = np.where(np.abs(filtered_d) < d_threshold)[0]
        # Remove small consective groups of indices, which may be at the vertex centre
        consec_groups = [list(group) for group in consecutive_groups(idxs_to_plot)]
        idxs_to_plot = [l for subgroup in consec_groups for l in subgroup if len(subgroup) > 25]

        # Find the vertices, then get the points closest to the vertex.
        most_ads_1 = np.array([max(ad_ids, key=Counter(ad_ids).get) if ad_ids else 0
                               for ad_ids in cell.adhesion_connections_identities])
        vertex_locs = np.where(most_ads_1[:-1] != most_ads_1[1:])[0]

        bicellular_ends = []
        for v_loc in vertex_locs:
            try:
                v1, v2 = idxs_to_plot[np.argmax(idxs_to_plot > v_loc)], idxs_to_plot[
                    np.argmax(idxs_to_plot > v_loc) - 1]
                if v1 not in bicellular_ends:
                    bicellular_ends.append(v1)
                if v2 not in bicellular_ends:
                    bicellular_ends.append(v2)
            except:
                pass

        bicellular_juncs = []
        for idx, j in enumerate(bicellular_ends):
            if idx % 2 == 0:
                bicellular_juncs.append((j, bicellular_ends[(idx + 3) % len(bicellular_ends)]))
        # bicellular_juncs = [(bicellular_ends[i], bicellular_ends[i+1]) for i in range(len(bicellular_ends))[::2]]

        return bicellular_juncs

    def check_if_cells_share_a_vertex(self, c1_ref, c2_ref, c3_ref):
        """True if each cell shares adehsions with both other cells.  Doesn't work for the tissue boundary

        :param c1_ref:  Identifier for first cell
        :type c1_ref: string
        :param c2_ref:  Identifier for second cell
        :type c2_ref: string
        :param c3_ref:  Identifier for third cell
        :type c3_ref: string
        :return vertex_check:  True if cells share a vertex
        :rtype: bool
        """

        if 'boundary' in [c1_ref, c2_ref, c3_ref]:
            return False

        return (c2_ref in list(itertools.chain.from_iterable(self.cellDict[c1_ref].adhesion_connections_identities)) and
                c3_ref in list(itertools.chain.from_iterable(self.cellDict[c1_ref].adhesion_connections_identities)) and
                c3_ref in list(itertools.chain.from_iterable(self.cellDict[c2_ref].adhesion_connections_identities)))

    def get_locations_of_cortex_adhesion_transition(self, cell_1_ref, vertex_neighbours):
        """Find where cell 1 (cell_1_ref) swaps from adhering to cell 2 to cell 3.
        vertex_neighbours = (cell_2_ref, cell_3_ref)

        :param cell_1_ref:  Identity of cell 1.
        :type cell_1_ref: string
        :param vertex_neighbours:  List (len=2) of neighbour identities.
        :type vertex_neighbours: list
        :return local_index:  Local idx in cell 1 where there is an adhesion transition.
        :rtype: int

        """

        cell_2_ref = vertex_neighbours[0]
        cell_3_ref = vertex_neighbours[1]

        cell_1 = self.cellDict[cell_1_ref]

        if not self.check_if_cells_share_a_vertex(cell_1_ref, cell_2_ref, cell_3_ref):
            print("Cells %s, %s, %s don't share a vertex" % (cell_1_ref, cell_2_ref, cell_3_ref))

            return 0

        # Find most adhesion connections for 1
        most_ads_1 = np.array([max(ad_ids, key=Counter(ad_ids).get) if ad_ids else 0
                               for ad_ids in cell_1.adhesion_connections_identities])
        changing_indices_1 = np.where(most_ads_1[:-1] != most_ads_1[1:])[0]
        changing_indices_23 = [idx for idx in changing_indices_1 if most_ads_1[idx] in [cell_2_ref, cell_3_ref]
                               and most_ads_1[idx + 1] in [cell_2_ref, cell_3_ref]]

        if len(changing_indices_23) != 1:
            print('error, vertex not well defined. Trying cortex with no adhesion')

            # Get the open bit
            changing_indices_none = [idx for idx in changing_indices_1 if most_ads_1[idx]
                                     in ['0', cell_2_ref, cell_3_ref] and most_ads_1[idx + 1]
                                     in ['0', cell_2_ref, cell_3_ref]]

            if len(changing_indices_none) != 2:
                print('error, failed no adhesion cortex')
                return 0

            midpoint = int(np.mean(changing_indices_none))

            return midpoint

        return changing_indices_23[0]

    def get_all_vertices_for_cell(self, cell_ref):
        """  Get a list of identities of neighbours that a cell shares adhesions with.

        :param cell_ref:  Identity of cell.
        :type cell_ref: string
        :return vertices:  List of neighbour identities.
        :rtype: list

        """

        cell = self.cellDict[cell_ref]
        # Get neighbours
        cell_neighbours = set(itertools.chain(*cell.adhesion_connections_identities))
        # Possible vertices
        possible_verts = [[cell_ref, pair[0], pair[1]] for pair in itertools.combinations(cell_neighbours, 2)]
        # Get real ones
        cell_vertices = [v for v in possible_verts if self.check_if_cells_share_a_vertex(*v)]

        return cell_vertices

    def get_all_tissue_vertices(self, apply_to='all'):
        """Get a set of all unique vertices (as lists of 3 cell identities) in the tissue

        :param apply_to:  (Default value = 'all')  If not ``'all'``, a list of cell identifiers that the function will be applied to.
        :type apply_to: string
        :return vertices:  A list of triplets of cell identities.
        :rtype: list

        """

        apply_to = [c.identifier for c in self.cells] if apply_to == "all" else apply_to

        vertices = set([])
        for ref in apply_to:
            cell_verts = [frozenset(v) for v in self.get_all_vertices_for_cell(ref)]
            vertices.update(cell_verts)

        return [list(vert) for vert in vertices]

    def get_cortex_coord_of_vertex_triangle(self, cell_1_ref, cell_2_ref, cell_3_ref):
        """Get local index for the coords of the vertex triangle on three cells.

        :param cell_1_ref:   Idientifier of first cell.
        :type cell_1_ref: string
        :param cell_2_ref:   Idientifier of second cell.
        :type cell_2_ref: string
        :param cell_3_ref:   Idientifier of third cell.
        :type cell_3_ref: string
        :return vertex:  The three local indices on each cortex, where the vertex is located on each cortex.
        :rtype: tuple

        """
        coord1 = self.get_locations_of_cortex_adhesion_transition(cell_1_ref,
                                                                  vertex_neighbours=(cell_2_ref, cell_3_ref))
        coord2 = self.get_locations_of_cortex_adhesion_transition(cell_2_ref,
                                                                  vertex_neighbours=(cell_1_ref, cell_3_ref))
        coord3 = self.get_locations_of_cortex_adhesion_transition(cell_3_ref,
                                                                  vertex_neighbours=(cell_1_ref, cell_2_ref))

        return coord1, coord2, coord3

    def get_cartesian_coords_of_vertex_triangle(self, cell_1_ref, cell_2_ref, cell_3_ref):
        """Get cartesian coords of the vertex traingle from the local idxs

        :param cell_1_ref:  Idientifier of first cell.
        :type cell_1_ref: string
        :param cell_2_ref:  Idientifier of second cell.
        :type cell_2_ref: string
        :param cell_3_ref:  Idientifier of third cell.
        :type cell_3_ref: string
        :return vertex:  The ((x1,y1), (x2,y2), (x3,y3)) cartesian coords of the triangle.
        :rtype: tuple

        """
        coord1, coord2, coord3 = self.get_cortex_coord_of_vertex_triangle(cell_1_ref, cell_2_ref, cell_3_ref)

        c1 = self.cellDict[cell_1_ref]
        c2 = self.cellDict[cell_2_ref]
        c3 = self.cellDict[cell_3_ref]

        return (c1.x[coord1], c1.y[coord1]), (c2.x[coord2], c2.y[coord2]), (c3.x[coord3], c3.y[coord3])

    def get_vertex_circumcircle(self, cell_1_ref, cell_2_ref, cell_3_ref):
        """Get centroid and radius of circumcircle\insribed at vertices

        :param cell_1_ref:  Idientifier of first cell.
        :type cell_1_ref: string
        :param cell_2_ref:  Idientifier of second cell.
        :type cell_3_ref: string
        :param cell_3_ref:  Idientifier of third cell.
        :type cell_3_ref: string
        :return circle_props:  The (x,y) centre and radius of the circumcircle.
        :rtype: tuple

        """

        T = self.get_cartesian_coords_of_vertex_triangle(cell_1_ref, cell_2_ref, cell_3_ref)
        (x1, y1), (x2, y2), (x3, y3) = T
        A = np.array([[x3 - x1, y3 - y1], [x3 - x2, y3 - y2]])
        Y = np.array([(x3 ** 2 + y3 ** 2 - x1 ** 2 - y1 ** 2), (x3 ** 2 + y3 ** 2 - x2 ** 2 - y2 ** 2)])
        if np.linalg.det(A) == 0:
            return False
        Ainv = np.linalg.inv(A)
        X = 0.5 * np.dot(Ainv, Y)
        x, y = X[0], X[1]
        r = np.sqrt((x - x1) ** 2 + (y - y1) ** 2)
        return (x, y), r

    def get_length_of_shared_junction(self, cell_1_ref, cell_2_ref, alternative_junc_refs=None):
        """Find the length of the bijunction shared between two cells

        :param cell_1_ref:  Idientifier of first cell.
        :type cell_1_ref: string
        :param cell_2_ref:  Idientifier of second cell.
        :type cell_2_ref: string
        :param alternative_junc_refs:  (Default value = None)  Possible alternative junctions to try. Passed as [['ref_1', 'ref_2'],...]
        :type alternative_junc_refs: list
        :return length:  The length of cortex where the two cells share adhesions.  Returns 0 if they don't share any.
        :rtype: float

        """
        cell_1 = self.cellDict[cell_1_ref]
        cell_2 = self.cellDict[cell_2_ref]

        if not (cell_1_ref in list(itertools.chain.from_iterable(cell_2.adhesion_connections_identities)) or
                cell_2_ref in list(itertools.chain.from_iterable(cell_1.adhesion_connections_identities))):
            print("Cells %s and %s don't share a junction" % (cell_1_ref, cell_2_ref))
            if not (alternative_junc_refs is None):
                print("Now trying alternatives %s and %s" % (alternative_junc_refs[0], alternative_junc_refs[1]))
                length = self.get_length_of_shared_junction(cell_1_ref=alternative_junc_refs[0],
                                                            cell_2_ref=alternative_junc_refs[1])
                return length
            else:
                return 0

        # Boolean where connected
        shared_junc_bool_1 = [cell_2_ref == max(ad_ids, key=Counter(ad_ids).get) if ad_ids else 0
                              for ad_ids in cell_1.adhesion_connections_identities]

        # Rare case where there is a connection only on one side.
        if any(shared_junc_bool_1) is False:
            print("Error, cells %s and %s don't share a junction" % (cell_1_ref, cell_2_ref))
            return 0

        # Get the segment lengths of cortex 1
        seg_lengths = cell_1.get_xy_segment_lengths()

        # Get the total distance of the segment lengths
        total_length = seg_lengths[shared_junc_bool_1].sum()

        return total_length

    def get_coordinates_of_junction_midpoint(self, cell_1_ref, cell_2_ref):
        """Find the local idx midpoint of the junction shared between two cells

        :param cell_1_ref:  Idientifier of first cell.
        :type cell_1_ref: string
        :param cell_2_ref:  Idientifier of second cell.
        :type cell_2_ref: string
        :return midpoint:  the local indices in each cell (idx_c1, idx_c2) of the junction midpoint
        :rtype: (int, int)

        """
        cell_1 = self.cellDict[cell_1_ref]
        cell_2 = self.cellDict[cell_2_ref]

        assert (cell_1_ref in list(itertools.chain.from_iterable(cell_2.adhesion_connections_identities)) or
                cell_2_ref in list(itertools.chain.from_iterable(cell_1.adhesion_connections_identities))), \
            "Error, cells %s and %s don't share a junction" % (cell_1_ref, cell_2_ref)

        # Boolean where connected
        shared_junc_bool_1 = [1 if cell_2_ref in ad_list else 0 for ad_list in cell_1.adhesion_connections_identities]
        # Rare case where there is a connection only on one side.
        if any(shared_junc_bool_1) is False:
            return None, None

        # If the indices start from the end of the list and run over, we need to roll it
        idx_to_roll = max([i for i, x in enumerate(shared_junc_bool_1) if not x])
        if idx_to_roll != len(shared_junc_bool_1) - 1:
            shared_junc_bool_1 = deque(shared_junc_bool_1)
            shared_junc_bool_1.rotate(len(shared_junc_bool_1) - idx_to_roll)

        # get midpoint
        mid_point_1 = int(np.median(np.where(shared_junc_bool_1)[0]))
        # Unroll it if we had to.
        if idx_to_roll != len(shared_junc_bool_1) - 1:
            mid_point_1 -= len(shared_junc_bool_1) - idx_to_roll
        mid_point_1_coords = np.array([[cell_1.x[mid_point_1], cell_1.y[mid_point_1]]])

        # Point closest on other cortex
        cell_2_coords = np.dstack([cell_2.x, cell_2.y])[0]
        distances = distance.cdist(mid_point_1_coords, cell_2_coords)[0]
        mid_point_2 = np.argmin(distances)

        return mid_point_1, mid_point_2

    def get_cell_sum_stress_tensor(self):
        """The sum of all cell stress tensors.

        :return stress:  The 2x2 tissue stress tensor.
        :rtype: np.array
        """

        stress = np.array([[0., 0.], [0., 0.]])
        for c in self.cells:
            stress += c.get_stress_tensor()

        return stress

    def get_tissue_pressure(self):
        """  Isotropic part of tissue-level stress

        :return stress:  Isotropic componenet of stress tensor.
        :rtype: float

        """
        return -0.5 * self.get_boundary_stress_tensor().trace()

    def get_stress_tensor_around_cell_cluster_depreciated(self):
        """Calculate stress tensor of all the internal cells."""

        cell_coords = []
        boundary_spacings = []
        boundary_coords = []
        for cell in self.cells:

            temp_ad_coords, temps_spacings, temps_boundary_coords = [], [], []
            for i in range(len(cell.adhesion_connections_identities)):
                if len(cell.adhesion_connections_identities[i]) > 0:
                    temp_ad_coords.append([cell.x[i], cell.y[i]])
                    temps_spacings.append(cell.adhesion_connections_spacings[i][0])
                    temps_boundary_coords.append(cell.adhesion_connections[i][0])

            cell_coords.extend(temp_ad_coords)
            boundary_spacings.extend(temps_spacings)
            boundary_coords.extend(temps_boundary_coords)

        cell_coords = np.array(cell_coords)
        boundary_spacings = np.array(boundary_spacings)
        boundary_coords = np.array(boundary_coords)

        # Get the directions
        dirs = cell_coords - boundary_coords
        # normalise directions
        dirs = np.array(dirs)
        dirs = np.divide(dirs, np.sqrt(inner1d(dirs, dirs))[:, np.newaxis])

        # Get forces
        distances = np.linalg.norm(cell_coords - boundary_coords, axis=1)
        e = distances - self.delta
        # Calc spring force
        force = self.cells[0].omega0 * e

        # Scale the force according to the spacing of adhesion nodes
        force *= boundary_spacings

        vector_of_forces = dirs * force[:, np.newaxis]

        # Centroid
        centroid = np.array([np.mean(self.boundary_adhesions[0]), np.mean(self.boundary_adhesions[1])])

        # Get the stress tensor
        stress = np.array([[0., 0.], [0., 0.]])
        for i in range(force.size):
            r_vector = boundary_coords[i] - centroid
            stress += np.outer(r_vector, vector_of_forces[i])

        area = geom.Polygon(np.dstack((self.boundary_adhesions))[0]).area
        stress /= area

        return stress

    def get_boundary_stress_tensor(self):
        """Calculate stress tensor of the boundary stencil \todo needs to be upgraded with new adhesion class
        make sure factoring for spacing on both sides.

        :return stress:  The boundary stress tensor.
        :rtype: np.array

        """

        if self.slow_adhesions_active:

            # Centroid
            centroid = np.array([np.mean(self.boundary_adhesions[0]), np.mean(self.boundary_adhesions[1])])

            # Get the stress tensor
            stress = np.array([[0., 0.], [0., 0.]])

            self.cellDict['boundary'].update_deformed_mesh_spacing()
            for ad in self.slow_adhesions:
                if 'boundary' in [ad.cell_1.identifier, ad.cell_2.identifier]:
                    r_vector = np.array(ad.get_xy_at_this_end(this_cell_id='boundary')) - centroid
                    force = ad.get_vector_force(from_cell_id='boundary')
                    stress += np.outer(r_vector, force)

            area = geom.Polygon(np.dstack((self.boundary_adhesions))[0]).area
            stress /= area

        else:
            cell_coords = []
            boundary_spacings = []
            boundary_coords = []
            for cell in self.cells:

                temp_ad_coords, temps_spacings, temps_boundary_coords = [], [], []
                for i in range(len(cell.adhesion_connections_identities)):
                    if len(cell.adhesion_connections_identities[i]) > 0:
                        if cell.adhesion_connections_identities[i][0] == 'boundary':
                            temp_ad_coords.append([cell.x[i], cell.y[i]])
                            temps_spacings.append(cell.adhesion_connections_spacings[i][0])
                            temps_boundary_coords.append(cell.adhesion_connections[i][0])

                cell_coords.extend(temp_ad_coords)
                boundary_spacings.extend(temps_spacings)
                boundary_coords.extend(temps_boundary_coords)

            cell_coords = np.array(cell_coords)
            boundary_spacings = np.array(boundary_spacings)
            boundary_coords = np.array(boundary_coords)

            # Get the directions
            dirs = cell_coords - boundary_coords
            # normalise directions
            dirs = np.array(dirs)
            dirs = np.divide(dirs, np.sqrt(inner1d(dirs, dirs))[:, np.newaxis])

            # Get forces
            distances = np.linalg.norm(cell_coords - boundary_coords, axis=1)
            e = distances - self.delta
            # Calc spring force
            force = self.cells[0].omega0 * e

            # Scale the force according to the spacing of adhesion nodes
            force *= boundary_spacings

            vector_of_forces = dirs * force[:, np.newaxis]

            # Centroid
            centroid = np.array([np.mean(self.boundary_adhesions[0]), np.mean(self.boundary_adhesions[1])])

            # Get the stress tensor
            stress = np.array([[0., 0.], [0., 0.]])
            for i in range(force.size):
                r_vector = boundary_coords[i] - centroid
                stress += np.outer(r_vector, vector_of_forces[i])

            area = geom.Polygon(np.dstack((self.boundary_adhesions))[0]).area
            stress /= area
        # stress /= np.sum(self.get_xy_segment_lengths(self.boundary_adhesions[0], self.boundary_adhesions[1]))

        # eigvals, eigvecs = np.linalg.eig(stress)
        # evec1, evec2 = eigvecs[:, 0], eigvecs[:, 1]
        #
        # fig, ax = plt.subplots(figsize=(11, 9))
        # # self.plot_self(ax=ax, axEqual=True, plot_stress=0, plot_adhesion_forces=1, plotAdhesion=1,
        # #                             plot_boundary=1,
        # #                             lagrangian_tracking=0, plot_tension=1)
        # ax.quiver(boundary_coords[:, 0], boundary_coords[:, 1], vector_of_forces[:, 0], vector_of_forces[:, 1],
        #           width=0.001, scale=.0010,  # scale=.00010,
        #           color='r', zorder=10)
        #
        # ax.quiver(*centroid, *evec1, color=['r'], scale=10)
        # ax.quiver(*centroid, *evec2, color=['b'], scale=10)
        #
        # stress_in_AP = np.dot(stress, [1, 0])
        # ax.quiver(*centroid, *stress_in_AP, color=['y'], scale=.0001)
        # stress_in_DV = np.dot(stress, [0, 1])
        # ax.quiver(*centroid, *stress_in_DV, color=['y'], scale=.0001)
        #
        # plt.show()
        #
        # print(Tre)

        return stress

    def deform_elastic_boundary(self, stiffness_x=None, stiffness_y=None, update_ads=False):
        """Relax tissue with an elastic BC for the boundary

        :param stiffness_x:  (Default value = None)  Stiffness along x-axis.
        :type stiffness_x: float
        :param stiffness_y:  (Default value = None)  Stiffness along y-axis.
        :type stiffness_y: float
        :param update_ads:  (Default value = False)  Whether to perform an update of the adhesions.
        :type update_ads: bool

        """

        if len(self.reference_boundary_adhesions) == 0:
            self.reference_boundary_adhesions = self.boundary_adhesions + 0.

        stiffness_x = stiffness_x if stiffness_x is not None else self.boundary_stiffness_x
        stiffness_y = stiffness_y if stiffness_y is not None else self.boundary_stiffness_y

        if update_ads:
            self.update_adhesion_points_between_all_cortices()

        # Get stress
        stress = self.get_boundary_stress_tensor()

        # Move boundary
        stretch_x = stress[0, 0] / stiffness_x
        stretch_y = stress[1, 1] / stiffness_y

        # Strain
        x_strain = 1 + stretch_x
        y_strain = 1 + stretch_y

        # Shift boundary
        self.boundary_adhesions[0] = self.reference_boundary_adhesions[0] * x_strain
        self.boundary_adhesions[1] = self.reference_boundary_adhesions[1] * y_strain

        for cell in self.cells:
            for i in range(cell.x.size):
                if cell.adhesion_connections_identities[i][0] == 'boundary':
                    cell.x[i] *= x_strain
                    cell.y[i] *= y_strain

        self.update_adhesion_points_between_all_cortices(build_trees=True)

        return

    def get_tissue_width_and_height(self):
        """Top and bottom (delta x,delta y) of max/min of boundary stencil

        :return width_height:  The (width, height) of the boundary stencil.
        :rtype: tuple

        """
        return (np.max(self.boundary_adhesions[0]) - np.min(self.boundary_adhesions[0]),
                np.max(self.boundary_adhesions[1]) - np.min(self.boundary_adhesions[1]))

    def relax_deformable_boundary(self, stiffness_x=None, stiffness_y=None, update_adhesions=True, pure_shear=False,
                                  max_shift=0.025, max_shift_tol=1e-4):
        """Calculate the stress at the boundary and deform it elastically, updating cell psoitions as it moves to equilibrium.

        :param stiffness_x:  (Default value = None)  Stiffness of boundary in x-direction.
        :type stiffness_x: float
        :param stiffness_y:  (Default value = None)  Stiffness of boundary in y-direction.
        :type stiffness_y: float
        :param update_adhesions:  (Default value = True)  Perform adhesion update?
        :type update_adhesions: bool
        :param pure_shear:  (Default value = False)  Enforce constant area?
        :type pure_shear: bool
        :param max_shift:  (Default value = 0.025)  Maximum boundary strain (can help prevent the boundary overlapping cells).
        :type max_shift: float
        :param max_shift_tol:  (Default value = 1e-4)  We don't get closer than this to a cell.
        :type max_shift_tol: float
        :return success:  Whether relaxation passed or not.
        :rtype: bool

        """

        stiffness_x = stiffness_x if stiffness_x is not None else self.boundary_stiffness_x
        stiffness_y = stiffness_y if stiffness_y is not None else self.boundary_stiffness_y

        if len(self.reference_boundary_adhesions) == 0:
            self.reference_boundary_adhesions = [self.boundary_adhesions[0] + 0,
                                                 self.boundary_adhesions[1] + 0]

        if update_adhesions:
            self.update_adhesion_points_between_all_cortices(only_fast=False, build_trees=True)

        # Get stress
        stress = self.get_boundary_stress_tensor()

        # Move boundary
        strain_x = stress[0, 0] / stiffness_x
        strain_y = stress[1, 1] / stiffness_y
        success = True

        ####

        if pure_shear:
            biggest_stretch = max([strain_x, strain_y])
            strain_x = biggest_stretch
            strain_y = biggest_stretch

        # Strain
        x_stretch = 1 + strain_x
        y_stretch = 1 + strain_y

        # boundary centroid
        c_x, c_y = np.mean(self.reference_boundary_adhesions[0]), np.mean(self.reference_boundary_adhesions[1])
        # min max of current and reference box
        upper_x, upper_x_ref = np.max(self.boundary_adhesions[0]), np.max(self.reference_boundary_adhesions[0])
        lower_x, lower_x_ref = np.min(self.boundary_adhesions[0]), np.min(self.reference_boundary_adhesions[0])
        upper_y, upper_y_ref = np.max(self.boundary_adhesions[1]), np.max(self.reference_boundary_adhesions[1])
        lower_y, lower_y_ref = np.min(self.boundary_adhesions[1]), np.min(self.reference_boundary_adhesions[1])
        ref_x, ref_y = np.array([lower_x_ref, upper_x_ref]), np.array([lower_y_ref, upper_y_ref])
        current_x, current_y = np.array([lower_x, upper_x]), np.array([lower_y, upper_y])

        # Predicted positions
        future_x, future_y = (ref_x - c_x) * x_stretch + c_x, (ref_y - c_y) * y_stretch + c_y

        # Make sure we don't get too close to a cell
        good_movement = all(abs(current_x - future_x) < max_shift)
        while not good_movement:
            x_stretch += 1e-4 * np.sign(1 - x_stretch)
            # Predicted positions
            future_x = (ref_x - c_x) * x_stretch + c_x
            good_movement = all(abs(current_x - future_x) < max_shift) or abs(1 - x_stretch) < max_shift_tol
            success = False

        good_movement = all(abs(current_y - future_y) < max_shift)
        while not good_movement:
            y_stretch += 1e-4 * np.sign(1 - y_stretch)
            # Predicted positions
            future_y = (ref_y - c_y) * y_stretch + c_y
            good_movement = all(abs(current_y - future_y) < max_shift) or abs(1 - y_stretch) < max_shift_tol
            success = False

        if self.boundary_bc == 'viscous':

            self.boundary_adhesions[0] = (self.boundary_adhesions[0] - c_x) * x_stretch + c_x
            self.boundary_adhesions[1] = (self.boundary_adhesions[1] - c_y) * y_stretch + c_y

            # for cell in self.cells:
            #     c_x, c_y = np.mean(cell.x), np.mean(cell.y)
            #     cell.x = (cell.x - c_x) * x_stretch + c_x
            #     cell.y = (cell.y - c_y) * y_stretch + c_y

        elif self.boundary_bc == 'elastic':

            # Shift boundary
            self.boundary_adhesions[0] = (self.reference_boundary_adhesions[0] - c_x) * x_stretch + c_x
            self.boundary_adhesions[1] = (self.reference_boundary_adhesions[1] - c_y) * y_stretch + c_y

            # Shift cells
            # See how much boundary will move
            # new_dx = np.max(self.boundary_adhesions[0]) - np.min(self.boundary_adhesions[0])
            # new_dy = np.max(self.boundary_adhesions[1]) - np.min(self.boundary_adhesions[1])
            # b_x_stretch = new_dx / old_dx
            # b_y_stretch = new_dy / old_dy
            # for cell in self.cells:
            #     c_x_cell, c_y_cell = np.mean(cell.x), np.mean(cell.y)
            #     cell.x = (cell.x - c_x_cell) * b_x_stretch + c_x_cell
            #     cell.y = (cell.y - c_y_cell) * b_y_stretch + c_y_cell

        self.boundary_cell.x = self.boundary_adhesions[0]
        self.boundary_cell.y = self.boundary_adhesions[1]

        return success

    def relax_eptm_with_boundary(self, stiffness_x=None, stiffness_y=None, max_solves=1):
        """Calculate the stress at the boundary and relax the boundary. N.B. stress and strain names
         are wrong way around below.

        :param stiffness_x:  (Default value = None)  Stiffness of boundary in x-direction.
        :type stiffness_x: float
        :param stiffness_y:  (Default value = None)  Stiffness of boundary in y-direction.
        :type stiffness_y: float
        :param max_solves:  (Default value = 1)  Maximum jiggles to try to get to equilibrium before giving up.
        :type max_solves: in

        """

        stiffness_x = stiffness_x if stiffness_x is not None else self.boundary_stiffness_x
        stiffness_y = stiffness_y if stiffness_y is not None else self.boundary_stiffness_y

        self.update_adhesion_points_between_all_cortices(only_fast=False, build_trees=True)
        for solve_num in range(max_solves):
            # Get stress and move boundary
            stress = self.get_boundary_stress_tensor()
            self.relax_deformable_boundary()

            # Relax
            self.solve_bvps_in_parallel()
            self.total_elastic_relaxes -= 1

        self.total_elastic_relaxes += 1

        return

    def relax_eptm_with_germband_bcs(self, posterior_pull_shift=None, stiffness_y=None, max_solves=5):
        """Relax eptm with BCs from GBE. Viscous D-V, fixed anterior and posterior pull.
        NB this is very specific to the fixed boundary with 14 cells.

        :param posterior_pull_shift:  (Default value = None)  Strain to apply to the right boundary nodes.
        :type posterior_pull_shift: float
        :param stiffness_y:  (Default value = None)  Tissue stiffness in y.
        :type stiffness_y: float
        :param max_solves:  (Default value = 5)  Maximum jiggles to try before giving up.
        :type max_solves: int

        """

        assert len(self.cells) == 14, "Must be 14 cells with fixed boundary."

        posterior_pull_shift = posterior_pull_shift if posterior_pull_shift is not None else self.posterior_pull_shift

        # Get boundary adhesions, find RHS and pull
        x, y = self.boundary_adhesions
        # Get reference
        centroid = np.mean([x, y], axis=1)
        # Angles to each point relative to centroid
        angles_from_centroid = np.arctan2(y - centroid[1], x - centroid[0])
        # Local tangent angles
        local_angles = np.arctan2(np.diff(y), np.diff(x))
        # Find the corners as points where the tangents change angle
        corners = np.array(
            [i for i in range(local_angles.size) if abs(local_angles[i] - local_angles[i - 1]) > np.pi / 50])
        # angle to corners
        centroid_corner_angles = np.arctan2(y[corners] - centroid[1], x[corners] - centroid[0])
        centroid_corner_angles[centroid_corner_angles < 0] += 2 * np.pi
        # order by angle
        idxs = np.argsort(centroid_corner_angles)
        centroid_corner_angles = centroid_corner_angles[idxs]
        # Find the RHS is defined as points between 2 corners
        min_angle, max_angle = centroid_corner_angles[-8] - 2 * np.pi, centroid_corner_angles[7]
        indices = np.where((min_angle <= angles_from_centroid) & (angles_from_centroid <= max_angle))

        # Shift rhs based on these values
        self.boundary_adhesions[0][indices] += posterior_pull_shift

        # Now relax internal cells, but fix the A-P sides
        self.relax_eptm_with_boundary(stiffness_x=np.inf, stiffness_y=stiffness_y, max_solves=max_solves)

        return

    def update_adhesion_distances_identifiers_and_indices_for_all_cortices(self):
        """Calculate the distances to all adhesions and the identies of the cortices.

        """
        for cell in self.cells:
            cell.update_adhesion_distances_identifiers_and_indices()

    def update_adhesions_for_hexagons(self):
        """For only 1 cell.  The cell is placed in a hexagonal wall and finds adhesions to the boundary.

        """

        self.verboseprint("Updating adhesions for hexagons", object(), 1)

        if len(self.cells[0].adhesion_point_coords) == 0:

            rNew = self.radius
            numPoints = self.n

            n = np.array([0, 1, 2, 3, 4, 5])

            x = rNew * np.sin(2 * np.pi * n / 6)
            y = rNew * np.cos(2 * np.pi * n / 6)

            pointsX = [np.linspace(x[i - 1], x[i], int(numPoints / 6), endpoint=False) for i in range(0, x.size)]
            pointsX = np.array(pointsX).flatten()
            pointsY = [np.linspace(y[i - 1], y[i], int(numPoints / 6), endpoint=False) for i in range(0, y.size)]
            pointsY = np.array(pointsY).flatten()

            points_all = np.dstack((pointsX, pointsY))[0]

            # Get spacing
            spacing = self.get_xy_segment_lengths(pointsX, pointsY)

            # Clear current adhesions
            self.cells[0].clear_adhesion_points()

            # Add new adhesions
            self.cells[0].update_adhesion_points(points_all, ['boundary'] * pointsX.size, spacing)

            self.cells[0].update_adhesion_distances_identifiers_and_indices(build_tree=True)

            if 'boundary' not in self.cellDict.keys():
                # if not hasattr(self, 'boundary_adhesions'):
                # Save
                self.boundary_adhesions = [pointsX, pointsY]
                # Make boundary cell
                initial_guessesA = {'D': [], 'C': [], 'gamma': [], 'theta': [],
                                    'x': self.boundary_adhesions[0], 'y': self.boundary_adhesions[1]}
                self.boundary_cell = Cell(initial_guesses=initial_guessesA, identifier='boundary')
                self.boundary_cell.prune_adhesion_data()
                self.boundary_adhesions[0] = self.boundary_cell.x
                self.boundary_adhesions[1] = self.boundary_cell.y
                self.cellDict['boundary'] = self.boundary_cell
                # self.boundary_cell.s_index_dict = {s: idx for (idx, s) in enumerate(self.boundary_cell.s)}
                self.cellDict['boundary'].update_deformed_mesh_spacing()

    def prune_slow_adhesions_by_length(self, max_length):
        """Remove slow adhesions if they are longer than 'max_length'

        :param max_length:  Adhesions with length above this will be discarded.
        :type max_length: float

        """

        self.verboseprint(f"Pruning slow adhesions longer than {max_length}", object(), 1)

        self.slow_adhesions = [ad for ad in self.slow_adhesions if ad.get_length() < max_length]

    def update_adhesion_stiffness_for_cells(self, new_omega: float, apply_to: list = None):
        """
        Sets adhesion stiffness for the list of given cell id's

        :param new_omega: The new stiffness.
        :type apply_to: float
        :param apply_to: (Default value = 'all')  If given, a list of cell identifiers that the function will be applied to.
        :type apply_to: list
        :return: None
        """
        apply_to = [c.identifier for c in self.cells] if apply_to is None else apply_to
        for c_id in apply_to:
            self.cellDict[c_id].omega0 = new_omega

    def update_slow_adhesions(self, prune=False, apply_to='all'):
        """Update the slow adhesions in the tissue and then store in every cell
        To save memory, slow adhesions are stored in Cell class as:
        slow_ad = (local_cell_index, other_cell_x, other_cell_y, other_cell_mesh_spacing)
        rather than as adhesion objects.

        :param prune:  (Default value = False)  Remove adhesions based on lifetime.
        :type prune: bool
        :param apply_to:  (Default value = 'all')  If not ``'all'``, a list of cell identifiers that the function will be applied to.
        :type apply_to: list

        """
        self.verboseprint("Updating slow adhesions", object(), 1)

        if self.slow_adhesions_active:

            apply_to = [c.identifier for c in self.cells] if apply_to == "all" else apply_to

            # keep a record of the nodes with a surviving adhesion after filtering
            connected_nodes = set([])

            # # Filter the existing connections and get whats left
            # if len(self.slow_adhesions) > 0 and prune:
            #     # Make a probabitlty distribution based on distances
            #     adhesion_distances = [1 / ad.get_length() for ad in self.slow_adhesions]
            #     sum_of_distance = sum(adhesion_distances)
            #     probabilities = [d/sum_of_distance for d in adhesion_distances]
            #
            #     # take a sample of the adhesions based on probabilities from distance
            #     number_of_ads_to_pick = int((1 / self.slow_adhesion_lifespan) * len(self.slow_adhesions))
            #     self.slow_adhesions = list(np.random.choice(self.slow_adhesions, number_of_ads_to_pick,
            #                                       p=probabilities, replace=False))

            # Remove slow adhesions randomly, based on lifetime in exponential distribution
            surviving_ads = []
            for ad in self.slow_adhesions:
                if prune:
                    if self.adhesion_shear_unbinding_factor > 0:
                        max_cortex_angle = np.max(ad.get_angle_relative_to_cortices()) / (0.5 * np.pi)
                    else:
                        max_cortex_angle = np.pi
                    # beta = self.slow_adhesion_lifespan * max_cortex_angle
                    # keep_adhesion = self.slow_adhesion_lifespan >= np.random.exponential(beta)
                    keep_adhesion = ad.age <= ad.lifespan * (max_cortex_angle ** self.adhesion_shear_unbinding_factor)
                    keep_adhesion = False if ad.get_length() > self.cells[0].max_adhesion_length else keep_adhesion
                else:
                    keep_adhesion = True

                if keep_adhesion:
                    connected_nodes.update([(ad.cell_1.identifier, ad.cell_1_s)])
                    connected_nodes.update([(ad.cell_2.identifier, ad.cell_2_s)])

                    surviving_ads.append(ad)

            self.slow_adhesions = surviving_ads

            # Build a tree between all nodes and get nearest neighbours, then connect nodes without a slow ad.
            all_points = []
            all_identifiers = []
            local_index = []
            # for cell in self.cells:
            for cell in self.cellDict.values():
                if 'boundary' not in cell.identifier:
                    all_points.extend(np.dstack((cell.x, cell.y))[0])
                    all_identifiers.extend([cell.identifier] * cell.x.size)
                    # local_index.extend([i for i in range(cell.x.size)])
                    local_index.extend([i for i in cell.s])

            # Num nodes from cells
            num_cell_nodes = len(all_identifiers)
            # Add the wall
            # all_points.extend(np.dstack((self.boundary_adhesions[0], self.boundary_adhesions[1]))[0])
            # all_identifiers.extend(['boundary'] * len(self.boundary_adhesions[0]))
            all_points.extend(np.dstack((self.boundary_cell.x, self.boundary_cell.y))[0])
            all_identifiers.extend([self.boundary_cell.identifier] * self.boundary_cell.x.size)
            local_index.extend([i for i in self.boundary_cell.s])

            tissue_tree = NearestNeighbors(radius=self.cells[0].max_adhesion_length,
                                           algorithm='auto', n_jobs=-1).fit(all_points)
            dists, indices = tissue_tree.radius_neighbors(all_points[:num_cell_nodes], sort_results=True)

            # Filter out ones with same identity
            adhesion_pairs = set([])
            for i in range(len(indices)):
                naboer = [nabo_id for nabo_id in indices[i] if all_identifiers[nabo_id] != all_identifiers[i]]
                if len(naboer) > 0:
                    adhesion_pairs.add(frozenset([i, naboer[0]]))
            # adhesion_pairs = {frozenset([i, [nabo_id for nabo_id in indices[i]
            #                                  if all_identifiers[nabo_id] != all_identifiers[i]][0]])
            #                                  for i in range(len(indices))}
            adhesion_pairs = [list(p) for p in adhesion_pairs]

            # Store all data
            # Append the adhesions if connections don't already exist
            active_cells = {c.identifier for c in self.cells}
            for coord in adhesion_pairs:
                if (all_identifiers[coord[0]] in active_cells and
                    (all_identifiers[coord[0]], local_index[coord[0]]) not in connected_nodes) or \
                        (all_identifiers[coord[1]] in active_cells and
                         (all_identifiers[coord[1]], local_index[coord[1]]) not in connected_nodes):
                    # if ('boundary' not in all_identifiers[coord[0]] and
                    #     (all_identifiers[coord[0]], local_index[coord[0]]) not in connected_nodes) or \
                    #    ('boundary' not in all_identifiers[coord[1]] and
                    #     (all_identifiers[coord[1]], local_index[coord[1]]) not in connected_nodes):

                    self.slow_adhesions.append(Adhesion(cells=(self.cellDict[all_identifiers[coord[0]]],
                                                               self.cellDict[all_identifiers[coord[1]]]),
                                                        s_coords=(local_index[coord[0]], local_index[coord[1]]),
                                                        average_lifespan=self.slow_adhesion_lifespan,
                                                        adhesion_type='cadherin'))
                    connected_nodes.update([(all_identifiers[coord[0]], local_index[coord[0]])])
                    connected_nodes.update([(all_identifiers[coord[1]], local_index[coord[1]])])

            # Add the data at the cell level
            # Slow adhesion entry = (local_cell_index, other_cell_x, other_cell_y, other_cell_mesh_spacing)
            for cell in self.cells:
                cell.slow_adhesions = []
            for ad in self.slow_adhesions:
                ad_cell_1, ad_cell_2 = ad.cell_1, ad.cell_2
                if ad.cell_1.identifier in apply_to:
                    # ad.cell_1.slow_adhesions.append(ad)
                    xy_nabo = ad.get_xy_at_other_end(ad.cell_1.identifier)
                    ad.cell_1.slow_adhesions.append([ad.cell_1_index, xy_nabo[0], xy_nabo[1],
                                                     ad_cell_2.deformed_mesh_spacing[ad.cell_2_index]])
                if ad.cell_2.identifier in apply_to:
                    # ad.cell_2.slow_adhesions.append(ad)
                    xy_nabo = ad.get_xy_at_other_end(ad.cell_2.identifier)
                    ad.cell_2.slow_adhesions.append([ad.cell_2_index, xy_nabo[0], xy_nabo[1],
                                                     ad_cell_1.deformed_mesh_spacing[ad.cell_1_index]])

    def update_sidekick_adhesions(self, fresh_sdk=True, apply_to='all'):
        """Update the sidekick vertices in the tissue and then add data to cells.
        As with slow adhesions, these are stored in the Cell classes as:
        slow_ad = (local_cell_index, other_cell_x, other_cell_y, other_cell_mesh_spacing)
        rather than as adhesion objects.

        :param fresh_sdk:  (Default value = True)  Remove the existing sidekick and make new ones.
        :type fresh_sdk: bool
        :param apply_to:  (Default value = 'all')  If not ``'all'``, a list of cell identifiers that the function will be applied to.
        :type apply_to: list

        """
        self.verboseprint("Updating sidekick adhesions", object(), 1)

        apply_to = [c.identifier for c in self.cells] if apply_to == "all" else apply_to

        if fresh_sdk or len(self.sidekick_adhesions) == 0:
            apply_to_set = set(apply_to)
            tissue_vertices = self.get_all_tissue_vertices(apply_to=apply_to)
            sidekick_vertices = [v for v in tissue_vertices if set(v).intersection(apply_to_set)]
            # Add s_coords of vertices in tuple form ('cell_id', s_loc)
            sidekick_vertices = [list(zip(v, self.get_cortex_coord_of_vertex_triangle(*v))) for v in sidekick_vertices]

            # Add the vertices as adhesions to each cell
            self.sidekick_adhesions = []
            for vert in sidekick_vertices:
                # Append all three adhesions
                c1, c2 = self.cellDict[vert[0][0]], self.cellDict[vert[1][0]]
                self.sidekick_adhesions.append(Adhesion(cells=(c1, c2), s_coords=(c1.s[vert[0][1]], c2.s[vert[1][1]]),
                                                        average_lifespan=np.inf,
                                                        adhesion_type='sidekick'))
                c1, c2 = self.cellDict[vert[0][0]], self.cellDict[vert[2][0]]
                self.sidekick_adhesions.append(Adhesion(cells=(c1, c2), s_coords=(c1.s[vert[0][1]], c2.s[vert[2][1]]),
                                                        average_lifespan=np.inf,
                                                        adhesion_type='sidekick'))
                c1, c2 = self.cellDict[vert[1][0]], self.cellDict[vert[2][0]]
                self.sidekick_adhesions.append(Adhesion(cells=(c1, c2), s_coords=(c1.s[vert[1][1]], c2.s[vert[2][1]]),
                                                        average_lifespan=np.inf,
                                                        adhesion_type='sidekick'))

            # Reset rest lengths
            average_len = np.median([ad.get_length() for ad in self.sidekick_adhesions])
            for cell in self.cells:
                cell.sdk_restlen = average_len

        for cell in self.cells:
            cell.sidekick_adhesions = []
        # Add the adhesions to cells.
        # sdk adhesion entry in a cell = (local_cell_index, other_cell_x, other_cell_y)
        for ad in self.sidekick_adhesions:
            if ad.cell_1.identifier in apply_to:
                # ad.cell_1.sidekick_adhesions.append(ad)
                xy_nabo = ad.get_xy_at_other_end(ad.cell_1.identifier)
                ad.cell_1.sidekick_adhesions.append([ad.cell_1_index, xy_nabo[0], xy_nabo[1],
                                                     ad.get_spacing_at_other_end(ad.cell_1.identifier)])
            if ad.cell_2.identifier in apply_to:
                # ad.cell_2.sidekick_adhesions.append(ad)
                xy_nabo = ad.get_xy_at_other_end(ad.cell_2.identifier)
                ad.cell_2.sidekick_adhesions.append([ad.cell_2_index, xy_nabo[0], xy_nabo[1],
                                                     ad.get_spacing_at_other_end(ad.cell_2.identifier)])

    def update_fast_adhesions(self, build_trees=True, apply_to='all'):
        """Updates the population of fast turnover adhesions, tau_ad < tau_cortex
        Adds the current positions of the cortices as the possible adhesion points for each cortex

        :param build_trees:  (Default value = True)  Rebuild the parameter trees with adhesion nodes.
        :type build_trees: bool
        :param apply_to:  (Default value = 'all')  If not ``'all'``, a list of cell identifiers that the function will be applied to.
        :type apply_to: list

        """
        building_text = "and building new tree" if build_trees else "but not building new tree"
        self.verboseprint("".join(["updating fast adhesions on all cortices ", building_text]), object(), 1)

        # Build a list with all of the points, spacings and identifiers.
        all_points = []
        all_spacings = []
        all_identifiers = []
        # for cell in self.cells:
        for cell in self.cellDict.values():
            all_points.extend(np.dstack((cell.x, cell.y))[0])
            all_spacings.extend(cell.get_xy_segment_lengths())
            all_identifiers.extend([cell.identifier] * cell.x.size)

        # Add the wall
        # # all_points.extend(np.dstack((self.boundary_adhesions[0], self.boundary_adhesions[1]))[0])
        # # all_spacings.extend(self.get_xy_segment_lengths(self.boundary_adhesions[0], self.boundary_adhesions[1]))
        # # all_identifiers.extend(['boundary'] * len(self.boundary_adhesions[0]))
        # all_points.extend(np.dstack((self.boundary_cell.x, self.boundary_cell.y))[0])
        # all_spacings.extend([i for i in self.boundary_cell.deformed_mesh_spacing])
        # all_identifiers.extend(['boundary'] * self.boundary_cell.x.size)

        # Need arrays
        all_points = np.array(all_points)
        all_spacings = np.array(all_spacings)
        all_identifiers = np.array(all_identifiers)

        #  For every cell add the possible adhesions within max_adhesion_length
        apply_to = [c.identifier for c in self.cells] if apply_to == "all" else apply_to
        for ref in apply_to:
            cell = self.cellDict[ref]
            # Minimum distance criterion
            cuttoff = cell.adhesion_search_radius * 1.1

            # Cell coords
            cell_coords = np.dstack((cell.x, cell.y))[0]

            # On the fixed wall
            indices_to_adhesions = self.get_indices_of_points_within_distance(all_points, cell_coords, cuttoff)

            # Filter the big lists
            adhesions_reached = all_points[indices_to_adhesions]
            adhesions_reached_spacings = all_spacings[indices_to_adhesions]
            adhesions_reached_identifiers = all_identifiers[indices_to_adhesions]

            # Filter out this cell
            adhesions_reached = adhesions_reached[adhesions_reached_identifiers != cell.identifier]
            adhesions_reached_spacings = adhesions_reached_spacings[adhesions_reached_identifiers != cell.identifier]
            adhesions_reached_identifiers = adhesions_reached_identifiers[
                adhesions_reached_identifiers != cell.identifier]

            # Store
            cell.update_adhesion_points(adhesions_reached, adhesions_reached_identifiers,
                                        adhesions_reached_spacings)

            if build_trees:
                # cell.build_adhesion_tree()
                cell.update_adhesion_distances_identifiers_and_indices(build_tree=build_trees)

    def update_adhesion_points_between_all_cortices(self, only_fast=False, build_trees=True, apply_to='all',
                                                    fresh_sdk=False, prune_slow=False):
        """Update all specified adhesion types.
        By default, doesn't prune slow because they may be updated before time progresses

        :param only_fast:  (Default value = False)  Update only fast adhesions (and thereby prestretches).
        :type only_fast: bool
        :param build_trees:  (Default value = True)  Update adhesion trees.
        :type build_trees: bool
        :param apply_to:  (Default value = 'all')  If not ``'all'``, a list of cell identifiers that the function will be applied to.
        :type apply_to: list
        :param fresh_sdk:  (Default value = False)  Update sidkekick adhesions.
        :type fresh_sdk: bool
        :param prune_slow:  (Default value = False)  Remove old and long slow adhesions.
        :type prune_slow: bool

        """

        # Verbose info
        building_text = "and building new tree" if build_trees else "but not building new tree"
        self.verboseprint("".join(["updating adhesion points on all cortices ", building_text]), object(), 1)

        if self.within_hexagons:
            if len(self.cells) == 1:
                self.update_adhesions_for_hexagons()
            else:
                self._update_adhesions_for_free_boundary_hexagons()

        else:
            self.set_adhesion_to_fixed_line_bool(False)

            # Fast adhesions
            self.update_fast_adhesions(build_trees=build_trees, apply_to=apply_to)

            # # We need to update the s_index dict if it's empty from saving
            # if self.slow_adhesions_active or self.sidekick_active:
            #     for cell in self.cells:
            #         if cell.s_index_dict == 0:
            #             cell.s_index_dict = {s: idx for (idx, s) in enumerate(cell.s)}

        if not only_fast:

            # Slow adhesion population
            if self.slow_adhesions_active:
                self.update_slow_adhesions(prune=prune_slow, apply_to=apply_to)
            else:
                for cell in self.cells:
                    cell.slow_adhesions = []

            # Sidekick
            if self.sidekick_active:
                self.update_sidekick_adhesions(fresh_sdk=fresh_sdk, apply_to=apply_to)
            else:
                for cell in self.cells:
                    cell.sidekick_adhesions = []

    def remove_sdk_from_a_junction(self, junc):
        """Removes sdk from the vertices of a junction defined as junc = [cell_ref_1, cell_ref_2]

        :param junc:  A list of the two cell identities that make the bicellular junction.
        :type junc: list

        """

        c1, c2 = self.cellDict[junc[0]], self.cellDict[junc[1]]

        assert (junc[0] in list(itertools.chain.from_iterable(c2.adhesion_connections_identities)) or
                junc[1] in list(itertools.chain.from_iterable(c1.adhesion_connections_identities))), \
            "Error, cells %s and %s don't share a junction" % (junc[0], junc[1])

        # Store a list of all cells connected to the junction vertices
        cells_at_junction_vertices = {c1.identifier, c2.identifier}
        # First get the other cells connected to the vertices
        # Get neighbours of both cells
        naboer = set(c1.get_neighbours())
        naboer.update(c2.get_neighbours())
        # If a neighbour contains c1 and c2, then it belongs to the vertex
        for nabo_ref in naboer:
            nabo = self.cellDict[nabo_ref]
            nabo_naboer = nabo.get_neighbours()
            if junc[0] in nabo_naboer and junc[1] in nabo_naboer:
                cells_at_junction_vertices.update([nabo.identifier])

        # Now remove all sdk triangles where all cells are at these vertices
        self.sidekick_adhesions = [ad for ad in self.sidekick_adhesions if not
        {ad.cell_1.identifier, ad.cell_2.identifier}.issubset(cells_at_junction_vertices)]

    def get_indices_of_points_within_distance(self, points1, points2, cutoff):
        """Function to return the indices of points 1 that are within 'cutoff' of points 2.
        \todo make a sklearn tree of the wall to increase speed.

        :param points1:  The list of (x,y) points that will find nearest points in points2
        :type points1: list
        :param points2:  The reference list of points.
        :type points2: list
        :param cutoff:  Maximum search distance.
        :type cutoff: float
        :return indices:  The indices of points1 list that are within cutoff to points2.
        :rtype: list

        """
        # build the KDTree using the *larger* points array
        tree = cKDTree(points1)
        groups = tree.query_ball_point(points2, cutoff, n_jobs=-1)

        indices = list(set(itertools.chain.from_iterable(groups)))

        return indices

    def smooth_all_variables_with_spline(self, apply_to='all', smoothing=1e-3):
        """Use a spline to smooth the variables.

        :param apply_to:  (Default value = 'all')  If not ``'all'``, a list of cell identifiers that the function will be applied to.
        :type apply_to: list
        :param smoothing:  (Default value = 1e-3)  How much smoothing to apply.
        :type smoothing: float

        """

        apply_to = [c.identifier for c in self.cells] if apply_to == "all" else apply_to

        for ref in apply_to:
            cell = self.cellDict[ref]

            theta_spline = splrep(cell.s, cell.theta, k=3, s=smoothing)
            cell.theta = splev(cell.s, theta_spline)
            cell.C = splev(cell.s, theta_spline, der=1)
            cell.D = splev(cell.s, theta_spline, der=2)

            x_spline = splrep(cell.s, cell.x, k=3, s=smoothing)
            cell.x = splev(cell.s, x_spline)
            y_spline = splrep(cell.s, cell.y, k=3, s=smoothing)
            cell.y = splev(cell.s, y_spline)

            gamma_spline = splrep(cell.s, cell.gamma, k=3, s=smoothing)
            cell.gamma = splev(cell.s, gamma_spline)

    def decimate_all_cells_onto_new_grid(self, factor):
        """Changes 'n' for every cell by interpolating their variables onto a new grid.

        :param factor:  The factor which which to scale the number of cortex nodes.
        :type factor: int

        """

        self.verboseprint("Decimating cortex variables by a factor of %i" % factor, object(), 1)

        for cell in self.cells:
            cell.decimate_all_variables_onto_new_grid(factor)

        self.boundary_adhesions = np.array([decimate(self.boundary_adhesions[0], factor),
                                            decimate(self.boundary_adhesions[1], factor)])

    def set_cables_in_eptm(self, prestretch, id1_cells=None, id2_cells=None, cable_type='bipolar'):
        """Initialises PCP cables in the epithelium by assigning prestrain to mis-matched identities

        :param prestretch:  The magnitude of prestretch to apply to the cortices
        :type prestretch: float
        :param id1_cells:  (Default value = [])  List of cells to be identity 1
        :type id1_cells: list
        :param id2_cells:  (Default value = [])  List of cells to be identity 2
        :type id2_cells: list
        :param cable_type:  (Default value = 'bipolar')  unipolar or bipolar cables.
        :type cable_type: string

        """

        assert cable_type in ['unipolar', 'bipolar'], "Cable must be bipolar or unipolar"

        id1_cells = [] if id1_cells is None else id1_cells
        id2_cells = [] if id2_cells is None else id2_cells

        # Bestow identities
        if len(id1_cells) == 0:
            id1_cells = ['E', 'G', 'A', 'K', 'M', 'H', 'J', 'L']
            # id1_cells = ['I', 'E', 'G', 'A', 'K', 'M']
        if len(id2_cells) == 0:
            id2_cells = [cell.identifier for cell in self.cells if cell.identifier not in id1_cells]

        # Add prestrain properly
        for cell in self.cells:
            # Identify the id
            if cable_type == 'bipolar':
                cells_to_add_prestrain = id1_cells if cell.identifier in id2_cells else id2_cells
            elif cable_type == 'unipolar':
                cells_to_add_prestrain = [] if cell.identifier in id2_cells else id2_cells
            # Update
            cell.prestrain_dict.update(cell.prestrain_dict.fromkeys(cells_to_add_prestrain, prestretch))

    def impose_pcp(self, prestretch):
        """Applies prestrain to all vertical junctions.  Works only in the 14-cell tissue.

        :param prestretch:  The magnitude of prestretch.
        :type prestretch: float

        """

        for cell in self.cells:
            if cell.identifier == 'A':
                cell.prestrain_dict.update(cell.prestrain_dict.fromkeys(['I', 'B'], prestretch))
            elif cell.identifier == 'B':
                cell.prestrain_dict.update(cell.prestrain_dict.fromkeys(['A', 'J'], prestretch))
            elif cell.identifier == 'C':
                cell.prestrain_dict.update(cell.prestrain_dict.fromkeys(['G', 'H'], prestretch))
            elif cell.identifier == 'D':
                cell.prestrain_dict.update(cell.prestrain_dict.fromkeys(['K', 'L'], prestretch))
            elif cell.identifier == 'E':
                cell.prestrain_dict.update(cell.prestrain_dict.fromkeys(['F'], prestretch))
            elif cell.identifier == 'F':
                cell.prestrain_dict.update(cell.prestrain_dict.fromkeys(['E'], prestretch))
            elif cell.identifier == 'G':
                cell.prestrain_dict.update(cell.prestrain_dict.fromkeys(['C'], prestretch))
            elif cell.identifier == 'H':
                cell.prestrain_dict.update(cell.prestrain_dict.fromkeys(['C'], prestretch))
            elif cell.identifier == 'I':
                cell.prestrain_dict.update(cell.prestrain_dict.fromkeys(['A'], prestretch))
            elif cell.identifier == 'J':
                cell.prestrain_dict.update(cell.prestrain_dict.fromkeys(['B'], prestretch))
            elif cell.identifier == 'K':
                cell.prestrain_dict.update(cell.prestrain_dict.fromkeys(['D'], prestretch))
            elif cell.identifier == 'L':
                cell.prestrain_dict.update(cell.prestrain_dict.fromkeys(['D'], prestretch))
            elif cell.identifier == 'M':
                cell.prestrain_dict.update(cell.prestrain_dict.fromkeys(['N'], prestretch))
            elif cell.identifier == 'N':
                cell.prestrain_dict.update(cell.prestrain_dict.fromkeys(['M'], prestretch))

    def duplicate_cell(self, cell_to_copy: Cell, x_shift: float, y_shift: float, roll: int = 0, theta_shift: int = 0,
                       identifier: str = 'B'):
        """Duplicates a cell in the tissue, with a specified shift in x and y coords

        :param cell_to_copy:  Identity of cell that will be duplicated
        :type cell_to_copy: Cell
        :param x_shift:   Translation in x-direction.
        :type x_shift: float
        :param y_shift:  Translation in y-direction.
        :type y_shift: float
        :param roll:  (Default value = 0)  Roll the cell variables to change position of starting index.
        :type roll: int
        :param theta_shift:  (Default value = 0)  Shift theta by a scale factor (multiples of 2pi).
        :type theta_shift: float
        :param identifier:  (Default value = 'B')  Identitfier for new cell.
        :type identifier: string

        """

        new_cell = copy.deepcopy(cell_to_copy)
        self.cells.append(new_cell)

        # Get members of A
        members_of_A = [attr for attr in dir(cell_to_copy) if
                        not callable(getattr(cell_to_copy, attr)) and not attr.startswith("__")]
        # Make a deep copy each member
        for member in members_of_A:
            setattr(new_cell, member, copy.deepcopy(getattr(cell_to_copy, member)))

        new_cell.identifier = identifier
        self.cellDict[identifier] = new_cell

        # Update position in tissue
        new_cell.x += x_shift
        new_cell.y += y_shift
        new_cell.theta = new_cell.theta + 0.
        new_cell.C = new_cell.C + 0.
        new_cell.D = new_cell.D + 0.
        new_cell.gamma = new_cell.gamma + 0.

        # Roll if requested
        if roll != 0:
            new_cell.x = np.roll(new_cell.x, roll)
            new_cell.y = np.roll(new_cell.y, roll)
            new_cell.theta += theta_shift
            new_cell.C = np.roll(new_cell.C, roll)
            new_cell.D = np.roll(new_cell.D, roll)
            new_cell.gamma = np.roll(new_cell.gamma, roll)
            new_cell.s = np.roll(new_cell.s, roll)

    def _build_single_cell(self, radius=35, n=2000, param_dict=None, verbose=False,
                           identifier='A', cell_kwargs=None):
        """ Create a new, circular cell in the tissue.

        :param radius: (Default value = 35)  Radius of new cell.
        :type radius: float
        :param n:  (Default value = 2000)  Size of mesh for cell.
        :type n: int
        :param param_dict: param verbose: (Default value = {})  kappa, omega0, delta param dictionary.
        :type param_dict: dict
        :param identifier:  (Default value = 'A')  Optional paramters for cell.
        :type identifier: string
        :param verbose: Whether to print information to console. (Default value = False)
        :type verbose:  bool
        :param cell_kwargs:  (Default value = {})  Optional parameters for cell.
        :type cell_kwargs: dict

        """

        self.verboseprint("Creating 1 cell in hexagon", object(), 1)
        # Initialise circle A
        circle_radius = np.sqrt(3) * 0.5 * radius
        param_dict = {} if param_dict is None else param_dict
        delta = param_dict.get('delta', 1)  # / 2
        anglesA = np.linspace(0, np.pi * 2, n, endpoint=False)
        anglesA = np.roll(anglesA, int(anglesA.size / 2))
        pointsXA = (circle_radius - delta) * np.cos(anglesA)
        pointsYA = (circle_radius - delta) * np.sin(anglesA)
        anglesA = np.linspace(0, np.pi * 2, n, endpoint=False) + 3 * np.pi / 2

        # Make closed loop
        pointsXA = np.append(pointsXA, pointsXA[0])  # Add the bottom point to make a closed loop
        pointsYA = np.append(pointsYA, pointsYA[0])  # Add the bottom point to make a closed loop
        anglesA = np.append(anglesA, anglesA[0] + 2 * np.pi)  # Add the bottom point to make a closed loop

        total_points = anglesA.size
        ds = (2 * radius * np.pi) / total_points

        # Do the curvatures
        c_guessA = np.gradient(anglesA, ds, edge_order=2)
        # Do the curvatures derivatives
        d_guessA = np.gradient(c_guessA, ds, edge_order=2)  # np.linspace(0, 0, total_points)
        # TAUS
        gamma_guessA = np.ones(anglesA.size)

        # We just build the first cell and then create all other instances from this cell.
        initial_guessesA = {'D': d_guessA, 'C': c_guessA, 'gamma': gamma_guessA, 'theta': anglesA,
                            'x': pointsXA, 'y': pointsYA}
        cell_kwargs = {} if cell_kwargs is None else cell_kwargs
        cellA = Cell(initial_guesses=initial_guessesA, verbose=verbose, param_dict=param_dict, identifier=identifier,
                     cell_kwargs=cell_kwargs)

        self.cells = [cellA]
        # Dictionary to point to cells
        self.cellDict = {'A': self.cells[0]}

    def create_single_cell_periodic_tissue_from_fitted_single_cell(self):
        """
        Builds a periodic tissue from a single cell within a stencil.  The stencil is removed  and the cell adheres
        to itself within a rhombus or parallelogram.
        :return:
        """
        self.verboseprint("Creating a periodic eptm from cell A in hexagon", object(), 1)

        self.within_hexagons = False

        self.boundary_bc = 'periodic'

        self.slow_adhesions = []

        # Force centroid to (0,0)
        centroid = (np.mean(self.cells[0].x), np.mean(self.cells[0].y))
        self.cells[0].x -= centroid[0]
        self.cells[0].y -= centroid[1]
        self.cells[0].constrain_centroid = True

        # Define the paralellogram control parameters: (v1, v2) where v1 and v2 are the vector projections of the two
        # edges from the centroid.
        max_x = self.radius * np.sin(2 * np.pi * 1 / 6)
        max_y = self.radius * np.cos(2 * np.pi * 0 / 6)

        self.parallelogram = np.array([[max_x - 0.5 * self.cells[0].delta,
                                        -(1.5 * max_y - self.cells[0].delta * np.sin(np.pi / 3))],
                                       [max_x - 0.5 * self.cells[0].delta,
                                        1.5 * max_y - self.cells[0].delta * np.sin(np.pi / 3)]])

        # Remove the boundary stencil
        self.boundary_adhesions = np.array([[], []])
        self.reference_boundary_adhesions = np.array([[], []])
        self.boundary_cell.identifier = 'periodic_boundary'
        self.boundary_cell.x = np.array([])
        self.boundary_cell.y = np.array([])

        # Add cells
        for cell_id in ['B', 'C', 'D', 'E', 'F', 'G']:
            self.duplicate_cell(self.cells[0], 0, 0, identifier=cell_id)

        self._rebuild_periodic_boundary()

        self.cells = [self.cells[0]]
        # self.cellDict = {'A': self.cells[0], 'periodic_boundary': self.boundary_cell}

    def _rebuild_periodic_boundary(self):
        """
        Surround the cell with copies of itself and make them the boundary cell.
        :return:
        """

        # all_points, all_spacings = [], []

        real_cell = self.cells[0]
        real_cell.update_deformed_mesh_spacing()
        for vec_combo in [(1, 1, 'B'), (0, 1, 'C'), (0, -1, 'F'), (-1, -1, 'E'), (-1, 0, 'D'), (1, 0, 'G')]:
            x = real_cell.x + vec_combo[0] * self.parallelogram[0][0] + vec_combo[1] * self.parallelogram[1][0]
            y = real_cell.y + vec_combo[0] * self.parallelogram[0][1] + vec_combo[1] * self.parallelogram[1][1]

            self.cellDict[vec_combo[2]].x = x
            self.cellDict[vec_combo[2]].y = y
            self.cellDict[vec_combo[2]].s = self.cells[0].s
            self.cellDict[vec_combo[2]].update_deformed_mesh_spacing()
            # all_points.extend(np.dstack((x, y))[0])
            # all_spacings.extend(real_cell.deformed_mesh_spacing)

        # self.boundary_cell.x = np.array([i[0] for i in all_points])
        # self.boundary_cell.y = np.array([i[1] for i in all_points])
        # self.boundary_cell.prune_adhesion_data()
        # self.boundary_cell.s = np.arange(self.boundary_cell.x.size)
        # self.boundary_cell.deformed_mesh_spacing = np.array(all_spacings)
        # self.cellDict['periodic_boundary'] = self.boundary_cell

        # self.boundary_adhesions = [self.boundary_cell.x, self.boundary_cell.y]

    def deform_periodic_boundary(self, strain_tensor):
        """
        Deform the periodic boundary around the cell.
        :param strain_tensor: (2x2) matrix with strain components.
        :return:
        """
        self.cells[0].x = self.cells[0].x * strain_tensor[0, 0] + self.cells[0].y * strain_tensor[0, 1]
        self.cells[0].y = self.cells[0].x * strain_tensor[1, 0] + self.cells[0].y * strain_tensor[1, 1]

        new_v1 = np.dot(strain_tensor, self.parallelogram[0])
        new_v2 = np.dot(strain_tensor, self.parallelogram[1])

        self.parallelogram = np.array([new_v1, new_v2])

        self._rebuild_periodic_boundary()

    def _apply_deformation_relax_cortex_and_get_energy(self, stretch_shear):
        """
        Filler
        :param stretch_shear:
        :return:
        """

        strain = self.build_strain_tensor(stretch=stretch_shear[0], shear=stretch_shear[1])
        self.deform_periodic_boundary(strain)
        self.solve_bvps_in_parallel()
        energy = self.cells[0].integrate_cortex_energy()

        return energy

    def relax_periodic_tissue(self, upper_bound: float = 0.002, min_deformation: float = 1e-5, num_trials: int = 3,
                              max_runs: int = 5, n_jobs: int = -1):
        """Relaxes the parallelogram surrounding a tissue by minimising the energy.  Call a cortex relaxation
        internally to get the energy."""

        # Make sure we are in equillibrium before minimising (else all minimisation will have to do this)
        self.solve_bvps_in_parallel()

        # Get pressure and shear to predict if compression or tension is needed.
        # Note minus for pressure (since p_eff = - 1/2 trace(stress))
        cell_pressure = -self.cells[0].get_effective_pressure()
        cell_shear = self.cells[0].get_shear_stress()

        # From those, get bounds for deformation.
        stretch_bound = min([2 * abs(cell_pressure), upper_bound])
        shear_bound = min([2 * abs(cell_shear), upper_bound])
        # Calculate stretches
        stretches = np.linspace(0, np.sign(cell_pressure) * stretch_bound, num_trials)
        stretches = [s for s in stretches if s < min_deformation]
        shears = np.linspace(0, np.sign(cell_shear) * shear_bound, num_trials)
        shears = [s for s in shears if s < min_deformation]
        all_trails = list(itertools.product(stretches, shears))

        # # We buckle under compression so reset the lower bound
        # average_tension = (self.cells[0].gamma * self.cells[0].get_prestrains()).mean()
        # lower_bound = max([lower_bound, average_tension - 1]) if average_tension < 1 else lower_bound
        # bounds = (lower_bound, upper_bound)
        #
        # # Get the stretches and shears that we will try
        # stretches = np.linspace(bounds[0], bounds[1], num_trials)
        # shears = np.linspace(bounds[0], bounds[1], num_trials)
        # all_trails = list(itertools.product(stretches, shears))

        success = False
        num_evals = 0
        while not success:
            # stretch_shear = np.array([0, 0])
            # # max_minimisations: int = 1, max_nfev: int = 10
            # # solution = minimize(self._apply_deformation_relax_cortex_and_get_energy, stretch_shear, args=(),
            # #                     method='COBYLA', options={'rhobeg': 0.005, 'maxiter': max_nfev})
            # solution = minimize(self._apply_deformation_relax_cortex_and_get_energy, stretch_shear, args=(),
            #                     method='L-BFGS-B', bounds=((-0.01, 0.01), (-0.01, 0.01)),
            #                     options={'maxfun': max_nfev})
            # print(solution)
            # stretch_shear = solution.x
            #
            # # Apply lots of deformations and see what minimises the energy
            # energies = Parallel(n_jobs=n_jobs)(delayed(self._apply_deformation_relax_cortex_and_get_energy)(stretch_shear)
            #                                    for stretch_shear in all_trails)
            energies = []
            for stretch_shear in all_trails:
                energies.append(self._apply_deformation_relax_cortex_and_get_energy(stretch_shear))
            min_energy_idx = np.argmin(energies)

            # Apply the minimising deformation
            best_stretch_shears = all_trails[min_energy_idx]
            strain = self.build_strain_tensor(stretch=best_stretch_shears[0], shear=best_stretch_shears[1])
            self.deform_periodic_boundary(strain)
            self.solve_bvps_in_parallel()

            num_evals += 1
            # success = (best_stretch_shears[0] not in bounds and best_stretch_shears[1] not in bounds)
            success = abs(best_stretch_shears[0]) < stretch_bound and abs(best_stretch_shears[1]) < shear_bound
            print('in bounds?', success, best_stretch_shears)
            success = num_evals >= max_runs if not success else success
            print('done solving?', success)

    def create_3_cell_eptm_from_fitted_single_cell(self):
        """Tissue with 3 cells enclosed in a stencil.  Need to start with a single cell in a hexagon.

        """

        self.verboseprint("Creating eptm from cell A in hexagon", object(), 1)

        # copy a cortices. Warning: the immutable objects are copied as references, but I think this is ok
        # because they are always reassigned rather than edited.
        self.within_hexagons = False

        cellB = copy.deepcopy(self.cells[0])
        cellC = copy.deepcopy(self.cells[0])

        self.cells.extend([cellB, cellC])

        # Make deep copies of all member variables \todo (this is hacky...)
        # Get members of A
        members_of_A = [attr for attr in dir(self.cells[0]) if
                        not callable(getattr(self.cells[0], attr)) and not attr.startswith("__")]
        # Make a deep copy each member
        for i in range(1, len(self.cells)):
            for member in members_of_A:
                setattr(self.cells[i], member, copy.deepcopy(getattr(self.cells[0], member)))

        # identifiers
        self.cells[1].identifier = 'B'
        self.cells[2].identifier = 'C'

        # param dict
        self.cellDict['B'] = self.cells[1]
        self.cellDict['C'] = self.cells[2]

        # Get coords of cell A
        xs_A = self.cells[0].x + 0.
        ys_A = self.cells[0].y + 0.
        thetas_A = self.cells[0].theta + 0.
        c_A = self.cells[0].C + 0
        gamma_A = self.cells[0].gamma + 0.
        d_A = self.cells[0].D + 0.

        # Centroid
        centroid_xA = np.mean(xs_A)
        centroid_yA = np.mean(ys_A)

        deltaNew = self.delta
        rNew = self.radius
        numPoints = self.n

        n = np.array([0, 1, 2, 3, 4, 5])

        x = rNew * np.sin(2 * np.pi * n / 6)
        y = rNew * np.cos(2 * np.pi * n / 6)

        pointsX = [np.linspace(x[i - 1], x[i], int(numPoints / 6), endpoint=False) for i in range(0, x.size)]
        pointsX = np.array(pointsX).flatten()
        pointsY = [np.linspace(y[i - 1], y[i], int(numPoints / 6), endpoint=False) for i in range(0, y.size)]
        pointsY = np.array(pointsY).flatten()

        min_xA, min_yA = np.min(pointsX), np.min(pointsY)
        max_xA, max_yA = np.max(pointsX), np.max(pointsY)

        # For bounding hexagon
        bounding_points_x = [b_coord for b_coord in pointsX]
        bounding_points_y = [b_coord for b_coord in pointsY]

        ### For B ###
        self.cells[1].x = xs_A + 2 * (max_xA - centroid_xA) - self.cells[0].delta
        self.cells[1].y = ys_A + 0.
        self.cells[1].theta = thetas_A + 0.
        self.cells[1].C = c_A + 0.
        self.cells[1].D = d_A + 0.
        self.cells[1].gamma = gamma_A + 0.

        self.cells[1].x = np.roll(self.cells[1].x, int(numPoints / 2))
        self.cells[1].y = np.roll(self.cells[1].y, int(numPoints / 2))
        # self.cells[1].theta = np.roll(self.cells[1].theta, int(numPoints / 2))
        self.cells[1].theta += np.pi
        self.cells[1].C = np.roll(self.cells[1].C, int(numPoints / 2))
        self.cells[1].D = np.roll(self.cells[1].D, int(numPoints / 2))
        self.cells[1].gamma = np.roll(self.cells[1].gamma, int(numPoints / 2))

        bounding_points_x.extend(pointsX + 2 * (max_xA - centroid_xA) - self.cells[0].delta)
        bounding_points_y.extend(pointsY + 0)

        ### For C ###
        # self.cells[2].x = xs_A + 0.5 * (max_xA - min_xA + self.cells[0].delta)
        self.cells[2].x = xs_A + max_xA - centroid_xA - 0.5 * self.cells[0].delta
        # 1.5 * np.max(pointsY) + deltaNew * np.sin(np.pi / 3)
        self.cells[2].y = ys_A + 1.5 * np.max(pointsY) - self.cells[0].delta * np.sin(np.pi / 3)
        # + 0.5 * self.cells[0].delta * np.cos(np.pi / 3)
        # self.cells[2].y = ys_A + (max_yA - min_yA + self.cells[0].delta) * np.sin(np.pi / 3)
        self.cells[2].theta = thetas_A + 0.
        self.cells[2].C = c_A + 0.
        self.cells[2].D = d_A + 0.
        self.cells[2].gamma = gamma_A + 0.

        for cell in self.cells:
            cell.update_deformed_mesh_spacing()

        bounding_points_x.extend(pointsX + max_xA - centroid_xA - 0.5 * self.cells[0].delta)
        bounding_points_y.extend(pointsY + 1.5 * np.max(pointsY) - self.cells[0].delta * np.sin(np.pi / 3))

        # Finish up the boundary
        all_points = np.dstack((bounding_points_x, bounding_points_y))[0]

        # This keeps on the hexagon boundaries
        hull = ConcaveHull()
        hull.loadpoints(all_points)
        hull.calculatehull(tol=self.concave_hull_tolerance * 1)

        scaled_boundary = scale(hull.boundary.exterior, xfact=1, yfact=1, zfact=1, origin='centroid')
        x, y = scaled_boundary.xy
        x = np.array([i for i in x])
        y = np.array([i for i in y])

        # Save
        self.boundary_adhesions = np.array([x, y])
        self.boundary_cell.x = x
        self.boundary_cell.y = y
        self.boundary_cell.s = np.arange(x.size)
        self.boundary_cell.update_deformed_mesh_spacing()

        self.update_adhesion_points_between_all_cortices()
        self.reference_boundary_adhesions = []

    def create_14_cell_eptm_from_fitted_single_cell(self):
        """ Build a 14-cell tissue, symmetric in x and y, from a tissue made up of a single cell in a hexagon.

        """

        self.verboseprint("Creating eptm from cell A in hexagon", object(), 1)

        # copy a cortices. Warning: the immutable objects are copied as references, but I think this is ok
        # because they are always reassigned rather than edited.
        self.within_hexagons = False

        # Get coords of cell A
        xs_A = self.cells[0].x + 0.
        ys_A = self.cells[0].y + 0.
        # Centroid
        centroid_xA = np.mean(xs_A)
        centroid_yA = np.mean(ys_A)

        # Calculations for bounding hexagon
        deltaNew = self.delta
        rNew = self.radius
        numPoints = self.n

        n = np.array([0, 1, 2, 3, 4, 5])

        x = rNew * np.sin(2 * np.pi * n / 6)
        y = rNew * np.cos(2 * np.pi * n / 6)

        pointsX = [np.linspace(x[i - 1], x[i], int(numPoints / 6), endpoint=False) for i in range(0, x.size)]
        pointsX = np.array(pointsX).flatten()
        pointsY = [np.linspace(y[i - 1], y[i], int(numPoints / 6), endpoint=False) for i in range(0, y.size)]
        pointsY = np.array(pointsY).flatten()

        min_xA, min_yA = np.min(pointsX), np.min(pointsY)
        max_xA, max_yA = np.max(pointsX), np.max(pointsY)

        # For bounding stencil
        bounding_points_x = []
        bounding_points_y = []

        # B
        x_shift = 2 * (max_xA - centroid_xA) - self.cells[0].delta
        y_shift = 0
        self.duplicate_cell(self.cells[0], x_shift, y_shift, roll=int(numPoints / 2), theta_shift=np.pi, identifier='B')
        # C
        x_shift = max_xA - centroid_xA - 0.5 * self.cells[0].delta
        y_shift = 1.5 * np.max(pointsY) - self.cells[0].delta * np.sin(np.pi / 3)
        self.duplicate_cell(self.cells[0], x_shift, y_shift, roll=0, theta_shift=0, identifier='C')
        # D
        x_shift = max_xA - centroid_xA - 0.5 * self.cells[0].delta
        y_shift = - (1.5 * np.max(pointsY) - self.cells[0].delta * np.sin(np.pi / 3))
        self.duplicate_cell(self.cells[0], x_shift, y_shift, roll=0, theta_shift=0, identifier='D')
        # E
        x_shift = 0
        y_shift = 2 * (1.5 * np.max(pointsY) - self.cells[0].delta * np.sin(np.pi / 3))
        self.duplicate_cell(self.cells[0], x_shift, y_shift, roll=0, theta_shift=0, identifier='E')
        # Its at boundary
        bounding_points_x.extend(pointsX + 0)
        bounding_points_y.extend(pointsY + 2 * (1.5 * np.max(pointsY) - self.cells[0].delta * np.sin(np.pi / 3)))
        # F
        x_shift = 2 * (max_xA - centroid_xA) - self.cells[0].delta
        y_shift = 2 * (1.5 * np.max(pointsY) - self.cells[0].delta * np.sin(np.pi / 3))
        self.duplicate_cell(self.cells[0], x_shift, y_shift, roll=0, theta_shift=0, identifier='F')
        # Its at boundary
        bounding_points_x.extend(pointsX + 2 * (max_xA - centroid_xA) - self.cells[0].delta)
        bounding_points_y.extend(pointsY + 2 * (1.5 * np.max(pointsY) - self.cells[0].delta * np.sin(np.pi / 3)))
        # G
        x_shift = - (max_xA - centroid_xA - 0.5 * self.cells[0].delta)
        y_shift = 1.5 * np.max(pointsY) - self.cells[0].delta * np.sin(np.pi / 3)
        self.duplicate_cell(self.cells[0], x_shift, y_shift, roll=0, theta_shift=0, identifier='G')
        # Its at boundary
        bounding_points_x.extend(pointsX - (max_xA - centroid_xA - 0.5 * self.cells[0].delta))
        bounding_points_y.extend(pointsY + 1.5 * np.max(pointsY) - self.cells[0].delta * np.sin(np.pi / 3))
        # For H
        x_shift = 3 * (max_xA - centroid_xA - 0.5 * self.cells[0].delta)
        y_shift = 1.5 * np.max(pointsY) - self.cells[0].delta * np.sin(np.pi / 3)
        self.duplicate_cell(self.cells[0], x_shift, y_shift, roll=0, theta_shift=0, identifier='H')
        # Its at boundary
        bounding_points_x.extend(pointsX + 3 * (max_xA - centroid_xA - 0.5 * self.cells[0].delta))
        bounding_points_y.extend(pointsY + 1.5 * np.max(pointsY) - self.cells[0].delta * np.sin(np.pi / 3))
        # For I
        x_shift = - 2 * (max_xA - centroid_xA - 0.5 * self.cells[0].delta)
        y_shift = 0
        self.duplicate_cell(self.cells[0], x_shift, y_shift, roll=0, theta_shift=0, identifier='I')
        # Its at boundary
        bounding_points_x.extend(pointsX - 2 * (max_xA - centroid_xA - 0.5 * self.cells[0].delta))
        bounding_points_y.extend(pointsY + 0)
        # For J
        x_shift = 4 * (max_xA - centroid_xA - 0.5 * self.cells[0].delta)
        y_shift = 0
        self.duplicate_cell(self.cells[0], x_shift, y_shift, roll=0, theta_shift=0, identifier='J')
        # Its at boundary
        bounding_points_x.extend(pointsX + 4 * (max_xA - centroid_xA - 0.5 * self.cells[0].delta))
        bounding_points_y.extend(pointsY + 0)
        # For K
        x_shift = - (max_xA - centroid_xA - 0.5 * self.cells[0].delta)
        y_shift = - (1.5 * np.max(pointsY) - self.cells[0].delta * np.sin(np.pi / 3))
        self.duplicate_cell(self.cells[0], x_shift, y_shift, roll=0, theta_shift=0, identifier='K')
        # Its at boundary
        bounding_points_x.extend(pointsX - (max_xA - centroid_xA - 0.5 * self.cells[0].delta))
        bounding_points_y.extend(pointsY - (1.5 * np.max(pointsY) - self.cells[0].delta * np.sin(np.pi / 3)))
        ### For L ###
        x_shift = 3 * (max_xA - centroid_xA - 0.5 * self.cells[0].delta)
        y_shift = - (1.5 * np.max(pointsY) - self.cells[0].delta * np.sin(np.pi / 3))
        self.duplicate_cell(self.cells[0], x_shift, y_shift, roll=0, theta_shift=0, identifier='L')
        # Its at boundary
        bounding_points_x.extend(pointsX + 3 * (max_xA - centroid_xA - 0.5 * self.cells[0].delta))
        bounding_points_y.extend(pointsY - (1.5 * np.max(pointsY) - self.cells[0].delta * np.sin(np.pi / 3)))
        ### For M ###
        x_shift = 0
        y_shift = - 2 * (1.5 * np.max(pointsY) - self.cells[0].delta * np.sin(np.pi / 3))
        self.duplicate_cell(self.cells[0], x_shift, y_shift, roll=0, theta_shift=0, identifier='M')
        # Its at boundary
        bounding_points_x.extend(pointsX + 0)
        bounding_points_y.extend(pointsY - 2 * (1.5 * np.max(pointsY) - self.cells[0].delta * np.sin(np.pi / 3)))
        ### For N ###
        x_shift = 2 * (max_xA - centroid_xA) - self.cells[0].delta
        y_shift = - 2 * (1.5 * np.max(pointsY) - self.cells[0].delta * np.sin(np.pi / 3))
        self.duplicate_cell(self.cells[0], x_shift, y_shift, roll=0, theta_shift=0, identifier='N')
        # Its at boundary
        bounding_points_x.extend(pointsX + 2 * (max_xA - centroid_xA) - self.cells[0].delta)
        bounding_points_y.extend(pointsY - 2 * (1.5 * np.max(pointsY) - self.cells[0].delta * np.sin(np.pi / 3)))

        for cell in self.cells:
            cell.update_deformed_mesh_spacing()

        ###### Update the boundary #######
        all_points = np.dstack((bounding_points_x, bounding_points_y))[0]
        # This keeps on the hexagon boundaries
        hull = ConcaveHull()
        hull.loadpoints(all_points)
        hull.calculatehull(tol=self.concave_hull_tolerance * 1)
        # Scale it
        scaled_boundary = scale(hull.boundary.exterior, xfact=1, yfact=1, zfact=1, origin='centroid')
        x, y = scaled_boundary.xy
        x = np.array([i for i in x])
        y = np.array([i for i in y])

        # Save
        self.boundary_adhesions = np.array([x, y])
        self.boundary_cell.x = x
        self.boundary_cell.y = y
        self.boundary_cell.s = np.arange(x.size)
        self.boundary_cell.update_deformed_mesh_spacing()

        self.update_adhesion_points_between_all_cortices()
        self.reference_boundary_adhesions = []

    def create_17_cell_eptm_from_fitted_single_cell(self):
        """Create a tissue with 17 cells and two axes of symmetry from a tissue with a single cell in a hexagon.

        """

        self.verboseprint("Creating eptm from cell A in hexagon", object(), 1)

        # copy a cortices. Warning: the immutable objects are copied as references, but I think this is ok
        # because they are always reassigned rather than edited.
        self.within_hexagons = False

        # Get coords of cell A
        xs_A = self.cells[0].x + 0.
        ys_A = self.cells[0].y + 0.
        # Centroid
        centroid_xA = np.mean(xs_A)
        centroid_yA = np.mean(ys_A)

        # Calculations for bounding hexagon
        deltaNew = self.delta
        rNew = self.radius
        numPoints = self.n

        n = np.array([0, 1, 2, 3, 4, 5])

        x = rNew * np.sin(2 * np.pi * n / 6)
        y = rNew * np.cos(2 * np.pi * n / 6)

        pointsX = [np.linspace(x[i - 1], x[i], int(numPoints / 6), endpoint=False) for i in range(0, x.size)]
        pointsX = np.array(pointsX).flatten()
        pointsY = [np.linspace(y[i - 1], y[i], int(numPoints / 6), endpoint=False) for i in range(0, y.size)]
        pointsY = np.array(pointsY).flatten()

        min_xA, min_yA = np.min(pointsX), np.min(pointsY)
        max_xA, max_yA = np.max(pointsX), np.max(pointsY)

        # For bounding stencil
        bounding_points_x = []
        bounding_points_y = []

        # B
        x_shift = 2 * (max_xA - centroid_xA) - self.cells[0].delta
        y_shift = 0
        self.duplicate_cell(self.cells[0], x_shift, y_shift, roll=int(numPoints / 2), theta_shift=np.pi, identifier='B')
        # C
        x_shift = max_xA - centroid_xA - 0.5 * self.cells[0].delta
        y_shift = 1.5 * np.max(pointsY) - self.cells[0].delta * np.sin(np.pi / 3)
        self.duplicate_cell(self.cells[0], x_shift, y_shift, roll=0, theta_shift=0, identifier='C')
        # D
        x_shift = max_xA - centroid_xA - 0.5 * self.cells[0].delta
        y_shift = - (1.5 * np.max(pointsY) - self.cells[0].delta * np.sin(np.pi / 3))
        self.duplicate_cell(self.cells[0], x_shift, y_shift, roll=0, theta_shift=0, identifier='D')
        # E
        x_shift = 0
        y_shift = 2 * (1.5 * np.max(pointsY) - self.cells[0].delta * np.sin(np.pi / 3))
        self.duplicate_cell(self.cells[0], x_shift, y_shift, roll=0, theta_shift=0, identifier='E')
        # Its at boundary
        bounding_points_x.extend(pointsX + 0)
        bounding_points_y.extend(pointsY + 2 * (1.5 * np.max(pointsY) - self.cells[0].delta * np.sin(np.pi / 3)))
        # F
        x_shift = 2 * (max_xA - centroid_xA) - self.cells[0].delta
        y_shift = 2 * (1.5 * np.max(pointsY) - self.cells[0].delta * np.sin(np.pi / 3))
        self.duplicate_cell(self.cells[0], x_shift, y_shift, roll=0, theta_shift=0, identifier='F')
        # Its at boundary
        bounding_points_x.extend(pointsX + 2 * (max_xA - centroid_xA) - self.cells[0].delta)
        bounding_points_y.extend(pointsY + 2 * (1.5 * np.max(pointsY) - self.cells[0].delta * np.sin(np.pi / 3)))
        # G
        x_shift = - (max_xA - centroid_xA - 0.5 * self.cells[0].delta)
        y_shift = 1.5 * np.max(pointsY) - self.cells[0].delta * np.sin(np.pi / 3)
        self.duplicate_cell(self.cells[0], x_shift, y_shift, roll=0, theta_shift=0, identifier='G')
        # For H
        x_shift = 3 * (max_xA - centroid_xA - 0.5 * self.cells[0].delta)
        y_shift = 1.5 * np.max(pointsY) - self.cells[0].delta * np.sin(np.pi / 3)
        self.duplicate_cell(self.cells[0], x_shift, y_shift, roll=0, theta_shift=0, identifier='H')
        # Its at boundary
        bounding_points_x.extend(pointsX + 3 * (max_xA - centroid_xA - 0.5 * self.cells[0].delta))
        bounding_points_y.extend(pointsY + 1.5 * np.max(pointsY) - self.cells[0].delta * np.sin(np.pi / 3))
        # For I
        x_shift = - 2 * (max_xA - centroid_xA - 0.5 * self.cells[0].delta)
        y_shift = 0
        self.duplicate_cell(self.cells[0], x_shift, y_shift, roll=0, theta_shift=0, identifier='I')
        # Its at boundary
        bounding_points_x.extend(pointsX + x_shift)
        bounding_points_y.extend(pointsY + y_shift)
        # For J
        x_shift = 4 * (max_xA - centroid_xA - 0.5 * self.cells[0].delta)
        y_shift = 0
        self.duplicate_cell(self.cells[0], x_shift, y_shift, roll=0, theta_shift=0, identifier='J')
        # Its at boundary
        bounding_points_x.extend(pointsX + 4 * (max_xA - centroid_xA - 0.5 * self.cells[0].delta))
        bounding_points_y.extend(pointsY + 0)
        # For K
        x_shift = - (max_xA - centroid_xA - 0.5 * self.cells[0].delta)
        y_shift = - (1.5 * np.max(pointsY) - self.cells[0].delta * np.sin(np.pi / 3))
        self.duplicate_cell(self.cells[0], x_shift, y_shift, roll=0, theta_shift=0, identifier='K')
        # Its at boundary
        bounding_points_x.extend(pointsX + x_shift)
        bounding_points_y.extend(pointsY + y_shift)
        ### For L ###
        x_shift = 3 * (max_xA - centroid_xA - 0.5 * self.cells[0].delta)
        y_shift = - (1.5 * np.max(pointsY) - self.cells[0].delta * np.sin(np.pi / 3))
        self.duplicate_cell(self.cells[0], x_shift, y_shift, roll=0, theta_shift=0, identifier='L')
        # Its at boundary
        bounding_points_x.extend(pointsX + 3 * (max_xA - centroid_xA - 0.5 * self.cells[0].delta))
        bounding_points_y.extend(pointsY - (1.5 * np.max(pointsY) - self.cells[0].delta * np.sin(np.pi / 3)))
        ### For M ###
        x_shift = 0
        y_shift = - 2 * (1.5 * np.max(pointsY) - self.cells[0].delta * np.sin(np.pi / 3))
        self.duplicate_cell(self.cells[0], x_shift, y_shift, roll=0, theta_shift=0, identifier='M')
        # Its at boundary
        bounding_points_x.extend(pointsX + 0)
        bounding_points_y.extend(pointsY - 2 * (1.5 * np.max(pointsY) - self.cells[0].delta * np.sin(np.pi / 3)))
        ### For N ###
        x_shift = 2 * (max_xA - centroid_xA) - self.cells[0].delta
        y_shift = - 2 * (1.5 * np.max(pointsY) - self.cells[0].delta * np.sin(np.pi / 3))
        self.duplicate_cell(self.cells[0], x_shift, y_shift, roll=0, theta_shift=0, identifier='N')
        # Its at boundary
        bounding_points_x.extend(pointsX + 2 * (max_xA - centroid_xA) - self.cells[0].delta)
        bounding_points_y.extend(pointsY - 2 * (1.5 * np.max(pointsY) - self.cells[0].delta * np.sin(np.pi / 3)))
        ### For O ###
        x_shift = - 2 * (max_xA - centroid_xA - 0.5 * self.cells[0].delta)
        y_shift = 2 * (1.5 * np.max(pointsY) - self.cells[0].delta * np.sin(np.pi / 3))
        self.duplicate_cell(self.cells[0], x_shift, y_shift, roll=0, theta_shift=0, identifier='O')
        # Its at boundary
        bounding_points_x.extend(pointsX - 2 * (max_xA - centroid_xA - 0.5 * self.cells[0].delta))
        bounding_points_y.extend(pointsY + 2 * (1.5 * np.max(pointsY) - self.cells[0].delta * np.sin(np.pi / 3)))
        ### For P ###
        x_shift = - 3 * (max_xA - centroid_xA - 0.5 * self.cells[0].delta)
        y_shift = 1.5 * np.max(pointsY) - self.cells[0].delta * np.sin(np.pi / 3)
        self.duplicate_cell(self.cells[0], x_shift, y_shift, roll=0, theta_shift=0, identifier='P')
        # Its at boundary
        bounding_points_x.extend(pointsX - 3 * (max_xA - centroid_xA - 0.5 * self.cells[0].delta))
        bounding_points_y.extend(pointsY + 1.5 * np.max(pointsY) - self.cells[0].delta * np.sin(np.pi / 3))
        ### For Q ###
        x_shift = 5 * (max_xA - centroid_xA - 0.5 * self.cells[0].delta)
        y_shift = - (1.5 * np.max(pointsY) - self.cells[0].delta * np.sin(np.pi / 3))
        self.duplicate_cell(self.cells[0], x_shift, y_shift, roll=0, theta_shift=0, identifier='Q')
        # Its at boundary
        bounding_points_x.extend(pointsX + x_shift)
        bounding_points_y.extend(pointsY - (1.5 * np.max(pointsY) - self.cells[0].delta * np.sin(np.pi / 3)))
        ### For R ###
        x_shift = 4 * (max_xA - centroid_xA - 0.5 * self.cells[0].delta)
        y_shift = - 2 * (1.5 * np.max(pointsY) - self.cells[0].delta * np.sin(np.pi / 3))
        self.duplicate_cell(self.cells[0], x_shift, y_shift, roll=0, theta_shift=0, identifier='R')
        # Its at boundary
        bounding_points_x.extend(pointsX + x_shift)
        bounding_points_y.extend(pointsY - 2 * (1.5 * np.max(pointsY) - self.cells[0].delta * np.sin(np.pi / 3)))

        for cell in self.cells:
            cell.update_deformed_mesh_spacing()

        # Prestrain dict
        p_dict = {'A': 1.0, 'B': 1.0, 'C': 1.0, 'D': 1.0, 'E': 1.0, 'F': 1.0, 'G': 1.0, 'H': 1.0, 'I': 1.0, 'J': 1.0,
                  'K': 1.0, 'L': 1.0, 'M': 1.0, 'N': 1.0, 'O': 1.0, 'P': 1.0, 'Q': 1.0, 'R': 1.0, 'S': 1.0,
                  'boundary': 1, 'none': 1}
        for cell in self.cells:
            cell.prestrain_dict = p_dict.copy()
            cell.adhesion_density_dict = p_dict.copy()

        ###### Update the boundary #######
        all_points = np.dstack((bounding_points_x, bounding_points_y))[0]
        # This keeps on the hexagon boundaries
        hull = ConcaveHull()
        hull.loadpoints(all_points)
        hull.calculatehull(tol=self.concave_hull_tolerance * 1)
        # Scale it
        scaled_boundary = scale(hull.boundary.exterior, xfact=1, yfact=1, zfact=1, origin='centroid')
        x, y = scaled_boundary.xy
        x = np.array([i for i in x])
        y = np.array([i for i in y])

        # Save
        self.boundary_adhesions = np.array([x, y])
        self.boundary_cell.x = x
        self.boundary_cell.y = y
        self.boundary_cell.s = np.arange(x.size)
        self.boundary_cell.update_deformed_mesh_spacing()

        self.update_adhesion_points_between_all_cortices()
        self.reference_boundary_adhesions = []

    def create_19_cell_eptm_from_fitted_single_cell(self):
        """Build a tissue with 19 cells from a tissue with a single cell enclosed in a hexagon.

        """

        self.verboseprint("Creating eptm from cell A in hexagon", object(), 1)

        # copy a cortices. Warning: the immutable objects are copied as references, but I think this is ok
        # because they are always reassigned rather than edited.
        self.within_hexagons = False

        # Get coords of cell A
        xs_A = self.cells[0].x + 0.
        ys_A = self.cells[0].y + 0.
        # Centroid
        centroid_xA = np.mean(xs_A)
        centroid_yA = np.mean(ys_A)

        # Calculations for bounding hexagon
        deltaNew = self.delta
        rNew = self.radius
        numPoints = self.n

        n = np.array([0, 1, 2, 3, 4, 5])

        x = rNew * np.sin(2 * np.pi * n / 6)
        y = rNew * np.cos(2 * np.pi * n / 6)

        pointsX = [np.linspace(x[i - 1], x[i], int(numPoints / 6), endpoint=False) for i in range(0, x.size)]
        pointsX = np.array(pointsX).flatten()
        pointsY = [np.linspace(y[i - 1], y[i], int(numPoints / 6), endpoint=False) for i in range(0, y.size)]
        pointsY = np.array(pointsY).flatten()

        min_xA, min_yA = np.min(pointsX), np.min(pointsY)
        max_xA, max_yA = np.max(pointsX), np.max(pointsY)

        # For bounding stencil
        bounding_points_x = []
        bounding_points_y = []

        # B
        x_shift = 2 * (max_xA - centroid_xA) - self.cells[0].delta
        y_shift = 0
        self.duplicate_cell(self.cells[0], x_shift, y_shift, roll=int(numPoints / 2), theta_shift=np.pi, identifier='B')
        # C
        x_shift = max_xA - centroid_xA - 0.5 * self.cells[0].delta
        y_shift = 1.5 * np.max(pointsY) - self.cells[0].delta * np.sin(np.pi / 3)
        self.duplicate_cell(self.cells[0], x_shift, y_shift, roll=0, theta_shift=0, identifier='C')
        # D
        x_shift = max_xA - centroid_xA - 0.5 * self.cells[0].delta
        y_shift = - (1.5 * np.max(pointsY) - self.cells[0].delta * np.sin(np.pi / 3))
        self.duplicate_cell(self.cells[0], x_shift, y_shift, roll=0, theta_shift=0, identifier='D')
        # E
        x_shift = 0
        y_shift = 2 * (1.5 * np.max(pointsY) - self.cells[0].delta * np.sin(np.pi / 3))
        self.duplicate_cell(self.cells[0], x_shift, y_shift, roll=0, theta_shift=0, identifier='E')
        # Its at boundary
        bounding_points_x.extend(pointsX + 0)
        bounding_points_y.extend(pointsY + 2 * (1.5 * np.max(pointsY) - self.cells[0].delta * np.sin(np.pi / 3)))
        # F
        x_shift = 2 * (max_xA - centroid_xA) - self.cells[0].delta
        y_shift = 2 * (1.5 * np.max(pointsY) - self.cells[0].delta * np.sin(np.pi / 3))
        self.duplicate_cell(self.cells[0], x_shift, y_shift, roll=0, theta_shift=0, identifier='F')
        # Its at boundary
        bounding_points_x.extend(pointsX + 2 * (max_xA - centroid_xA) - self.cells[0].delta)
        bounding_points_y.extend(pointsY + 2 * (1.5 * np.max(pointsY) - self.cells[0].delta * np.sin(np.pi / 3)))
        # G
        x_shift = - (max_xA - centroid_xA - 0.5 * self.cells[0].delta)
        y_shift = 1.5 * np.max(pointsY) - self.cells[0].delta * np.sin(np.pi / 3)
        self.duplicate_cell(self.cells[0], x_shift, y_shift, roll=0, theta_shift=0, identifier='G')
        # For H
        x_shift = 3 * (max_xA - centroid_xA - 0.5 * self.cells[0].delta)
        y_shift = 1.5 * np.max(pointsY) - self.cells[0].delta * np.sin(np.pi / 3)
        self.duplicate_cell(self.cells[0], x_shift, y_shift, roll=0, theta_shift=0, identifier='H')
        # Its at boundary
        bounding_points_x.extend(pointsX + 3 * (max_xA - centroid_xA - 0.5 * self.cells[0].delta))
        bounding_points_y.extend(pointsY + 1.5 * np.max(pointsY) - self.cells[0].delta * np.sin(np.pi / 3))
        # For I
        x_shift = - 2 * (max_xA - centroid_xA - 0.5 * self.cells[0].delta)
        y_shift = 0
        self.duplicate_cell(self.cells[0], x_shift, y_shift, roll=0, theta_shift=0, identifier='I')
        # For J
        x_shift = 4 * (max_xA - centroid_xA - 0.5 * self.cells[0].delta)
        y_shift = 0
        self.duplicate_cell(self.cells[0], x_shift, y_shift, roll=0, theta_shift=0, identifier='J')
        # Its at boundary
        bounding_points_x.extend(pointsX + 4 * (max_xA - centroid_xA - 0.5 * self.cells[0].delta))
        bounding_points_y.extend(pointsY + 0)
        # For K
        x_shift = - (max_xA - centroid_xA - 0.5 * self.cells[0].delta)
        y_shift = - (1.5 * np.max(pointsY) - self.cells[0].delta * np.sin(np.pi / 3))
        self.duplicate_cell(self.cells[0], x_shift, y_shift, roll=0, theta_shift=0, identifier='K')
        ### For L ###
        x_shift = 3 * (max_xA - centroid_xA - 0.5 * self.cells[0].delta)
        y_shift = - (1.5 * np.max(pointsY) - self.cells[0].delta * np.sin(np.pi / 3))
        self.duplicate_cell(self.cells[0], x_shift, y_shift, roll=0, theta_shift=0, identifier='L')
        # Its at boundary
        bounding_points_x.extend(pointsX + 3 * (max_xA - centroid_xA - 0.5 * self.cells[0].delta))
        bounding_points_y.extend(pointsY - (1.5 * np.max(pointsY) - self.cells[0].delta * np.sin(np.pi / 3)))
        ### For M ###
        x_shift = 0
        y_shift = - 2 * (1.5 * np.max(pointsY) - self.cells[0].delta * np.sin(np.pi / 3))
        self.duplicate_cell(self.cells[0], x_shift, y_shift, roll=0, theta_shift=0, identifier='M')
        # Its at boundary
        bounding_points_x.extend(pointsX + 0)
        bounding_points_y.extend(pointsY - 2 * (1.5 * np.max(pointsY) - self.cells[0].delta * np.sin(np.pi / 3)))
        ### For N ###
        x_shift = 2 * (max_xA - centroid_xA) - self.cells[0].delta
        y_shift = - 2 * (1.5 * np.max(pointsY) - self.cells[0].delta * np.sin(np.pi / 3))
        self.duplicate_cell(self.cells[0], x_shift, y_shift, roll=0, theta_shift=0, identifier='N')
        # Its at boundary
        bounding_points_x.extend(pointsX + 2 * (max_xA - centroid_xA) - self.cells[0].delta)
        bounding_points_y.extend(pointsY - 2 * (1.5 * np.max(pointsY) - self.cells[0].delta * np.sin(np.pi / 3)))
        ### For O ###
        x_shift = - 2 * (max_xA - centroid_xA - 0.5 * self.cells[0].delta)
        y_shift = 2 * (1.5 * np.max(pointsY) - self.cells[0].delta * np.sin(np.pi / 3))
        self.duplicate_cell(self.cells[0], x_shift, y_shift, roll=0, theta_shift=0, identifier='O')
        # Its at boundary
        bounding_points_x.extend(pointsX - 2 * (max_xA - centroid_xA - 0.5 * self.cells[0].delta))
        bounding_points_y.extend(pointsY + 2 * (1.5 * np.max(pointsY) - self.cells[0].delta * np.sin(np.pi / 3)))
        ### For P ###
        x_shift = - 3 * (max_xA - centroid_xA - 0.5 * self.cells[0].delta)
        y_shift = 1.5 * np.max(pointsY) - self.cells[0].delta * np.sin(np.pi / 3)
        self.duplicate_cell(self.cells[0], x_shift, y_shift, roll=0, theta_shift=0, identifier='P')
        # Its at boundary
        bounding_points_x.extend(pointsX - 3 * (max_xA - centroid_xA - 0.5 * self.cells[0].delta))
        bounding_points_y.extend(pointsY + 1.5 * np.max(pointsY) - self.cells[0].delta * np.sin(np.pi / 3))
        ### For Q ###
        x_shift = - 4 * (max_xA - centroid_xA - 0.5 * self.cells[0].delta)
        y_shift = 0
        self.duplicate_cell(self.cells[0], x_shift, y_shift, roll=0, theta_shift=0, identifier='Q')
        # Its at boundary
        bounding_points_x.extend(pointsX - 4 * (max_xA - centroid_xA - 0.5 * self.cells[0].delta))
        bounding_points_y.extend(pointsY + 0)
        ### For R ###
        x_shift = - 3 * (max_xA - centroid_xA - 0.5 * self.cells[0].delta)
        y_shift = - (1.5 * np.max(pointsY) - self.cells[0].delta * np.sin(np.pi / 3))
        self.duplicate_cell(self.cells[0], x_shift, y_shift, roll=0, theta_shift=0, identifier='R')
        # Its at boundary
        bounding_points_x.extend(pointsX - 3 * (max_xA - centroid_xA - 0.5 * self.cells[0].delta))
        bounding_points_y.extend(pointsY - (1.5 * np.max(pointsY) - self.cells[0].delta * np.sin(np.pi / 3)))
        ### For S ###
        x_shift = - 2 * (max_xA - centroid_xA - 0.5 * self.cells[0].delta)
        y_shift = - 2 * (1.5 * np.max(pointsY) - self.cells[0].delta * np.sin(np.pi / 3))
        self.duplicate_cell(self.cells[0], x_shift, y_shift, roll=0, theta_shift=0, identifier='S')
        # Its at boundary
        bounding_points_x.extend(pointsX - 2 * (max_xA - centroid_xA - 0.5 * self.cells[0].delta))
        bounding_points_y.extend(pointsY - 2 * (1.5 * np.max(pointsY) - self.cells[0].delta * np.sin(np.pi / 3)))

        # Prestrain dict
        p_dict = {'A': 1.0, 'B': 1.0, 'C': 1.0, 'D': 1.0, 'E': 1.0, 'F': 1.0, 'G': 1.0, 'H': 1.0, 'I': 1.0, 'J': 1.0,
                  'K': 1.0, 'L': 1.0, 'M': 1.0, 'N': 1.0, 'O': 1.0, 'P': 1.0, 'Q': 1.0, 'R': 1.0, 'S': 1.0,
                  'boundary': 1, 'none': 1}
        for cell in self.cells:
            cell.prestrain_dict = p_dict.copy()
            cell.adhesion_density_dict = p_dict.copy()

        ###### Update the boundary #######
        all_points = np.dstack((bounding_points_x, bounding_points_y))[0]
        # This keeps on the hexagon boundaries
        hull = ConcaveHull()
        hull.loadpoints(all_points)
        hull.calculatehull(tol=self.concave_hull_tolerance * 1)
        # Scale it
        scaled_boundary = scale(hull.boundary.exterior, xfact=1, yfact=1, zfact=1, origin='centroid')
        x, y = scaled_boundary.xy

        # Save
        self.boundary_adhesions = np.array([x, y])

        self.update_adhesion_points_between_all_cortices()
        self.reference_boundary_adhesions = []

    def _get_convex_hull_area_of_centroids_with_radii(self, centroids, radii):
        """Helper method for initialising a disordered tissue. Calculates the convex area of a cluster of round cells,
        which is used to make sure they are tightly packed.

        :param centroids: A list of centroids (x,y) coords for each cell.
        :type centroids: list
        :param radii:  A list of cell radii corresponding to those coords.
        :type radii: list

        """

        # Get boundary coords
        boundary_coords = np.dstack((self.boundary_adhesions[0], self.boundary_adhesions[1]))[0]
        # Get centroids
        centroids = np.dstack((centroids[::2], centroids[1::2]))[0]

        overlap_penalty, circ_areas = 0, 0
        points = []
        angles = np.linspace(0, 2 * np.pi, 1000)
        for i, c in enumerate(centroids):
            # Radius of current cell
            radius = radii[i]

            # Get other cell centroids
            other_cs = np.array([centroids[idx] for idx in range(len(centroids)) if idx != i])
            # Other radii
            other_radii = [radii[idx] for idx in range(len(centroids)) if idx != i]
            # Distance to them
            dists = distance.cdist([c], other_cs)
            for d_idx, d in enumerate(dists[0]):
                radius_2 = other_radii[d_idx]
                if d < radius + radius_2:
                    overlap_penalty += ((radius + radius_2) - d) * 1

            # Close to boundary
            min_dist = distance.cdist([c], boundary_coords).min()
            if min_dist < radius:
                overlap_penalty += (radius - min_dist) * 10

            x, y = c[0] + radius * np.cos(angles), c[1] + radius * np.cos(angles)

            circ_areas += np.pi * radius * radius

            points.extend(np.dstack((x, y))[0])

        points = np.array(points)
        hull = ConvexHull(points)
        total_area = overlap_penalty + hull.area - circ_areas

        return total_area

    def _get_distances_between_centroids_with_radii(self, centroids, radii):
        """Helper method for initialising a disordered tissue. Calculates the distances between cell centroids,
        which is used to make sure they are tightly packed.

        :param centroids: A list of centroids (x,y) coords for each cell.
        :type centroids: list
        :param radii:  A list of cell radii corresponding to those coords.
        :type radii: list

        """

        # Get boundary coords
        boundary_coords = np.dstack((self.boundary_adhesions[0], self.boundary_adhesions[1]))[0]
        # Get centroids
        centroids = np.dstack((centroids[::2], centroids[1::2]))[0]

        cumulative_distances = 0
        for i, c in enumerate(centroids):
            # Radius of current cell
            radius = radii[i]

            # Get other cell centroids
            other_cs = np.array([centroids[idx] for idx in range(len(centroids)) if idx != i])
            # Other radii
            other_radii = [radii[idx] for idx in range(len(centroids)) if idx != i]
            # Distance to them
            dists = distance.cdist([c], other_cs)
            for d_idx, d in enumerate(dists[0]):
                radius_2 = other_radii[d_idx]
                if d < radius + radius_2:
                    scaling = len(radii) * 2
                    cumulative_distances += ((radius + radius_2) - d) * scaling
                else:
                    cumulative_distances += (d - (radius + radius_2))

            # Close to boundary
            min_dist = distance.cdist([c], boundary_coords).min()
            if min_dist < radius:
                cumulative_distances += (radius - min_dist) * 1

        return cumulative_distances

    def _pack_centroids_together(self, centroids, radii=None):
        """Given a list of centroids, define radii based on their relative distances and pack together more densely via minimisation

        :param centroids:  List of cell centroid (xy) coords.
        :type centroids: list
        :param radii:  (Default value = [])  Optional list of radii for those cells.  Otherwise its calculated from centroids.
        :type radii: list
        :return centroids:  The new centroid locations after packing.
        :rtype: list

        """

        self.verboseprint("Packing the centroids together", object(), 1)

        radii = [] if radii is None else radii

        boundary_coords = np.dstack((self.boundary_adhesions[0], self.boundary_adhesions[1]))[0]

        # centroids = [[59.19914007469464, -26.433803996836982], [-1.534100787501092, 59.078695551287375], [-52.06592950911108, -10.362478064085945], [32.84152340879301, -93.37408338671601], [73.25750764601091, 43.442251003972224], [-37.463652031344864, -79.68420967046461], [-72.33201585627047, 55.24755184974827], [45.010659331709796, 95.68920399831025], [6.435527624855871, -0.19363773362153852], [-39.89568913568516, 101.37354092790487], [-83.37871347024672, -51.64724193213894], [4.52062124513836, -50.83679363376616]]

        # Get the radii of the nearest cells
        if len(radii) != len(centroids):
            radii = []
            for centroid in centroids:
                other_cells = np.array([c for c in centroids if c[0] != centroid[0] and c[1] != centroid[1]])
                dist_to_cells = 0.5 * distance.cdist(np.array([centroid]), other_cells).min()
                dist_to_boundary = distance.cdist(np.array([centroid]), boundary_coords).min()
                # Get closest point
                radii.append(min([dist_to_boundary, dist_to_cells]) - 0.5)

        # Past to minimiser
        centroids = np.array(centroids)
        # centroids = minimize(self._get_distances_between_centroids_with_radii, centroids, args=radii,
        #                      method='Nelder-Mead').x
        centroids = minimize(self._get_distances_between_centroids_with_radii, centroids, args=radii,
                             method='Powell').x
        # centroids = minimize(self._get_convex_hull_area_of_centroids_with_radii, centroids, args=radii,
        #                      method='Powell').x
        # centroids = minimize(self._get_convex_hull_area_of_centroids_with_radii, centroids, args=radii,
        #                      method='Nelder-Mead').x  # , options={'fatol': 1e-10, 'xatol': 1e-10}).x
        centroids = np.dstack((centroids[::2], centroids[1::2]))[0]

        return centroids

    def create_disordered_tissue_in_ellipse(self, param_dict=None, **kwargs):
        """Creates a disordered tissue using a matern point process.

        :param param_dict:  (Default value = None)  kappa, omega0, delta dictionary for cells.
        :type param_dict: dict
        :param kwargs: Optional arguments for the tissue.
        :type kwargs: dict

        """

        self.verboseprint("Creating disordered eptm", object(), 1)

        max_density = kwargs.get('max_density', 15000)
        ellipse_minor = kwargs.get('ellipse_minor', 120)
        ellipse_major = kwargs.get('ellipse_major', 140)
        min_radius = kwargs.get('min_radius', 30)
        max_radius = kwargs.get('max_radius', 35)

        angles = np.linspace(0, 2 * np.pi, 4090)
        x, y = ellipse_minor * np.cos(angles), ellipse_major * np.sin(angles)
        self.boundary_adhesions = np.array([x, y])
        boundary_coords = np.dstack((x, y))[0]

        density, temp_max_radius = 0, max_radius + 0
        angles = np.linspace(0, 2 * np.pi, self.n)
        centroids, last_num_centroids = [], 0
        done = False
        while not done:
            while density < max_density:
                # Get random point in ellipse
                random_guess, random_angle = np.random.uniform(0, 1, 1)[0], np.random.uniform(0, 2 * np.pi, 1)[0]
                c_x = np.sqrt(random_guess) * np.cos(random_angle)
                c_y = np.sqrt(random_guess) * np.sin(random_angle)
                c_x *= ellipse_minor
                c_y *= ellipse_major

                c_guess = np.array([[c_x, c_y]])
                if distance.cdist(c_guess, boundary_coords).min() > temp_max_radius:
                    if len(centroids) == 0:
                        centroids.append([c_x, c_y])
                    elif distance.cdist(c_guess, np.array(centroids)).min() > temp_max_radius * 2:
                        centroids.append([c_x, c_y])
                        density = 0

                if density == max_density - 1:
                    temp_max_radius -= 2
                    density = 0
                if temp_max_radius < min_radius:
                    density = max_density

                density += 1

            if last_num_centroids < 16:
                centroids = list(self._pack_centroids_together(centroids))

            if last_num_centroids == len(centroids) or len(centroids) > 16:
                done = True
            min_radius = min_radius - 3 if min_radius > max_radius / 3 else min_radius
            last_num_centroids = len(centroids)
            temp_max_radius = max_radius + 0.
            density = 0

        self.cellDict = dict([])

        # And rest of the cells:
        self.cells = []
        alphabet = itertools.cycle('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        for centroid in centroids:
            # All points excluding this one
            other_cells = np.array([c for c in centroids if c[0] != centroid[0] and c[1] != centroid[1]])
            dist_to_cells = 0.5 * distance.cdist(np.array([centroid]), other_cells).min()
            dist_to_boundary = distance.cdist(np.array([centroid]), boundary_coords).min()
            # Get closest point
            radius = min([dist_to_boundary, dist_to_cells]) - 0.5
            x, y = centroid[0] + radius * np.cos(angles), centroid[1] + radius * np.sin(angles)
            ds = (2 * radius * np.pi) / self.n
            # Do the curvatures
            c = np.gradient(angles, ds, edge_order=2)
            # Do the curvatures derivatives
            d = np.gradient(c, ds, edge_order=2)  # np.linspace(0, 0, total_points)
            # TAUS
            gamma = np.ones(angles.size)

            # The next id, but making sure it's unique
            next_id = next(alphabet)
            while next_id in self.cellDict.keys():
                next_id = ''.join([next_id, next(alphabet)])

            # We just build the first cell and then create all other instances from this cell.
            initial_guessesA = {'D': d, 'C': c, 'gamma': gamma, 'theta': angles, 'x': x, 'y': y}
            cell = Cell(initial_guesses=initial_guessesA, verbose=0, param_dict=param_dict, within_hexagon=False,
                        identifier=next_id)

            self.cells.append(cell)
            self.cellDict[next_id] = cell

        # Prestrain and adhesion density dicts
        # Dictionary for prestrain
        prestrain_dict = {key: 1 for key in self.cellDict.keys()}
        prestrain_dict['boundary'] = 1
        prestrain_dict['none'] = 1
        # Scaling for omega0 for different cortices e.g. when adhesion less in A-B than A-C.
        adhesion_density_dict = {key: 1 for key in self.cellDict.keys()}
        adhesion_density_dict['boundary'] = 1
        adhesion_density_dict['none'] = 0
        for cell in self.cells:
            cell.prestrain_dict = copy.deepcopy(prestrain_dict)
            cell.adhesion_density_dict = copy.deepcopy(adhesion_density_dict)

        # Create the boundary cell
        initial_guesses = {'D': [], 'C': [], 'gamma': [], 'theta': [],
                           'x': self.boundary_adhesions[0], 'y': self.boundary_adhesions[1]}
        self.boundary_cell = Cell(initial_guesses=initial_guesses, identifier='boundary')
        self.boundary_cell.prune_adhesion_data()
        self.cellDict['boundary'] = self.boundary_cell

    def pickle_self(self, SAVE_DIR=None, name=None, prune_adhesions=True):
        """Pickles and saves an instance of this class in its current state.

        :param SAVE_DIR: (Default value = None)  Save location.
        :type SAVE_DIR: string
        :param name:  (Default value = None)  Filename
        :type name: string
        :param prune_adhesions: (Default value = True)  Remove fast adhesions and cell-stored adhesions before saving (recommended for space).
        :type prune_adhesions : bool

        """

        self.verboseprint("Saving T1 objects", object(), 1)

        if prune_adhesions:
            for cell in self.cells:
                cell.prune_adhesion_data()
            if self.boundary_bc != 'elastic':
                self.reference_boundary_adhesions = []

        if SAVE_DIR == None:
            SAVE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'pickled_tissues'))
        # Make sure the directory exists
        if not os.path.exists(SAVE_DIR):
            os.makedirs(SAVE_DIR)
        # Filename
        if name == None:
            name = 'T1_omega0_' + str(self.cells[0].omega0) + '_' + '_pressure_' \
                   + str(self.cells[0].pressure)

        # saveloc
        saveloc = SAVE_DIR + '/' + name
        # Pickle
        with open(saveloc, 'wb') as s:
            dill.dump(self, s)

    def plot_bijunction_tension_arrows(self, cell_list, ax=None, num_extra_indices=2, arrow_col='k',
                                       arrow_scale=.015, arrow_width=0.004):
        """Plot arrows representing magnitude of tension in cortex

        :param cell_list:  List of cell idetifiers on which arrows will be plotted.
        :type cell_list: list
        :param ax:  (Default value = None)  Axis on which to plot.
        :type ax: mpl axis
        :param num_extra_indices:  (Default value = 2)  Average the force over a few indices.
        :type num_extra_indices: int
        :param arrow_scale:  (Default value = .015)  Scale the matplotlib arrow.
        :type arrow_scale: float
        :param arrow_col:  (Default value = 'k')  Arrow colour.
        :type arrow_col: string
        :param arrow_width:  (Default value = 0.004)  Width of arrow.
        :type arrow_width: float

        """

        if ax is None:
            fig, ax = plt.subplots()

        for cell_id in cell_list:
            cell = self.cellDict[cell_id]
            forces_t = cell.gamma - 1
            for bi_junc in self.get_bicellular_junctions_for_cell(cell.identifier):
                theta = np.arctan2((cell.y[bi_junc[1]] - cell.y[bi_junc[0]]), (cell.x[bi_junc[1]] - cell.x[bi_junc[0]]))
                for j_end in bi_junc:
                    x, y = cell.x[j_end], cell.y[j_end]
                    theta = theta - np.pi if j_end == bi_junc[1] else theta
                    force = np.mean(forces_t[j_end - num_extra_indices:j_end + num_extra_indices]) \
                            * np.array([np.cos(theta), np.sin(theta)])
                    ax.quiver(x, y, force[0], force[1], width=arrow_width, scale=arrow_scale, color=arrow_col,
                              zorder=10)

    def plot_self(self, ax=None, axEqual=True, plotAdhesion=True, plot_stress=False, plot_shape=False,
                  plot_adhesion_forces=True, plot_boundary=True, cell_ids=None, lagrangian_tracking=False,
                  plot_tension=False, plot_boundary_movement=True, plot_cbar=True, sim_type='auto', cell_kwargs=None):
        """Plot the tissue and the boundary.

        :param ax: (Default value = None)  Axis object to plot on.
        :type ax: mpl axis
        :param plotAdhesion: (Default value = True)  Whether to plot adhesions.
        :type plotAdhesion: bool
        :param plot_shape: (Default value = False)  Plot the principal axis of shape.
        :type plot_shape: bool
        :param plot_boundary: (Default value = True)  Plot the boundary?
        :type plot_boundary: bool
        :param lagrangian_tracking: (Default value = False)  Plot Lagrange markers.
        :type lagrangian_tracking: bool
        :param plot_boundary_movement: (Default value = True)  Plot the position of the boundary in the last step.
        :type plot_boundary_movement: bool
        :param cell_kwargs: (Default value = {})  Optional arguments for plotting the cells.
        :type cell_kwargs: dict
        :param axEqual:  (Default value = True)  As in Matplotlib axis.
        :type axEqual: bool
        :param plot_stress:  (Default value = False)  Plot the principal axis of stress.
        :type plot_stress: bool
        :param plot_adhesion_forces:  (Default value = True)  Plot arrows for the adhesion forces.
        :type plot_adhesion_forces: bool
        :param cell_ids:  (Default value = [])  List of cell identifiers to plot.
        :type cell_ids: list
        :param plot_tension:  (Default value = False)  Plot the magnitude of tension in cortex with a heatmap.
        :type plot_tension: bool
        :param plot_cbar:  (Default value = True)  Plot colourbars for the cell stress and tension.
        :type plot_cbar: bool
        :param sim_type:  (Default value = 'auto')  Used to scale the stress colourmaps e.g. if there are lots of cables.
        :type sim_type: string

        """

        self.verboseprint("Plotting full T1", object(), 1)

        cell_ids = [] if cell_ids is None else cell_ids
        cell_kwargs = {} if cell_kwargs is None else cell_kwargs

        # Figure out what type of simulation we did
        if sim_type == 'auto':
            nonzero_pressures = [c.pressure != 0 for c in self.cells]
            if any(nonzero_pressures):
                sim_type = 'medial'
            else:
                max_prestrain_refs = max([len([pval for pval in c.prestrain_dict.values() if pval < 0.9995])
                                          for c in self.cells])
                if max_prestrain_refs < 2:
                    sim_type = 'single'
                elif max_prestrain_refs < len(self.cells):
                    sim_type = 'cable'
                else:
                    sim_type = 'whole'

        if len(cell_ids) == 0:
            cell_ids = [c.identifier for c in self.cells]

        if ax is None:
            f, ax = plt.subplots()

        # Check if pressure or prestrain has been added to any cell
        plot_pressure = any([c.pressure != 0 for c in self.cells])

        # Colourbar for stress
        plot_cbar = len(plt.gcf().axes) < 3 if plot_cbar else False
        if plot_stress and plot_cbar:
            if sim_type == 'single':
                max_pressure = 6e-4  # For singe junction
            elif sim_type == 'cable':
                max_pressure = 1.2e-3 / 1  # For cables
            elif sim_type == 'whole':
                max_pressure = 4e-3  # For whole cells
            elif sim_type == 'medial':
                max_pressure = 2e-3
            else:
                raise NotImplementedError

            # Truncate the colourmap to use lighter colours
            # lower, upper = 5e-4, -5e-4
            lower, upper = max_pressure * (5. / 6.), -max_pressure * (5. / 6.)
            minColor = 0.5 - 0.5 * (upper / max_pressure)
            maxColor = 0.5 + 0.5 * (upper / max_pressure)
            truncated_map = self.truncate_colormap(plt.get_cmap("RdBu"), minColor, maxColor)
            # Normalise about new points
            norm = matplotlib.colors.Normalize(lower, upper)
            # Create scaled cmap
            scaled_cmap = matplotlib.cm.ScalarMappable(norm=norm, cmap=truncated_map)

            cbar = plt.colorbar(scaled_cmap, ax=[ax], format=self.OOMFormatter(-4, mathText=False), location="right",
                                shrink=0.5, pad=0.02)
            cbar.set_label(r'$P^{\mathrm{cell}}$', rotation=270, size=30, labelpad=10)
            cbar.ax.tick_params(labelsize=26)
            # ticks = np.linspace(lower, upper, 9)
            # cbar.ax.set_yticklabels([str(np.format_float_positional(tick, precision=4, trim='-')) for tick in ticks])

            # Colourbar for tension
        if plot_tension and plot_cbar:
            max_tension = 0.0025

            # Truncate the colourmap to use lighter colours
            lower, upper = -max_tension, max_tension
            minColor = 0.5 - 0.5 * (upper / max_tension)
            maxColor = 0.5 + 0.5 * (upper / max_tension)
            truncated_map = self.truncate_colormap(plt.get_cmap("seismic"), minColor, maxColor)
            # Normalise about new points
            norm = matplotlib.colors.Normalize(lower, upper)
            # Create scaled cmap
            scaled_cmap = matplotlib.cm.ScalarMappable(norm=norm, cmap=truncated_map)
            cbar2 = plt.colorbar(scaled_cmap, ax=[ax], format=self.OOMFormatter(-3, mathText=False), location='left',
                                 shrink=.5, pad=.02)
            # cbar2 = plt.colorbar(scaled_cmap, orientation="vertical")
            cbar2.set_label(r'$\varepsilon$', labelpad=-7)
            cbar2.ax.tick_params(labelsize=26)
            # ticks = np.linspace(lower, upper, 9)
            # cbar2.ax.set_yticklabels([str(np.format_float_positional(tick, precision=4, trim='-')) for tick in ticks])

        if (plot_adhesion_forces or plot_tension or plot_stress) and self.cells[0].adhesion_point_coords == []:
            self.update_adhesion_points_between_all_cortices()

        for cell_ref in cell_ids:
            # Specify cell label
            cell_label = r'$%s$' % int(list(self.cellDict.keys()).index(cell_ref) + 1)
            # plot the cell
            self.cellDict[cell_ref].plot_self(ax=ax, equalAx=axEqual, plotAdhesion=plotAdhesion,
                                              plot_shape=plot_shape,
                                              plot_adhesion_forces=plot_adhesion_forces,
                                              plot_pressure=plot_pressure,
                                              plot_tension=plot_tension,
                                              lagrangian_tracking=lagrangian_tracking,
                                              label=cell_label,
                                              plot_stress=plot_stress,
                                              sim_type=sim_type,
                                              **cell_kwargs)

        if self.within_hexagons:
            for cell in self.cells:
                ax.plot(cell.adhesion_point_coords[:, 0], cell.adhesion_point_coords[:, 1], '-', ms=1, c='k')
        elif plot_boundary:
            ax.plot(self.boundary_adhesions[0], self.boundary_adhesions[1], '-', c='k')
            if self.boundary_bc == 'periodic':
                for c in self.cellDict.values():
                    if 'boundary' not in c.identifier and c.identifier != 'A':
                        c.plot_self(ax=ax, equalAx=axEqual, plotAdhesion=0, plot_shape=0,
                                    plot_adhesion_forces=0, plot_pressure=0, plot_tension=0,
                                    lagrangian_tracking=0, label=c.identifier, plot_stress=0,
                                    col='k')

        if axEqual:
            if len(self.cells) == 14:
                if min(self.cellDict['I'].x) < -160:
                    ax.set_xlim([-167, 275])
                    ax.set_ylim([-140, 140])
                elif min(self.cellDict['I'].x) < -100:
                    ax.set_xlim([-143, 232])
                    ax.set_ylim([-148, 148])
                else:
                    ax.set_xlim([-100, 160])
                    ax.set_ylim([-150, 150])
            elif len(self.cells) == 3:
                ax.set_xlim([-40, 100])
                ax.set_ylim([-45, 95])
            ax.set_aspect('equal', 'box')

        if plot_boundary_movement and self.boundary_bc not in ['fixed', 'periodic']:
            # a 4x4 box (counterclockwise)
            if len(self.reference_boundary_adhesions) == 0:
                initial_box_x = [-89.96192087, -89.96192087, 149.61295261, 149.61295261, -89.96192087]
                initial_box_y = [-138.26794919, 138.26794919, 138.26794919, -138.26794919, -138.26794919]
            else:
                min_x, max_x = np.min(self.reference_boundary_adhesions[0]), \
                               np.max(self.reference_boundary_adhesions[0])
                min_y, max_y = np.min(self.reference_boundary_adhesions[1]), \
                               np.max(self.reference_boundary_adhesions[1])
                initial_box_x = [min_x, min_x, max_x, max_x, min_x]
                initial_box_y = [min_y, max_y, max_y, min_y, min_y]

            # a 2x2 hole in the box (clockwise)
            current_min_xy = np.min(self.boundary_adhesions, axis=-1)
            current_max_xy = np.max(self.boundary_adhesions, axis=-1)
            current_box_x = [current_min_xy[0], current_min_xy[0], current_max_xy[0],
                             current_max_xy[0], current_min_xy[0]][::-1]
            current_box_y = [current_min_xy[1], current_max_xy[1], current_max_xy[1],
                             current_min_xy[1], current_min_xy[1]][::-1]

            # if you don't specify a color, you will see a seam
            # plt.fill(current_box_x + initial_box_x, current_box_y + initial_box_y, color='#a6a6a6')

            plt.plot(current_box_x, current_box_y, '-', color='k')
            plt.plot(initial_box_x, initial_box_y, '--', color='gray')

    class OOMFormatter(matplotlib.ticker.ScalarFormatter):
        """ """

        def __init__(self, order=0, fformat="%1.1f", offset=True, mathText=True):
            self.oom = order
            self.fformat = fformat
            matplotlib.ticker.ScalarFormatter.__init__(self, useOffset=offset, useMathText=mathText)

        def _set_order_of_magnitude(self):
            """ """
            self.orderOfMagnitude = self.oom

        def _set_format(self, vmin=None, vmax=None):
            """

            :param vmin:  (Default value = None)
            :param vmax:  (Default value = None)

            """
            self.format = self.fformat
            if self._useMathText:
                self.format = r'$\mathdefault{%s}$' % self.format

    @staticmethod
    def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=1000):
        """Truncate a colourmap between 2 values

        :param cmap: A matplotlib colourmap.
        :param cmap: mpl cmap
        :param minval:  (Default value = 0.0)  New max value for the colourmap.
        :param minval:  (Default value = 0.0)  float
        :param maxval:  (Default value = 1.0)  New min value for colourmap.
        :param maxval:  (Default value = 1.0)  float
        :param n:  (Default value = 1000)  Number of values in cmap.
        :param n:  (Default value = 1000)  int
        :return:  The truncaetd cmap
        :rtype:  mpl cmap

        """
        new_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
            'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
            cmap(np.linspace(minval, maxval, n)))
        return new_cmap

    @staticmethod
    def rotate_curve_about_line(x, y, m, c=None):
        """For plotting the symmetric junctions, rotate the soln
        m is gradient.

        :param x:  x-coords of the line.
        :type x: np.arrays
        :param y:  y-coords of the line.
        :type y: np.array
        :param m:  Gradient of line
        :type m:  float
        :param c:  (Default value = None)  y-intercept
        :type c:  float
        :return:  The new (x,y) arrays representing the line coordinates.
        :rtype:  tuple

        """
        if c is None:
            c = y[0] - m * x[0]
        d = (x + (y - c) * m) / (1 + m * m)
        x_ = 2 * d - x
        y_ = 2 * d * m - y + 2 * c

        return (x_, y_)

    @staticmethod
    def get_indices_to_order_anticlockwise(points, centroid=None):
        """Order a given set of points anticlockwise, centroid calculated from points if not given

        :param points:  Input array of (x,y) coord pairs.
        :type points: np.array
        :param centroid:  (Default value = None)  Centroid of points. Can also calculate on the fly.
        :type centroid:  tuple
        :return:  The indices that order the list.
        :rtype:  list

        """
        if not centroid:
            centroid = (np.mean(points[:, 0]), np.mean(points[:, 1]))
        # Order the points anticlockwise
        angles = np.arctan2(centroid[0] - points[:, 0], centroid[1] - points[:, 1])
        index_list = np.argsort(-angles)

        return index_list

    @staticmethod
    def build_strain_tensor(stretch: float = 0, shear: float = 1):

        strain = np.array([[1, 0], [0, 1]])

        # isotropic
        strain = strain + np.array([[stretch, 0], [0, stretch]])

        # Shear
        strain = strain + np.array([[0, 0.5 * shear], [0.5 * shear, 0]])
        # strain = strain + np.array([[0, shear], [0, 0]])
        # strain = strain + np.array([[0, 0], [shear, 0]])

        return strain

#
#
#
#
#
#
#
#
#
