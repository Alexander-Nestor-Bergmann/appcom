#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Author  : Alexander Nestor-Bergmann
# Released: 08/03/2021
# =============================================================================
"""Implementation of a class to represent the cell cortex."""

import colorsys
import dill
import itertools
import matplotlib
import numpy as np
import os
import shapely.geometry as geom

from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection

from collections import Counter
from numpy.core.umath_tests import inner1d
from scipy.integrate import solve_bvp as solve_bvp_scipy
from scipy.interpolate import interp1d
from scipy.signal import decimate, resample, savgol_filter
from scipy.spatial import cKDTree
from sklearn.neighbors import NearestNeighbors

matplotlib.rcParams.update({'font.size': 22})

CURRENT_DIR = os.path.dirname(__file__)


class Cell(object):
    """
    Class that represents the apical cell cortex and holds all variables and functions to manipulate it.

    The class stores cortex variables ``x``, ``y``, ``theta``, theta' (called ``c``), theta'' (called ``d``) and alpha (called ``gamma``, here).

 .. note:: Due to old naming conventions, some variables are different to the published names:
    (name here = published name),
    ``kappa`` = kappa^2,
    ``gamma`` = alpha,
    ``prestrain`` = prestretch,
    ``s`` = S_0.
    """

    def __init__(self, initial_guesses, identifier='A', verbose=True, param_dict=None, **cell_kwargs):
        """
        Initialiser to set up the class properties.

        :param initial_guesses:  A dictionary, with keys [``x``, ``y``, ``theta``, ``c``, ``d``, ``gamma``], of the
        initial guesses for the cortex variables.
        :type initial_guesses:  dict
        :param identifier:  A unique identifier for the cell.
        :type identifier:  string
        :param verbose:  Whether to print information on the simulation to the console.
        :type verbose:  bool
        :param param_dict:  Dictionary of the mechanical parameters for the cell, ``delta``, ``kappa``, ``omega``.
        :type param_dict:  dict
        :param cell_kwargs:  Additional arguments for the cell class (see below).
        :type cell_kwargs: dict

        """

        # Give the cell its unique id
        self.identifier = identifier

        self.verbose = verbose
        self.verboseprint("Creating cell %s" % identifier, object(), 1)

        ############ Paramter and cortex variable initialisation

        # If no params given, set some default ones
        if param_dict == None:
            kappa, delta, omega0 = .0001, 1, 0
        # Else read.
        else:
            kappa = param_dict['kappa']
            delta = param_dict['delta']
            omega0 = param_dict['omega0']
        # Store params:
        self.kappa = kappa  # (is kappa^2 in manuscript)
        self.delta = delta
        self.omega0 = omega0

        # Initial guesses for cortex variables
        self.D = initial_guesses['D']  # D = C'
        self.C = initial_guesses['C']  # C = theta'
        self.gamma = initial_guesses['gamma']
        self.theta = initial_guesses['theta']
        self.x = initial_guesses['x']
        self.y = initial_guesses['y']

        # Number of cortex nodes
        self.n = self.x.size
        # If solving requires new nodes to be put in, set the maximum.
        self.decimateAbove = self.n * 1.1
        # Adaptive mesh parameters
        self.max_mesh_coarsen_fraction = cell_kwargs.get('max_mesh_coarsen_fraction', 0.01)  # What proportion of mesh nodes can be removed in a timestep

        ############ Geometry ##############

        # Cortex rest length set to initial length
        self.rest_len = self.get_length()
        # Set the Lagrangian reference, S_0, domain from the current configuration and stretch/prestretch
        self.domain = (0, self.rest_len)

        # Area term: \todo
        self.area_stiffness = cell_kwargs.get('area_stiffness', 0)
        self.pref_area = self.get_area()

        # S_0 (Lagrangian reference material coord in undeformed configuration)
        self.s = np.linspace(0, self.rest_len, self.n)
        # Tracking of Lagrangian material points.  Each point is given an identifier, so we can find it when s changes.
        self.lagrangian_point_ids = np.arange(self.s.size)

        ############ Mechanics ##############

        # Consitutive model for cortex stress-strain relationship \todo can use only 'linear' atm.
        self.possible_constitutive_models = ["linear", "hyperelastic"]
        self.constitutive_model = cell_kwargs.get('constitutive_model', 'linear')

        # Prestretch applied via 'identity sensing' adhesions.  Specify rule for force calculation.
        self.prestrain_type = cell_kwargs.get('prestrain_type', 'min')  # Can be "most_common", "average", "min", "nearest"
        # Dictionary for prestretch (referred to as prestrain)
        self.prestrain_dict = {'A': 1.0, 'B': 1.0, 'C': 1.0, 'D': 1.0, 'E': 1.0, 'F': 1.0, 'G': 1.0, 'H': 1.0,
                               'I': 1.0, 'J': 1.0, 'K': 1.0, 'L': 1.0, 'M': 1.0, 'N': 1.0, 'boundary': 1, 'none': 1}
        # Initialse prestretches to 1
        self.prestrains = np.ones_like(self.x)
        self.prestrain_indices_list = []
        # For identity sensing, how far can the adheions 'look' to find other cortices
        self.adhesion_search_radius = self.domain[1] / 20

        # Cell pressure forces from medial myosin
        self.pressure = cell_kwargs.get('pressure', 0)
        self.pressure_on_off = cell_kwargs.get('pressure_on_off', True)  # Whether to bother looking for medial pressures. Generally, just use True.

        # Protrusion forces
        self.protrusion_target = cell_kwargs.get('protrusion_target', None)  # Would be the unique identifier for another cell
        # And the magnitude of the force
        self.protrusion_force = cell_kwargs.get('protrusion_force', 0.)

        # Tolerance for solving the BVP using Scipy
        self.relax_tol = cell_kwargs.get('relax_tol', 1e0)
        # Store if solving was successful or not to see if we can trust configuration
        self.last_solve_status = cell_kwargs.get('last_solve_status', 0)

        ############## Adhesion properties ##############

        ##### Fast adhesions

        # Default off
        self.fast_adhesions_active = cell_kwargs.get('fast_adhesions_active', False)

        # Their constitutive properties
        self.adhesion_force_law = cell_kwargs.get('adhesion_force_law', 'spring')  # Can be 'spring' only, for now.
        # Set a maximum size for adhesions (force=0 for adhesions longer than this)
        self.max_adhesion_length = self.domain[1] / 20
        # How is the force calculated for fast adhesions: can be in ["nearest", "meanfield", "fixed_radius"]
        # 'meanfield' is most appropriate for fast adhesions.
        self.adhesion_type = cell_kwargs.get('adhesion_type', 'fixed_radius')
        # How many adhesions to use in the force calculation. Doesn't need to be more than 20, smaller is faster.
        self.max_num_adhesions_for_force = cell_kwargs.get('max_num_adhesions_for_force', 20)
        # How forces are scaled based on length of adhesion molecule.
        self.adhesion_beta_scale_factor = cell_kwargs.get('adhesion_beta_scale_factor', 50)

        # Scaling for omega to modulute adhesion density on certain junctions
        self.adhesion_density_dict = {'A': 1.0, 'B': 1.0, 'C': 1.0, 'D': 1.0, 'E': 1.0, 'F': 1.0, 'G': 1.0, 'H': 1.0,
                                      'I': 1.0, 'J': 1.0, 'K': 1.0, 'L': 1.0, 'M': 1.0, 'N': 1.0, 'boundary': 1,
                                      'none': 1}

        # This is where information about nearby cortices (within adhesion_search_radius) is held, which is what fast
        # adhesions could possibly couple to.   The information is passed from the eptm class
        self.adhesion_point_coords = []  # (x,y) coords of nearby neighbouring cells
        self.adhesion_point_identifiers = []  # identifier of cell for the (x,y) coords above.
        self.adhesion_point_spacings = []  # Cortex spacing (in current configuration) of neighbouring nodes

        # Containers for the actual adhesion connections that are made. Each idx represents the data for the cortex
        # idx. So self.adhesion_distances[idx] gives the lengths of the adhesion bonds at self.s[idx]
        # We store this information so it doesn't need to be calculated on the fly
        # Total force from adhesion at every node
        self.adhesion_forces = np.zeros(self.x.size)
        # Lengths of adhesion connections that have been made.
        self.adhesion_distances = []
        # The (x,y) coords of connections from self.adhesion_point_coords that are actually made.
        self.adhesion_connections = []
        # The ids of the connections
        self.adhesion_connections_identities = []
        # A list of the cells that each node is connected to (the unique list of all connections)
        self.adhesion_connections_identities_unique = []
        # Spacings of the connections that were made.
        self.adhesion_connections_spacings = []

        ########### SDK and slow adhesions.

        # We need to keep these lists around when saving.  The lists are populated Adhesion class objects, which
        # contain information about which cortex nodes are connected to where.

        # Sidekick (vertex-specific)
        self.sidekick_adhesions = []
        self.sdk_stiffness = cell_kwargs.get('sdk_stiffness', 100 * self.omega0)
        self.sdk_restlen = cell_kwargs.get('sdk_restlen', 1)

        # Slow adhesions (when tau_ad >= 1).  Need to keep a perma
        self.slow_adhesions = []

        ############## Misc ##############

        # How close to plot prestrain next to cortex
        self.prestrain_plot_offset = cell_kwargs.get('prestrain_plot_offset', 0.9)


    def verboseprint(self, *args):
        """Function to print out details as code runs

        :param args:  Information to be printed to console.

        """
        try:
            self.verbose
        except AttributeError:
            self.verbose = True
        print(args) if self.verbose else None


    def prune_adhesion_data(self):
        """ Remove all locally stored adhesion data.  This data is usually permanently held by the Epithelium class.
        """

        self.adhesion_point_coords = []
        self.adhesion_point_identifiers = []
        self.adhesion_point_spacings = []
        self.adhesion_forces = []
        self.adhesion_distances = []
        self.adhesion_connections = []
        self.adhesion_connections_identities = []
        self.adhesion_connections_identities_unique = []
        self.adhesion_connections_spacings = []
        self.sidekick_adhesions = []
        self.slow_adhesions = []

        self.prestrains = []

        self.adhesion_tree = None
        self.adhesion_polygon = None


    def update_reference_configuration_to_current(self):
        """Update the Lagrangian coordinate S_0 to the current configuration from the stretches and prestretches:
         S_0 <- s = S_0 * stretch * prestretch

        """

        # Calculate the prestretches
        prestrains = self.get_prestrains()
        # Current undeformed segment lengths
        ds = np.diff(self.s)
        # Update S_0
        for idx in range(self.s.size - 1):
            self.s[idx + 1] = self.gamma[idx] * prestrains[idx] * ds[idx] + self.s[idx]

        # Reset the rest length and integration domain
        self.rest_len = self.s[-1] - self.s[0]
        self.domain = (self.s[0], self.s[-1])

        # self.get_mesh_spacing()
        self.update_deformed_mesh_spacing()


    def update_adhesion_points(self, points, ids, spacing):
        """Store points that fast adhesions can adhere to.

        :param points:  A list of (x,y) coordinates, which the cell cortex xan adhere to.
        :type points: list
        :param ids:  A list of the cell identifiers that those adhesion points came from.
        :type ids:  list
        :param spacing:  A list of the discretised spacing (on the other cortices) for those adhesion points.
        :type spacing: list

        """
        # Store the nodes
        self.adhesion_point_coords = points

        # Store these identifiers
        self.adhesion_point_identifiers = ids

        # Store the spacings
        self.adhesion_point_spacings = spacing


    def clear_adhesion_points(self):
        """Remove all stored possible adhesion points.
        """
        self.adhesion_point_coords = []
        self.adhesion_point_identifiers = []
        self.adhesion_point_spacings = []


    def get_neighbours(self):
        """Get the identities of neighbouring cells
        """

        neighbours = set(itertools.chain(*self.adhesion_connections_identities))
        return list(neighbours)


    def get_intersection_indices(self, points):
        """ Get at list of the idxs, from given points, that intersect adhesion boundaries. DEPRECIATED

        :param points:  The coords of points to check
        :type points: list
        :return indices: The indices of the points that intersected the boundary.

        """

        # Build the boundary polygon, if it doesn't exist
        if self.adhesion_polygon is None:
            self.build_adhesion_tree()
        try:
            intersect_points = self.adhesion_polygon.intersection(geom.Polygon(points))
            intersect_points_coords = np.dstack(intersect_points.exterior.coords.xy)[0]
        except:

            return [i for i in range(0, len(points)) if
                    not self.adhesion_polygon.contains(geom.Point(points[i, 0], points[i, 1]))]

        indices = [i for i in range(0, points.shape[0]) if not points[i] in intersect_points_coords]

        return indices


    def check_if_cell_intersects_adhesion_boundary(self, x=None, y=None):
        """Check if any cortex node intersects the adhesion boundary

        :param x:  (Default value = None) x coords for cell cortex points.  Can default to use stored values.
        :type x:  list
        :param y:  (Default value = None) y coords for cell cortex points.  Can default to use stored values.
        :type y:  list
        :return: If the boundary intersects.
        :rtype: bool

        """

        if x is None or y is None:
            x = self.x
            y = self.y

        if not self.adhesion_polygon:
            self.build_adhesion_tree(build_polygon=True)

        # Build the polygon and check for intersections
        return not self.adhesion_polygon.contains(geom.Polygon(zip(x, y)))


    def move_cortex_nodes_to_equilibrium_dist(self, scaling=1, neighbours_moving=True):
        """Enforces that the cortex is no closer than delta to any neighbouring cortex

        :param scaling:  (Default value = 1)  Additional scaling relative to delta that the cortex nodes will sit from neighbours.
        :type scaling:  float
        :param neighbours_moving:  (Default value = True)  An additional check if the neighbouring cortices will move.
        :type neighbours_moving:  bool
        :return: Wether we were successful.
        :rtype: bool

        """

        # Cortex nodes
        cortex_nodes = np.dstack((self.x, self.y))[0]

        # Indices of nearest adhesions
        min_idxs = np.array([np.argmin(ads) if len(ads) > 0 else None for ads in self.adhesion_distances])

        # Get indices of short distances
        too_close_indices = [self.adhesion_distances[i][min_idxs[i]] < self.delta if min_idxs[i] != None else False
                             for i in range(len(min_idxs))]

        if any(too_close_indices):
            # Vectors from adhesion
            adhesion_nodes = np.array([self.adhesion_connections[i][min_idxs[i]]
                                       for i in range(len(too_close_indices)) if too_close_indices[i]])

            # Get the distances
            adhesion_distances = np.array([self.adhesion_distances[i][min_idxs[i]]
                                           for i in range(len(too_close_indices)) if too_close_indices[i]])

            # Sliced cortex nodes
            cortex_nodes = cortex_nodes[too_close_indices]

            # vec to New positions
            shift_from_adhesions = cortex_nodes - adhesion_nodes
            # shift_from_adhesions /= np.sqrt((shift_from_adhesions ** 2).sum(-1))[..., np.newaxis]
            shift_from_adhesions /= adhesion_distances[..., np.newaxis]
            shift_from_adhesions *= scaling * self.delta

            if neighbours_moving:
                # Get the identities
                adhesion_nodes_connections = np.array([self.adhesion_connections_identities[i][min_idxs[i]]
                                                       for i in range(len(min_idxs)) if too_close_indices[i]])

                # Nodes not neighbouring boundary move only half each (because other cell will also move)
                adhesion_distances_sliced = adhesion_distances[adhesion_nodes_connections != 'boundary']
                shift_from_adhesions[adhesion_nodes_connections != 'boundary'] *= 0.5 * (
                            adhesion_distances_sliced + self.delta)[..., np.newaxis]

            self.x[too_close_indices] = adhesion_nodes[:, 0] + shift_from_adhesions[:, 0]
            self.y[too_close_indices] = adhesion_nodes[:, 1] + shift_from_adhesions[:, 1]

            return True
        else:
            return False


    def scale_whole_cell_to_fit_adhesion_to_delta(self, stretch_factor=1.01, delta_tol=1.05,
                                                  update_adhesion_lengths=True):
        """Applies a homogenous isotropic stretch/compression to the cell such that the shortest
        adhesion is delta

        :param stretch_factor:  (Default value = 1.01)  How much to scale the cortex ``x -> x * stretch_factor``
        :type stretch_factor:  float
        :param delta_tol:  (Default value = 1.05)  Tolerance range for ``delta``.
        :type delta_tol:  float
        :param update_adhesion_lengths:  (Default value = True) Whether to perfrom an initial adhesion update.
        :type update_adhesion_lengths:  bool

        """

        # Get the current shortest adhesion
        min_adh = self.get_length_of_shortest_adhesion(rerun_distance_calculation=update_adhesion_lengths)
        # If it's longer, stretch the cell until it fits
        while min_adh > self.delta * delta_tol:
            # centroids
            C_x = np.mean(self.x)
            C_y = np.mean(self.y)
            # temp locations
            xs = self.x - C_x
            ys = self.y - C_y
            # scale
            self.x = xs * stretch_factor + C_x
            self.y = ys * stretch_factor + C_y
            self.update_reference_configuration_to_current()
            # now see how small it is
            min_adh = self.get_length_of_shortest_adhesion(rerun_distance_calculation=update_adhesion_lengths)

        # Otherwise, if the cell is too big, compress it.
        while min_adh < self.delta * (2 - delta_tol):
            # centroids
            C_x = np.mean(self.x)
            C_y = np.mean(self.y)
            # temp locations
            xs = self.x - C_x
            ys = self.y - C_y
            # scale
            self.x = xs * (2 - stretch_factor) + C_x
            self.y = ys * (2 - stretch_factor) + C_y
            self.update_reference_configuration_to_current()

            min_adh = self.get_length_of_shortest_adhesion(rerun_distance_calculation=update_adhesion_lengths)

        self.update_reference_configuration_to_current()


    def activate_fast_adhesions(self, on_off):
        """(De)activates the forces coming from fast adhesions

        :param on_off:  Set whether fast adhesions exert forces on the cortex.
        :type on_off: bool

        """
        self.fast_adhesions_active = on_off


    def build_adhesion_tree(self, build_polygon=False):
        """Builds a kdtree of the possible adhesion locations from self.adhesion_point_coords

        :param build_polygon:  (Default value = False) Set whether the adhesion polygon will be built. (It's not used much anymore)
        :type build_polygon:  bool

        """

        #  Extract the adhesion points
        adhesionNodes = self.adhesion_point_coords

        if not hasattr(self, 'adhesion_type'):
            self.adhesion_type = "nearest"
        self.adhesion_type = "fixed_radius" if self.adhesion_type == "fixed_length" else self.adhesion_type  # Todo remove
        assert (self.adhesion_type in ["nearest", "meanfield", "fixed_radius"]), \
            "Error, self.adhesion_type not available"

        if self.adhesion_type == "nearest":
            # Create the tree with just 1 neighbour
            nbrs = NearestNeighbors(n_neighbors=1,
                                    algorithm='auto', n_jobs=-1).fit(adhesionNodes)
        # Meanfiled adjusts the maximum adhesion length to make sure we can bind everywhere.
        elif self.adhesion_type == "meanfield":
            # Find what the furthest neighbour is
            temp_tree = NearestNeighbors(n_neighbors=1,
                                         algorithm='auto', n_jobs=-1).fit(adhesionNodes)
            distances, indices = temp_tree.kneighbors(np.dstack((self.x, self.y))[0])
            max_dist = np.max(distances)
            self.max_adhesion_length = max_dist * 1.2
            # max_dist = self.max_adhesion_length
            # Now build the new tree for mean-field
            nbrs = NearestNeighbors(radius=max_dist * 1.2,
                                    algorithm='auto', n_jobs=-1).fit(adhesionNodes)
        # Fixed radius gets all connections within self.adhesion_search_radius
        elif self.adhesion_type == "fixed_radius":

            # Create the tree
            nbrs = NearestNeighbors(radius=self.adhesion_search_radius,
                                    algorithm='auto', n_jobs=-1).fit(adhesionNodes)
        # Store it
        self.adhesion_tree = nbrs

        # Create an adhesion polygon as well.
        if build_polygon:
            self.adhesion_polygon = geom.Polygon(adhesionNodes)

        return


    def get_adhesion_nodes_connected_to_xy(self, nodes, sort_by_distance=True):
        """Calculate the adhesions that will be connected to given a list of (x,y) coordinates.

        :param nodes:  The list of coordinates that will look for adhesion connections.
        :type nodes: list
        :param sort_by_distance:  (Default value = True)  Sort the return list by distance.
        :type sort_by_distance:  bool
        :return: The ``distances`` that the nodes have to the adhesion boundary, and the ``indices`` matching the output distances to the input nodes.
        :rtype: (list, list)

        """

        if self.adhesion_type == "nearest":
            # Query the tree to get the nearest neighbours.
            distances, indices = self.adhesion_tree.kneighbors(nodes)
        else:
            # Query the tree to get the nearest neighbours.
            distances, indices = self.adhesion_tree.radius_neighbors(nodes, return_distance=True,
                                                                     sort_results=sort_by_distance)
            # distances, indices = self.adhesion_tree.kneighbors(nodes)

        return distances, indices


    def update_adhesion_distances_identifiers_and_indices(self, x=None, y=None, sort_by_distance=True,
                                                      build_tree=False):
        """Query the adhesion tree to build adhesion connections, storing the distance, id and index of the connection.
        The data is stored internally, with no output here.

        :param x:  (Default value = None)  x-coordintes to use.  Defaults to stored cortex x variable.
        :type x:  np.array
        :param y:  (Default value = None) y-coordintes to use.  Defaults to stored cortex x variable.
        :type y:  np.array
        :param sort_by_distance:  (Default value = True)  Whether to rganise the adhesion connections by distance.
        :type sort_by_distance:  bool
        :param build_tree:  (Default value = False)  Whether to fresh-build the adhesion tree.
        :type build_tree:  bool

        """

        if x is None or y is None:
            x = self.x
            y = self.y

        cortex_nodes = np.dstack((x, y))[0]

        # Get the connections and their indices.
        if build_tree or (not hasattr(self, 'adhesion_tree')) or self.adhesion_tree is None:
            self.build_adhesion_tree()
        adh_distances, indices = self.get_adhesion_nodes_connected_to_xy(cortex_nodes,
                                                                               sort_by_distance=sort_by_distance)

        # If in hex, just fit closest
        if self.adhesion_type == "nearest":
            adh_distances = [[ads[0]] if ads else [] for ads in adh_distances]
            indices = [[idx[0]] if idx else [] for idx in indices]

        # get unique attachments for calculating prestrain
        all_attachments = [[self.prestrain_dict[self.adhesion_point_identifiers[i]] for i in ind] for ind in indices]
        # self.adhesion_connections_identities_unique = [set(sub_list) for sub_list in all_attachments]
        self.adhesion_connections_identities_unique = [Counter(sub_list) for sub_list in all_attachments]

        if not self.adhesion_type == 'nearest':
            # Filter anything beyond max adhesion length.
            to_keep = [np.where(np.array(d) <= self.max_adhesion_length)[0] for d in adh_distances]
            indices = [indices[i][to_keep[i]] for i in range(len(indices))]
            adh_distances = [adh_distances[i][to_keep[i]] for i in range(len(adh_distances))]

        # # Sort by distance
        # if sort_by_distance:
        #     sorted_lists = [sort_together([adh_distances[i], indices[i]]) if len(adh_distances[i]) > 0 else [[], []]
        #                     for i in range(0, adh_distances.shape[0])]
        #     adh_distances, indices = zip(*sorted_lists)

        # Limit max number of adhesions for speed
        max_adhesions_stored = self.max_num_adhesions_for_force
        # slice_facs = [int(len(d)/max_adhesions_stored) + 1 if int(len(d)/max_adhesions_stored) % 2 ==0
        #               else int(len(d)/max_adhesions_stored) for d in adh_distances]
        # adh_distances = [adh_distances[i][::slice_facs[i]] for i in range(len(adh_distances))]
        # indices = [indices[i][::slice_facs[i]] for i in range(len(adh_distances))]

        indices = [indices[i][np.argsort(adh_distances[i])][:max_adhesions_stored] for i in range(len(adh_distances))]
        adh_distances = [adh_distances[i][np.argsort(adh_distances[i])][:max_adhesions_stored] for i in
                         range(len(adh_distances))]

        # Store the distances
        self.adhesion_distances = adh_distances
        # Store the connections
        self.adhesion_connections = [[self.adhesion_point_coords[i] for i in ind] for ind in indices]
        # And their identities
        self.adhesion_connections_identities = [[self.adhesion_point_identifiers[i] for i in ind] for ind in indices]
        # and their spacings
        self.adhesion_connections_spacings = [[self.adhesion_point_spacings[i] for i in ind] for ind in indices]


    def get_total_adhesion_force_from_adhesion_indices(self, coord, adhesion_index, is_intersection=False):
        """For fast adhesions.  Given (x,y) coord on this cortex and index (adhesion_idx) in adhesion list that it is connected to,
        calculate vector force.  Note self.adhesion_connections[adhesion_index] can be a list of multiple connections.

        :param coord:  The (x,y) coord to get the adhesion force for.
        :type adhesion_index:   The index of the adhesion node it will be connected to.
        :param is_intersection:  (Default value = False).  Unused now.  If it is intersecting the adhesion boundary, the force is reversed.
        :type is_intersection:  bool
        :return: The ``distances`` that the nodes have to the adhesion boundary, and the ``indices`` matching the output distances to the input nodes.
        :rtype: (list, list)
        
        """

        # Get the lengths of the connections that were made.
        d = self.adhesion_distances[adhesion_index]

        # This if statement is depreciated \todo
        if d is None:
            d = self.max_adhesion_length
            e = d - self.delta
            force = -0.5 * self.omega0 * e

        # Get force magnitude
        else:
            # Make into numpy array
            d = np.array(d)

            # calculate the forces
            if self.adhesion_force_law == "spring":
                # Get spring extension
                e = d - self.delta
                # Set maximum bonding distance
                e[d > self.max_adhesion_length] = 0.
                # Calc spring force
                force = self.omega0 * e

                # Multiply by spacing on the other cortices.
                force *= np.array(self.adhesion_connections_spacings[adhesion_index])
                # spacing = np.array(self.adhesion_connections_spacings[adhesion_index])
                # np.multiply(force, spacing[:, np.newaxis], casting='unsafe', out=force)

                # Short range repulsion if compressed
                # force[e < 0] = self.get_dOmega_dY( d[e<0] )
                # force[e < 0] *= 1/d[e < 0]
                # force[e < 0] *= 0.5*(np.log(d[e < 0]) / np.log(0.5))
                # force[e < 0] = - 1 * self.omega0 * (np.log(d[e < 0]) / np.log(0.5))

                # scale the force from each adhesion using softmax function
                exp_factor = self.adhesion_beta_scale_factor
                dist_scaling = np.exp(-exp_factor * d)
                if np.sum(dist_scaling) != 0:
                    dist_scaling /= np.sum(dist_scaling)
                force *= dist_scaling
            else:
                raise NotImplementedError('Fast adhesions must have self.adhesion_force_law="spring"')

            # # Log_1/2 force
            # stretch = d / (self.delta * multiplier)
            # # force[stretch < 1] = self.omega0 * (np.log(stretch[stretch < 1]) / np.log(0.5))
            # force = self.omega0 * (np.log(stretch) / np.log(0.5))
            # force[stretch <= 1e-8] = self.omega0 * (np.log(1e-8) / np.log(0.5))

        # If the adhesion density != 1 on this cortex, or the connection, scale the forces
        if any([ad_den != 1 for ad_den in self.adhesion_density_dict.values()]):
            ids = self.adhesion_connections_identities[adhesion_index]
            force_scale = np.array(list(map(lambda x: self.adhesion_density_dict[x], ids)))
            # force_scale = np.array(list(map(lambda x: self.adhesion_density_dict[x] * id_spacing_dict[x], ids)))
            force *= force_scale

        # Get the unit direction of the force
        dirs = [[ad[0] - coord[0], ad[1] - coord[1]] for ad in self.adhesion_connections[adhesion_index]]
        dirs = np.array(dirs)
        dirs = np.divide(dirs, np.sqrt(inner1d(dirs, dirs))[:, np.newaxis])

        # If we haven't intersected the other cell, multiply by the direction
        if not is_intersection:
            # Mulitply normals by magnitude of force to get force vectors
            vector_of_forces = dirs * force[:, np.newaxis]
        else:
            vector_of_forces = - dirs * np.linalg.norm(force)  # 1e6

        # Sum them all to get the average.
        vector_force = np.sum(vector_of_forces, axis=0)

        return vector_force

    def get_fast_adhesion_forces_across_cortex(self, x=None, y=None, sort_by_distance=True):
        """Query the adhesion tree to get indices of neighouring cortices within the max adhesion length.
        Then calculate the distance to each point and calculate the force.

        :param x:  (Default value = None)  x-coordintes to use.  Defaults to stored cortex x variable.
        :type x:  np.array
        :param y:  (Default value = None) y-coordintes to use.  Defaults to stored cortex y variable.
        :type y:  np.array
        :param sort_by_distance:  (Default value = True) Organise the adhesions by distance.
        :type sort_by_distance:  bool
        :return: A list of the adhesion forces acting across the cortex.
        :rtype: np.array
        
        """

        if x is None or y is None:
            x = self.x
            y = self.y

        # Get total num nodes
        num_nodes = x.size

        # Get (x,y) coords of nodes on this cortex
        cortex_nodes = np.dstack((x, y))[0]

        # Update the distances
        self.update_adhesion_distances_identifiers_and_indices(x=x, y=y, sort_by_distance=sort_by_distance)

        # Determine if we have intersected another cell or the adhesion boundary.  Actually this isn't necessary
        # idxs = self.get_intersection_indices(cortex_nodes)
        # intersection_bools = np.zeros(num_nodes)
        # intersection_bools[idxs] = True
        # intersection_bools = np.zeros(num_nodes)

        # Calculate the forces
        forces = [
            self.get_total_adhesion_force_from_adhesion_indices(cortex_nodes[i], i)
            if self.adhesion_connections[i] else np.array([0, 0])
            for i in range(0, num_nodes)]
        forces = np.array(forces)

        # Scale the forces with the discretisation.
        if x.size > 2:
            spacing = self.get_xy_segment_lengths(x, y)  # ** 2
            # Multiple forces by scaling
            np.multiply(forces, spacing[:, np.newaxis], casting='unsafe', out=forces)

        # Storing the forces so they don't need to be recalculated
        self.adhesion_forces = forces

        return forces

    def get_sdk_forces_across_cortex(self, x, y, s):
        """Get a vector len(self.s) of the total force on each cortex node due to sdk

        :param x:  (Default value = None)  x-coordintes to use.  Defaults to stored cortex x variable.
        :type x:  np.array
        :param y:  (Default value = None) y-coordintes to use.  Defaults to stored cortex y variable.
        :type y:  np.array
        :param s:  (Default value = None) S_0-coordintes to use.  Defaults to stored cortex S_0 variable.
        :type s:  np.array
        :return: A list of the sidekick adhesion forces acting across the cortex.
        :rtype: np.array

        """
        s = self.s if s is None else s
        x = self.x if x is None else x
        y = self.y if y is None else y

        # Update the mesh spacing
        self.update_deformed_mesh_spacing(x, y)

        # Intitialise the vector of forces, which we will fill for every node that has a sidekick adhesion.
        force_vector = np.zeros((s.size, 2))

        # For every sidekick adhesion, find the node it's attached to and calculate the force.
        for ad in self.sidekick_adhesions:
            local_idx = ad[0]
            # Using the collocation method, the size of x changes. If it's too big, just use end value
            local_idx = x.size - 1 if local_idx > x.size - 1 else local_idx
            x_local, y_local = x[local_idx], y[local_idx]
            # Force and direction
            dx, dy = ad[1] - x_local, ad[2] - y_local
            direction = [dx, dy]
            magnitude = np.sqrt(direction[0] ** 2 + direction[1] ** 2)
            direction[0] /= magnitude
            direction[1] /= magnitude

            length = np.sqrt(dx * dx + dy * dy)
            e = length - self.sdk_restlen if length < self.max_adhesion_length else 0
            force = self.sdk_stiffness * e

            force_acting = [direction[0] * force * self.deformed_mesh_spacing[local_idx],
                            direction[1] * force * self.deformed_mesh_spacing[local_idx]]

            force_vector[local_idx, 0] += force_acting[0]  # * self.mesh_spacing[local_idx]
            force_vector[local_idx, 1] += force_acting[1]  # * self.mesh_spacing[local_idx]

        return force_vector

    def get_slow_adhesion_forces_across_cortex(self, x=None, y=None):
        """Calculate a vector len(self.s) giving the total force at every cortexz node coming from the population of
        slow adhesions.

        :param x:  (Default value = None)  x-coordintes to use.  Defaults to stored cortex x variable.
        :type x:  np.array
        :param y:  (Default value = None) y-coordintes to use.  Defaults to stored cortex y variable.
        :type y:  np.array
        :return: A list of the slow adhesion forces acting across the cortex.
        :rtype: np.array

        """
        x = self.x if x is None else x
        y = self.y if y is None else y

        #Update mesh spacing for scaling
        self.update_deformed_mesh_spacing(x, y)

        force_vector = np.zeros((x.size, 2))

        for ad in self.slow_adhesions:
            # Slow adhesion data format = (local_cell_index, other_cell_x, other_cell_y, other_cell_spacing)

            local_idx = ad[0]
            # Using the collocation method, the size of x changes. If it's too big, just use end value
            local_idx = x.size - 1 if local_idx > x.size - 1 else local_idx
            x_local, y_local = x[local_idx], y[local_idx]
            # Force and direction
            dx, dy = ad[1] - x_local, ad[2] - y_local
            direction = [dx, dy]
            magnitude = np.sqrt(direction[0] ** 2 + direction[1] ** 2)
            direction[0] /= magnitude
            direction[1] /= magnitude

            length = np.sqrt(dx * dx + dy * dy)
            e = length - self.delta if length < self.max_adhesion_length else 0
            force = self.omega0 * e

            force_acting = [direction[0] * force * self.deformed_mesh_spacing[local_idx],
                            direction[1] * force * self.deformed_mesh_spacing[local_idx]]

            # Multiple by mesh spacing on the other cotrex.
            other_mesh_spacing = ad[3]
            force_vector[local_idx, 0] += force_acting[0] * other_mesh_spacing
            force_vector[local_idx, 1] += force_acting[1] * other_mesh_spacing

        return force_vector


    def get_total_adhesion_force_across_cortex(self, x=None, y=None, s=None):
        """Get the total adhesion force, with fast, sdk and slow adhesions

        :param x:  (Default value = None)  x-coordintes to use.  Defaults to stored cortex x variable.
        :type x:  np.array
        :param y:  (Default value = None) y-coordintes to use.  Defaults to stored cortex y variable.
        :type y:  np.array
        :param s:  (Default value = None) S_0-coordintes to use.  Defaults to stored cortex S_0 variable.
        :type s:  np.array
        :return: A list of the sidekick adhesion forces acting across the cortex.
        :rtype: np.array

        """
        x = self.x if x is None else x
        y = self.y if y is None else y
        s = self.s if s is None else s

        # Get fast adhesion forces
        if self.fast_adhesions_active:
            ad_forces = self.get_fast_adhesion_forces_across_cortex(x, y)
        else:
            ad_forces = np.zeros((x.size, 2))

        # Sidekick forces
        if len(self.sidekick_adhesions) > 0:
            sdk_forces = self.get_sdk_forces_across_cortex(x, y, s)
            # add the forces to the adhesions
            np.add(ad_forces, sdk_forces, casting='unsafe', out=ad_forces)

        # Slow adhesions
        if len(self.slow_adhesions) > 0:
            slow_adh_forces = self.get_slow_adhesion_forces_across_cortex(x, y)
            # add the forces to the adhesions
            np.add(ad_forces, slow_adh_forces, casting='unsafe', out=ad_forces)

        self.adhesion_forces = ad_forces

        return ad_forces

    def get_protrusion_force(self, x, y):
        """Make a protrusion pointing to another cell and get a list of the forces acting across the cortex due to it
         (this will mostly be a sparse list of [0,0] except where the protrusion is)  \todo this is old might not work anymore

        :param x:  (Default value = None)  x-coordintes to use.  Defaults to stored cortex x variable.
        :type x:  np.array
        :param y:  (Default value = None) y-coordintes to use.  Defaults to stored cortex y variable.
        :type y:  np.array
        :return: A list of the forces acting across the cortex due to protrusions.
        :rtype: np.array

        """
        if self.protrusion_target == None or self.protrusion_force == 0:
            return np.zeros((x.size, 2))
        else:
            target_cell = self.protrusion_target

        # Stack a list of the x/y positions
        nodes_on_cortex = np.dstack((x, y))[0]

        # Get the coordinates of the target cell from the list of possible adhesion points.
        adhesionNodes = self.adhesion_point_coords
        # Extract which cortex they are adhering to
        id_list_adh = self.adhesion_point_identifiers
        # Get the ones belonging to he desired cortex only.
        target_cell_nodes = adhesionNodes[id_list_adh == target_cell]

        # Create a KD tree to get the distances between all nodes
        tree_of_neighbours = cKDTree(target_cell_nodes)
        distance, indices_to_points = tree_of_neighbours.query(nodes_on_cortex, k=1)
        # Find the closest node on this cortex
        node_on_this_cortex = nodes_on_cortex[np.argmin(distance)]
        # And the closest on the neighbour
        node_on_target_cortex = target_cell_nodes[indices_to_points[np.argmin(distance)]]

        # Specify the direction of the force
        direction = node_on_target_cortex - node_on_this_cortex
        direction_unit = direction / np.sqrt(inner1d(direction, direction))
        # Add the force
        force = self.protrusion_force * direction_unit

        force_list = [[0, 0] for i in x]
        if np.min(distance) > self.max_adhesion_length * .75:
            force_list[np.argmin(distance)] = force

        force_list = np.array(force_list)

        # Get the ids of the nearest/most common cortex
        ids = [max(self.adhesion_connections_identities[i], key=Counter(self.adhesion_connections_identities[i]).get)
               if self.adhesion_connections_identities[i] else 'none'
               for i in range(0, x.size)]
        # Remove forces on the boundary
        force_list[ids == 'boundary'] = 0.

        return force_list


    def update_prestrains(self):
        """Update the values of prestretches (called prestrain here based on the identity of (fast) adhesion connections    
        """
        
        self.prestrains = self.get_prestrains()


    def get_prestrains(self, shape=None):
        """Returns an array of the calculated prestrethc (called prestrain) at every mesh point, using the
        fast adhesion methods

        :param shape:  (Default value = None) The desired return shape (used when solving bvp with collocation nodes).
        :type shape:  int
        :return: The prestretch across the cortex.
        :rtype: np.array

        """
        assert len(self.adhesion_connections_identities) > 0, 'Error, need to update adhesions'

        shape = self.x.size if shape is None else shape

        prestrain = np.ones(shape)
        # Calculate prestrain based on type
        if any([i != 1 for i in self.prestrain_dict.values()]):

            assert self.prestrain_type in ["most_common", "average", "min", "nearest"], "Error self.prestrain_type " \
                                                                                        "not in [most_common, average, min, nearest]."

            # Choose to prestain in cortex according to most common connection:
            if self.prestrain_type == "most_common":
                id_list = [max(ad_id, key=Counter(ad_id).get) if ad_id else 'none'
                           for ad_id in self.adhesion_connections_identities]
                # Get the dictionary to scale with
                prestrain = np.array(list(map(lambda val: self.prestrain_dict[val], id_list)))

            # choose the average prestrain
            elif self.prestrain_type == "average":
                # prestrain = np.array([np.mean([self.prestrain_dict[val] for val in sub_list])
                #                             if sub_list else 1. for sub_list in self.adhesion_connections_identities])
                # If it's a counter dict
                prestrain = np.array([sum(k * v for k, v in sub_list.items()) / sum(sub_list.values())
                                      if sub_list else 1. for sub_list in
                                      self.adhesion_connections_identities_unique[:shape]])

            elif self.prestrain_type == "min":
                # choose the min prestrain (will contract if any ads are connected to the cortex)
                # prestrain = np.array([np.min([self.prestrain_dict[val] for val in sub_list])
                #                       if sub_list else 1. for sub_list in self.adhesion_connections_identities])
                # # When it's a set
                # prestrain = np.array([np.min([self.prestrain_dict[val] for val in sub_list])
                #                       if sub_list else 1. for sub_list in self.adhesion_connections_identities_unique])
                # If it's a counter dict
                prestrain = np.array([min(sub_list.keys())
                                      if sub_list else 1. for sub_list in
                                      self.adhesion_connections_identities_unique[:shape]])

            elif self.prestrain_type == "nearest":
                # prestrain = np.array([self.prestrain_dict[self.adhesion_connections_identities[idx][np.argmin(self.adhesion_distances[idx])]]
                #                       if self.adhesion_connections_identities[idx] else 1. for idx in range(x.size)])
                #  If sorted, choose nearest.
                prestrain = np.array(
                    [self.prestrain_dict[self.adhesion_connections_identities[idx][0]]
                     if self.adhesion_connections_identities[idx] else 1. for idx in range(shape)])

        # Add any local prestrian to specific indices
        if len(self.prestrain_indices_list) != 0:
            indices_to_change = [pair[0] for pair in self.prestrain_indices_list]
            values_to_change = [pair[1] for pair in self.prestrain_indices_list]
            prestrain[indices_to_change] *= values_to_change

        return prestrain


    def get_first_derivative_of_cortex_variables(self, x=None, y=None, theta=None, gamma=None, C=None, D=None, s=None):
        """ Get the first derivative of all cortex variables. (Note, return order differs to input order \todo)

        :param x:  (Default value = None)  x-coordinates to use.  Defaults to stored cortex x variable.
        :type x:  np.array
        :param y:  (Default value = None) y-coordinates to use.  Defaults to stored cortex y variable.
        :type y:  np.array
        :param theta:  (Default value = None) cortex angles to use.  Defaults to stored cortex theta variable.
        :type theta:  np.array
        :param gamma:  (Default value = None) cortex stretches to use.  Defaults to stored cortex gamma variable.
        :type gamma:  np.array
        :param C:  (Default value = None) theta' to use.  Defaults to stored cortex C variable.
        :type C:  np.array
        :param D:  (Default value = None) theta'' to use.  Defaults to stored cortex D variable.
        :type D:  np.array
        :param s:  (Default value = None) S_0-coordintes to use.  Defaults to stored cortex S_0 variable.
        :type s:  np.array
        :return: A list of arrays, where each array is the first deriv of a cortex variable.
        :rtype: list

        """

        update_adh_force = True if x is not None or self.adhesion_forces is None else False

        x = self.x if x is None else x
        y = self.y if y is None else y
        theta = self.theta if theta is None else theta
        gamma = self.gamma if gamma is None else gamma
        C = self.C if C is None else C
        D = self.D if D is None else D
        s = self.s if s is None else s

        #######
        # External forces
        #######

        # Get adhesion forces
        ad_forces = self.adhesion_forces if not update_adh_force else \
            self.get_total_adhesion_force_across_cortex(x, y, s)

        # Include protrusions
        if self.protrusion_force != 0:
            # Add the protrusion forces
            protrusion_forces = self.get_protrusion_force(x, y)
            # add the protrusion forces to the adhesions
            np.add(ad_forces, protrusion_forces, casting='unsafe', out=ad_forces)

        # Project onto normal and tangential directions
        normal = [-np.sin(theta), np.cos(theta)]
        # normal = [np.sin(theta), -np.cos(theta)]
        tangent = [np.cos(theta), np.sin(theta)]
        gradOmegaDotN = (ad_forces.T * normal).sum(axis=0)
        gradOmegaDotT = (ad_forces.T * tangent).sum(axis=0)

        # Pressure forces:
        if self.pressure_on_off and (self.area_stiffness != 0 or self.pressure != 0):
            # pressure = self.pressure
            pressure = - self.area_stiffness * (self.get_area(x, y) - self.pref_area) + self.pressure

            # Get the ids of the nearest/most common cortex
            ids = [
                max(self.adhesion_connections_identities[i], key=Counter(self.adhesion_connections_identities[i]).get)
                if self.adhesion_connections_identities[i] else 'none'
                for i in range(0, x.size)]

            # Scale medial with adhesion density
            pressure_scale = np.array(list(map(lambda val: self.adhesion_density_dict[val], ids)))
            # Remove pressure force if there are no cadherins
            pressure_scale[np.sum(ad_forces, axis=1) == 0] = 0

            spacing = self.get_xy_segment_lengths(x, y)

            total_pressure = pressure * pressure_scale * spacing

            # Add pressure to normal force
            np.subtract(gradOmegaDotN, total_pressure, casting='unsafe', out=gradOmegaDotN)

        #######
        # Add cortex material forces
        #######

        # Get the prestrains.
        prestrain = self.get_prestrains(shape=x.size)

        # Now gamma (alpha in manuscript) and D
        if self.constitutive_model == 'linear':
            # Gradients of prestrain
            prestrain_inv = 1 / prestrain
            prestrain_grad1 = np.gradient(prestrain, s)
            # prestrain_grad1 = savgol_filter(prestrain_grad1, 21, 1)
            prestrain_grad2 = np.gradient(prestrain_grad1, s)
            # prestrain_grad2 = savgol_filter(prestrain_grad2, 21, 1)

            # Gamma
            gamma_inv = 1 / gamma
            # d_gamma_ds = self.kappa * (prestrain_inv ** 3) * gamma_inv * C * (C * prestrain_grad1 - D * gamma) \
            d_gamma_ds = self.kappa * (prestrain_inv ** 3) * gamma_inv * C * (C * prestrain_grad1 - D * prestrain) \
                         - prestrain * gradOmegaDotT

            # D
            dDds = self.kappa * D * (3 * prestrain_grad1 * prestrain_inv + d_gamma_ds * gamma_inv) + \
                   self.kappa * C * (prestrain_grad2 * prestrain_inv - 3 * (prestrain_grad1 ** 2) * (gamma_inv ** 2) -
                                     d_gamma_ds * prestrain_grad1 * gamma_inv * prestrain_inv) + \
                   (prestrain ** 2) * gamma * C * (gamma - 1) + \
                   (prestrain ** 3) * gamma * gradOmegaDotN
            dDds /= self.kappa

        elif self.constitutive_model == 'hyperelastic':
            raise NotImplementedError('Hyperelastic model not implemented')

        # Theta
        dThetaDs = C

        # C
        dCds = D

        # X, Y
        dXds = np.cos(theta) * (gamma * prestrain)
        dYds = np.sin(theta) * (gamma * prestrain)

        return np.vstack((dThetaDs, d_gamma_ds, dXds, dYds, dDds, dCds))


    def funcForScipyBVPSolver(self, s, U):
        """Function to pass to Scipy's BVP solver; returns the cortex equilibrium equations from the force balance.
        U = [theta, gamma, x, y, D, C]

        :param s: S_0 cortex coordintes (the undeformed mesh).
        :type s:  np.array
        :param U: A list of the cortex variables ``[theta, gamma, x, y, C, D]`` (passed as arrays; unintuitive ordering)
        :type U:
        :return: The first derivative of the cortex variables, which will be solved by Scipy.
        :rtype: list

        """

        theta, gamma, x, y, D, C = U[0], U[1], U[2], U[3], U[4], U[5]

        cortex_eqns = self.get_first_derivative_of_cortex_variables(s=s, x=x, y=y, theta=theta, gamma=gamma, C=C, D=D)

        return cortex_eqns


    def bvp_bcs(self, ya, yb):
        """Check how close we are to periodic BCS for cortex variables.  Used for solving BVP

        :param ya: The current value of the cortex variables at the first index: ``[theta[0], gamma[0], x[0], y[0], C[0], D[0]]``
        :type ya: list
        :param yb: The current value of the cortex variables at the first index: ``[theta[-1], gamma[-1], x[-1], y[-1], C[-1], D[-1]]``
        :type yb: list
        :return: A list of the differences between the enp-point variables (theta is mod 2pi).
        :rtype: np.array

        """
        BCStateVector = []

        # For theta  it needs to be mod(2pi)
        BCStateVector.append(ya[0] % (2 * np.pi) - yb[0] % (2 * np.pi))
        # Others are just continuity.
        BCStateVector.append((ya[1] - yb[1]) / 1)
        BCStateVector.append((ya[2] - yb[2]) / 1)
        BCStateVector.append((ya[3] - yb[3]) / 1)
        BCStateVector.append((ya[4] - yb[4]) / 1)
        BCStateVector.append((ya[5] - yb[5]) / 1)

        return np.array(BCStateVector)


    def solve_bvp(self):
        """Solve BVP for cortex variables using Scipy's BVP solver

        :return:  A tuple, whether solving was a success; a list, the new cortex variables: ``[s, theta, gamma, x, y, C, D]``
        :rtype: (bool, list)
        """

        verbose = self.verbose if not self.verbose else self.verbose + 1

        self.verboseprint(" ".join(["Solving BVP for cell %s " % self.identifier]), object(), 1)

        # Make copies of the variables and reduce their length if the lists are too long.
        gammaTemp = self.gamma + 0.
        thetaTemp, xTemp, yTemp = self.theta + 0., self.x + 0., self.y + 0.
        DTemp, CTemp = self.D + 0., self.C + 0.
        sTemp = self.s + 0.
        #  Make a vector of the variables
        U1 = np.array([thetaTemp, gammaTemp, xTemp, yTemp, DTemp, CTemp])

        # Tell the solver not to add nodes
        maxNodes = xTemp.size

        # sTemp must be strictly increasing, so enforce that if it isn't
        checker = all(x < y for x, y in zip(sTemp, sTemp[1:]))
        while not checker:
            self.verboseprint("s is not striclty increasing. Forcing monotonicity", object(), 1)
            vals, idx_start = np.unique(sTemp, return_index=True)

            # sets of indices
            res = np.split(np.arange(sTemp.size), idx_start[1:])
            # filter them with respect to their size, keeping only items occurring more than once
            res = list(filter(lambda x: x.size > 1, res))
            res = np.array(res).flatten()
            # Get consecutive values and remove them
            grouped_idxs = np.split(res, np.where(np.diff(res) != 1)[0] + 1)
            # Ungroup and every second element of consecutive numbers
            indices_to_add_to = [val for sublist in grouped_idxs for val in sublist[::2]]

            # Alter s
            sTemp[indices_to_add_to] -= 1e-6

            # check again
            checker = all(x < y for x, y in zip(sTemp, sTemp[1:]))

        # Solve the BVP using Scipy
        sol1 = solve_bvp_scipy(self.funcForScipyBVPSolver, self.bvp_bcs, sTemp, U1, max_nodes=int(maxNodes),
                               verbose=verbose, tol=self.relax_tol)

        # Check if the sol1ver passed
        if sol1.status != 0 and self.verbose:
            print("WARNING: sol1.status is %d for cell %s" % (sol1.status, self.identifier))

        # Update the cortex variables
        self.s = sol1.x
        self.gamma = sol1.y[1]
        self.theta, self.x, self.y = sol1.y[0], sol1.y[2], sol1.y[3]
        self.D, self.C = sol1.y[4], sol1.y[5]
        self.n = self.s.size

        # If the solver added nodes, just remove them (but we hard code it not to now)
        if sTemp.size != sol1.x.size:
            indices_to_keep = np.where(np.isin(sol1.x, sTemp))[0]
            self.s = self.s[indices_to_keep]
            self.theta = self.theta[indices_to_keep]
            self.gamma = self.gamma[indices_to_keep]
            self.x = self.x[indices_to_keep]
            self.y = self.y[indices_to_keep]
            self.C = self.C[indices_to_keep]
            self.D = self.D[indices_to_keep]

            # If above failed, just return what we put in
            if len(self.s) != len(sTemp):  # or sol1.status != 0:
                self.s = sTemp
                self.theta = thetaTemp
                self.gamma = gammaTemp
                self.x = xTemp
                self.y = yTemp
                self.C = CTemp
                self.D = DTemp

        # Unnecessary code lingering from previous iteration
        if self.n > self.decimateAbove:
            new_s = np.linspace(0, self.rest_len, self.decimateAbove)
            # Update variables
            self.theta = self.interpolate_variable_onto_new_grid(sol1.y[0], sol1.x, new_s)
            self.gamma = self.interpolate_variable_onto_new_grid(sol1.y[1], sol1.x, new_s)
            self.x = self.interpolate_variable_onto_new_grid(sol1.y[2], sol1.x, new_s)
            self.y = self.interpolate_variable_onto_new_grid(sol1.y[3], sol1.x, new_s)
            self.D = self.interpolate_variable_onto_new_grid(sol1.y[4], sol1.x, new_s)
            self.C = self.interpolate_variable_onto_new_grid(sol1.y[5], sol1.x, new_s)
            self.n = self.decimateAbove
            self.s = new_s

        # store status of last solve
        self.last_solve_status = sol1.status

        return sol1.status, [self.s, self.theta, self.gamma, self.x, self.y, self.D, self.C]


    def get_normal_and_tangential_components_of_cortex_forces(self):
        """Get normal and tangential components of force gradient exerted by cortex.
        This is sum of adhesion and pressure in equilibrium

        :return:  Cortex forces in the normal and tangential directions
        :rtype: (np.array, np.array)

        """

        # Get adhesion forces
        ad_forces = self.get_total_adhesion_force_across_cortex(self.x, self.y, self.s)

        # Normal and tangential dirs
        normal = [np.sin(self.theta), -np.cos(self.theta)]
        tangent = [np.cos(self.theta), np.sin(self.theta)]
        gradOmegaDotN = (ad_forces.T * normal).sum(axis=0)
        gradOmegaDotT = (ad_forces.T * tangent).sum(axis=0)

        # Pressure forces:
        if self.pressure_on_off and (self.area_stiffness != 0 or self.pressure != 0):
            # pressure = self.pressure
            pressure = - self.area_stiffness * (self.get_area(self.x, self.y) - self.pref_area) + self.pressure

            # Get the ids of the nearest/most common cortex
            ids = [
                max(self.adhesion_connections_identities[i], key=Counter(self.adhesion_connections_identities[i]).get)
                if self.adhesion_connections_identities[i] else 'none'
                for i in range(0, self.x.size)]

            # Scale medial with adhesion density
            pressure_scale = np.array(list(map(lambda val: self.adhesion_density_dict[val], ids)))
            # Remove pressure force if there are no cadherins
            pressure_scale[np.sum(ad_forces, axis=1) == 0] = 0
            pressure_scale = 1

            spacing = self.get_xy_segment_lengths(self.x, self.y)

            total_pressure = pressure * pressure_scale * spacing

            # Add pressure to normal force
            np.subtract(gradOmegaDotN, total_pressure, casting='unsafe', out=gradOmegaDotN)

        f_t = -gradOmegaDotT
        f_n = -gradOmegaDotN

        return f_n, f_t


    def get_cortex_forces(self):
        """Get the total forces exerted by the cortex (against adhesion), summed of normal and tang dirs.

        :return:  Total force acting across cortex.
        :rtype:  np.array

        """

        # Components of force gradient
        f_n, f_t = self.get_normal_and_tangential_components_of_cortex_forces()

        # Normal and tangential dirs
        normal = [np.sin(self.theta), -np.cos(self.theta)]
        tangent = [np.cos(self.theta), np.sin(self.theta)]

        # Forces
        normal_forces = np.multiply(normal, f_n)
        tangential_forces = np.multiply(tangent, f_t)

        # Total force:
        total_forces = np.dstack((np.add(normal_forces, tangential_forces)))[0]

        return total_forces


    def get_stress_tensor(self):
        """Calculate cell stress tensor

        :return:  The 2x2 stress tensor
        :rtype:  np.array
        """

        # Total force:
        total_forces = self.get_cortex_forces()

        # Centroid
        centroid = self.get_centroid()

        # Get the stress tensor
        stress = np.array([[0., 0.], [0., 0.]])
        for i in range(self.s.size):
            r_vector = np.array([self.x[i] - centroid[0], self.y[i] - centroid[1]])  # - centroid
            stress += np.outer(r_vector, total_forces[i])

        stress /= self.get_area()
        stress = 0.5 * (stress + stress.T)  # Enforce symmetric

        return stress


    def get_effective_pressure(self):
        """Isotropic part of cell-level stress

        :return:  Trace of the stress tensor
        :rtype:  float
        """

        return -0.5 * self.get_stress_tensor().trace()


    def get_shape_tensor(self):
        """Get cell shape tensor

        :return:  2x2 shape tensor.
        :rtype:  np.array

        """
        # Centroid
        centroid = np.array([np.mean(self.x), np.mean(self.y)])

        # Get the stress tensor
        shape = np.array([[0., 0.], [0., 0.]])
        for i in range(self.s.size):
            r_vector = np.array([self.x[i] - centroid[0], self.y[i] - centroid[1]])  # - centroid
            shape += np.outer(r_vector, r_vector)

        shape /= self.get_area()
        shape = 0.5 * (shape + shape.T)

        return shape


    def double_mesh(self):
        """put new mesh points at mean locations of current"""

        mean_s = [np.mean([self.s[i - 1], self.s[i]]) for i in range(1, self.s.size)]
        self.s = np.insert(self.s, range(1, self.s.size), mean_s)

        mean_x = [np.mean([self.x[i - 1], self.x[i]]) for i in range(1, self.x.size)]
        self.x = np.insert(self.x, range(1, self.x.size), mean_x)

        mean_y = [np.mean([self.y[i - 1], self.y[i]]) for i in range(1, self.y.size)]
        self.y = np.insert(self.y, range(1, self.y.size), mean_y)

        mean_theta = [np.mean([self.theta[i - 1], self.theta[i]]) for i in range(1, self.theta.size)]
        self.theta = np.insert(self.theta, range(1, self.theta.size), mean_theta)

        mean_gamma = [np.mean([self.gamma[i - 1], self.gamma[i]]) for i in range(1, self.gamma.size)]
        self.gamma = np.insert(self.gamma, range(1, self.gamma.size), mean_gamma)

        mean_C = [np.mean([self.C[i - 1], self.C[i]]) for i in range(1, self.C.size)]
        self.C = np.insert(self.C, range(1, self.C.size), mean_C)

        mean_D = [np.mean([self.D[i - 1], self.D[i]]) for i in range(1, self.D.size)]
        self.D = np.insert(self.D, range(1, self.D.size), mean_D)

        self.n = self.s.size
        self.decimateAbove = self.n * 1.1

        self.update_reference_configuration_to_current()


    def adaptive_mesh_update(self, coarsen=True, refine=True, nodes_to_keep=set()):
        """refine and coarsen mesh based on curvature

        :param coarsen:  (Default value = True) Whether to apply coarsening by removing nodes.
        :type coarsen:  bool
        :param refine:  (Default value = True)  Whether to apply refinement by adding nodes.
        :type refine:  bool
        :param nodes_to_keep:  (Default value = set())  A list of nodes that cannot be removed.
        :type nodes_to_keep:  set

        """
        self.verboseprint("Adapting mesh", object(), 1)
        if refine:
            self.refine_mesh()
        if coarsen:
            self.coarsen_mesh(nodes_to_keep=nodes_to_keep)

        # self.get_mesh_spacing()
        self.update_deformed_mesh_spacing()


    def refine_mesh(self, method='spacing'):
        """Refine the mesh by adding nodes in regions of high curvature or low spacing

        :param method:  (Default value = 'spacing')  Method to use. Add mesh nodes based on their spacing or curvature.
        :type method:  string

        """

        self.verboseprint("Refining mesh", object(), 1)

        done = False
        while not done:

            # Get the lengths of the elements
            spacing = self.get_mesh_spacing()

            if method == 'spacing':
                max_spacing = 0.25
                # find the nodes with low density
                sparse_idxs = np.where(spacing > max_spacing)[0]
                sparse_idxs = sparse_idxs[sparse_idxs > 0]
                sparse_idxs = sparse_idxs[sparse_idxs < self.n - 1]

            elif method == 'curvature':
                # max_curvature = 1/50.
                max_curvature = self.kappa * 2000  #\todo make this self.s.size
                max_curv_density = max_curvature * 0.1329  # (.1329 = 2658 / 2000; original circle len / 2000)

                # find the nodes with high curvature density
                sparse_idxs = np.where(self.C * spacing > max_curv_density)[0]
                sparse_idxs = sparse_idxs[sparse_idxs > 0]
                sparse_idxs = sparse_idxs[sparse_idxs < self.n - 1]

            # Insert new nodes before all that were not dense enough
            # S
            mean_s = [np.mean([self.s[i - 1], self.s[i]]) for i in sparse_idxs]
            self.s = np.insert(self.s, sparse_idxs, mean_s)
            # theta
            mean_thetas = [np.mean([self.theta[i - 1], self.theta[i]]) for i in sparse_idxs]
            self.theta = np.insert(self.theta, sparse_idxs, mean_thetas)
            # gamma
            mean_gammas = [np.mean([self.gamma[i - 1], self.gamma[i]]) for i in sparse_idxs]
            self.gamma = np.insert(self.gamma, sparse_idxs, mean_gammas)
            # x
            mean_xs = [np.mean([self.x[i - 1], self.x[i]]) for i in sparse_idxs]
            self.x = np.insert(self.x, sparse_idxs, mean_xs)
            # y
            mean_ys = [np.mean([self.y[i - 1], self.y[i]]) for i in sparse_idxs]
            self.y = np.insert(self.y, sparse_idxs, mean_ys)
            # c
            mean_cs = [np.mean([self.C[i - 1], self.C[i]]) for i in sparse_idxs]
            self.C = np.insert(self.C, sparse_idxs, mean_cs)
            # d
            mean_ds = [np.mean([self.D[i - 1], self.D[i]]) for i in sparse_idxs]
            self.D = np.insert(self.D, sparse_idxs, mean_ds)

            # Update the Lagrange trackers
            largest_val = max(self.lagrangian_point_ids)
            additional_trackers = [i for i in range(largest_val, max(self.lagrangian_point_ids) + len(sparse_idxs))]
            self.lagrangian_point_ids = np.insert(self.lagrangian_point_ids, sparse_idxs, additional_trackers)

            # n
            self.n = self.s.size

            # Don't loop as that can break the lagrange tracking.
            done = True

        # Decimate above
        self.decimateAbove = self.n * 1.1


    def coarsen_mesh(self, method='spacing', nodes_to_keep=set()):
        """Removes nodes in the mesh that are no longer needed.

        :param method:  (Default value = 'spacing')  Method to determine which nodes are removed.  spacing or curvature
        :type method:  string
        :param nodes_to_keep:  (Default value = set())  A set of nodes that can't be removed.
        :type nodes_to_keep:  set

        """

        self.verboseprint("Coarsening mesh", object(), 1)

        keep_going = True
        while keep_going:
            # Get the lengths of cortex elements
            spacing = self.get_mesh_spacing()

            if method == 'spacing':
                # Maximum node spacing.
                max_spacing = 0.1  # 0.05

                # find the nodes with high density (note threshold is different to above)
                # Only include nodes that have sufficiently dense spacing
                indices_to_remove = np.where((spacing < max_spacing))[0]
            elif method == 'curvature':
                # Maximum node spacing.
                max_spacing = 0.1329

                # min_curvature = .1 / 50.
                min_curvature = self.kappa * 20
                min_curv_density = min_curvature * max_spacing

                # find the nodes with low curvature density (note threshold is different to above)
                # Only include nodes that have sufficiently dense spacing
                indices_to_remove = np.where((self.C * spacing < min_curv_density) & (spacing < 3 * max_spacing))[0]

            # Kepp specified indices
            indices_to_remove = [idx for idx in indices_to_remove if idx not in nodes_to_keep]

            # Now remove every element that comes in sequence (i = (i-1_ + 1)) so we don't create large gaps.
            # Group consecutive streams
            grouped_idxs = np.split(indices_to_remove, np.where(np.diff(indices_to_remove) != 1)[0] + 1)
            # Ungroup and every second element of consecutive numbers
            indices_to_remove = [val for sublist in grouped_idxs for val in sublist[::2] if
                                 val != 0 and val != self.s.size - 1]

            # Make sure don't remove too many at once
            if len(indices_to_remove) > self.s.size * self.max_mesh_coarsen_fraction:
                keep_going = False
                number_of_ads_to_pick = int(self.s.size * self.max_mesh_coarsen_fraction)
                prob_of_removal = 1 / spacing[indices_to_remove]
                prob_of_removal /= sum(prob_of_removal)
                indices_to_remove = np.random.choice(indices_to_remove, number_of_ads_to_pick,
                                                     p=prob_of_removal, replace=False)

            if len(indices_to_remove) < 3:
                keep_going = False

            # Filter all indices by the ones to remove
            indices_to_keep = np.arange(self.n)
            indices_to_keep = indices_to_keep[~np.in1d(indices_to_keep, indices_to_remove)]

            # Make sure the boundary nodes are kept
            if 0 not in indices_to_keep:
                indices_to_keep = np.insert(indices_to_keep, 0, 0)
            if self.n - 1 not in indices_to_keep:
                indices_to_keep = np.insert(indices_to_keep, indices_to_keep.size, self.n - 1)

            # Remove nodes that are on straight curves.
            # S
            self.s = self.s[indices_to_keep]
            # theta
            self.theta = self.theta[indices_to_keep]
            # gamma
            self.gamma = self.gamma[indices_to_keep]
            # x
            self.x = self.x[indices_to_keep]
            # y
            self.y = self.y[indices_to_keep]
            # c
            self.C = self.C[indices_to_keep]
            # d
            self.D = self.D[indices_to_keep]
            # Lagrange trackers
            self.lagrangian_point_ids = self.lagrangian_point_ids[indices_to_keep]

            # n
            self.n = self.theta.size

            # Decimate above
            self.decimateAbove = self.n * 1.1

            keep_going = False  # \TODO indices to remove change after initial pruning so can't loop atm.


    def get_mesh_spacing(self):
        """Get the spacing between nodes in the undeformed configuration

        :return:  The current undeformed mesh spacing
        :rtype:  np.array

        """
        s = self.s

        spacing = [s[i + 1] - s[i - 1] for i in range(1, s.size - 1)]
        spacing.insert(0, s[1] - s[0])
        spacing.insert(-1, s[-1] - s[-2])

        return np.array(spacing)


    def update_deformed_mesh_spacing(self, x=None, y=None):
        """Get the spacing between mesh nodes in the deformed configuration

        :param x:  (Default value = None)  x-coordinates to use.  Defaults to stored cortex x variable.
        :type x:  np.array
        :param y:  (Default value = None) y-coordinates to use.  Defaults to stored cortex y variable.
        :type y:  np.array

        """

        self.deformed_mesh_spacing = self.get_xy_segment_lengths(x, y)

        return self.deformed_mesh_spacing


    def get_xy_segment_lengths(self, x=None, y=None):
        """

        :param x:  (Default value = None)  x-coordinates to use.  Defaults to stored cortex x variable.
        :type x:  np.array
        :param y:  (Default value = None) y-coordinates to use.  Defaults to stored cortex y variable.
        :type y:  np.array

        """

        if x is self.get_xy_segment_lengths.__defaults__[0] or y is self.get_xy_segment_lengths.__defaults__[0]:
            x, y = self.x, self.y

        # spacing = [0.5 * geom.LineString(np.dstack((x[i - 1:i + 2], y[i - 1:i + 2]))[0]).length \
        #            for i in range(1, x.size)]
        # spacing.insert(0, 0.5 * geom.LineString(np.dstack((x[0:2], y[0:2]))[0]).length)

        spacing = [0.5 * (np.sqrt((x[i] - x[i - 1]) ** 2 + (y[i] - y[i - 1]) ** 2) +
                          np.sqrt((x[i + 1] - x[i]) ** 2 + (y[i + 1] - y[i]) ** 2))
                   for i in range(-1, x.size - 1)]
        spacing = np.roll(spacing, -1)

        return spacing

    def smooth_variables(self, fac=40, poly_order=1):
        """Apply a savgol filter to smooth the cortex variables

        :param fac:  (Default value = 40)  Factor to determine window size, = num_nodes / fac, for smoothing.
        :type fac:  int
        :param poly_order:  (Default value = 1)  Order of polynomial used for smoothing.
        :type poly_order:  int

        """

        window = int(self.n / fac) if int(self.n / fac) % 2 != 0 else int(self.n / fac) + 1

        self.gamma = savgol_filter((self.gamma, self.s), window, poly_order)[0]
        self.theta, self.x, self.y = savgol_filter((self.theta, self.s), window, poly_order)[0], \
                                     savgol_filter((self.x, self.s), window, poly_order)[0], \
                                     savgol_filter((self.y, self.s), window, poly_order)[0]
        self.C = np.gradient(self.theta, self.s, edge_order=2)
        self.D = np.gradient(self.C, self.s, edge_order=2)


    def interpolate_variable_onto_new_grid(self, variable, current_grid, new_grid):
        """Interpolates a given cortex variable from old_grid to new_grid

        :param variable: The cortex variable (or any array) to be interpolated onto a new grid..
        :type variable: list
        :param current_grid: The current undeformed mesh spacing, S_0.
        :type current_grid: list
        :param new_grid: The new mesh grid that the variable will be interpolated to.
        :type new_grid: list

        """

        # interpolate on the current domain
        var_interp = interp1d(current_grid, variable)

        return var_interp(new_grid)

    def decimate_all_variables_onto_new_grid(self, factor, use_scipy=False, new_s=None):
        """Change the grid and interpolate all variables onto it

        :param factor: Scaling factor for removing nodes.
        :param factor: int
        :param use_scipy:  (Default value = False)  Whether to use Scipy's version of the function.
        :param use_scipy:  bool
        :param new_s:  (Default value = None)  The new undeformed mesh, S_0., which the decimated grid can be interpolated onto.
        :param new_s:  list

        """

        if new_s:
            self.theta = self.interpolate_variable_onto_new_grid(self.theta, self.s, new_s)
            self.gamma = self.interpolate_variable_onto_new_grid(self.gamma, self.s, new_s)
            self.x = self.interpolate_variable_onto_new_grid(self.x, self.s, new_s)
            self.y = self.interpolate_variable_onto_new_grid(self.y, self.s, new_s)
            self.D = self.interpolate_variable_onto_new_grid(self.D, self.s, new_s)
            self.C = self.interpolate_variable_onto_new_grid(self.C, self.s, new_s)
            self.n = new_s.size
            self.s = new_s

        elif use_scipy:
            self.s = decimate(self.s, factor)
            self.theta = decimate(self.theta, factor)
            self.gamma = decimate(self.gamma, factor)
            self.x = decimate(self.x, factor)
            self.y = decimate(self.y, factor)
            self.D = decimate(self.D, factor)
            self.C = decimate(self.C, factor)
            self.n = self.s.size

        else:
            self.s = self.s[::factor]
            self.theta = self.theta[::factor]
            self.gamma = self.gamma[::factor]
            self.x = self.x[::factor]
            self.y = self.y[::factor]
            self.D = self.D[::factor]
            self.C = self.C[::factor]
            self.n = self.s.size


    def upsample_all_variables_onto_new_grid(self, new_n):
        """Change the grid and interpolate all variables onto it

        :param new_n:  The new number of discretised nodes for the cortex variabels.
        :type new_n: int

        """

        self.s = resample(self.s, new_n)
        self.theta = resample(self.theta, new_n)
        self.gamma = resample(self.gamma, new_n)
        self.x = resample(self.x, new_n)
        self.y = resample(self.y, new_n)
        self.D = resample(self.D, new_n)
        self.C = resample(self.C, new_n)
        self.n = new_n


    def create_apposed_cortex(self):
        """Creates an additional cell cortex within this class."""

        return self.__class__()


    def get_length_of_adhesions(self, rerun_distance_calculation=True):
        """ Get the lengths of all connected adhesions.

        :param rerun_distance_calculation:  (Default value = True)  Whether to re-check the lengths of adhesions.
        :param rerun_distance_calculation:  bool
        :return:  The lengths of the adhesions
        :rtype:  np.array

        """
        if rerun_distance_calculation:
            self.update_adhesion_distances_identifiers_and_indices(self.x, self.y, sort_by_distance=True)

        # Get the discrete points on the cortex
        points = np.dstack((self.x, self.y))[0]
        # Get the adhesion points
        chosen_nodes = [i[0] if i else [np.nan, np.nan] for i in self.adhesion_connections]
        chosen_nodes = np.array(chosen_nodes)

        # Calculate distances
        return np.linalg.norm(chosen_nodes - points, axis=1)


    def get_length_of_longest_adhesion(self, rerun_distance_calculation=True):
        """ Length of the longest connected adhesion.

        :param rerun_distance_calculation:  (Default value = True)  Whether to update the adhesions.
        :param rerun_distance_calculation:  bool
        :return:  The length of the longest adhesion.
        :rtype:  float

        """
        return np.nanmax(self.get_length_of_adhesions(rerun_distance_calculation=rerun_distance_calculation))


    def get_length_of_shortest_adhesion(self, rerun_distance_calculation=True):
        """

        :param rerun_distance_calculation:  (Default value = True)  Whether to update the adhesions.
        :param rerun_distance_calculation:  bool
        :return:  The length of the shortest adhesion.
        :rtype:  float

        """
        return np.nanmin(self.get_length_of_adhesions(rerun_distance_calculation=rerun_distance_calculation))


    def get_length(self):
        """Get total length cortex

        :return:  The integrated deformed mesh spacing.
        :rtype:  float
        """
        # Approximate length using shapely.
        myLine = geom.LineString(
            np.dstack((self.x, self.y))[0])

        return myLine.length


    def get_area(self, x=None, y=None):
        """Calculate the signed area of the cell

        :param x:  (Default value = None)  x-coordinates to use.  Defaults to stored cortex x variable.
        :type x:  np.array
        :param y:  (Default value = None) y-coordinates to use.  Defaults to stored cortex y variable.
        :type y:  np.array
        :return:  The area enclosed by the cortex.
        :rtype:  float

        """
        if (x is None) or (y is None):
            x = self.x
            y = self.y

        # Create a shapely polygon
        cell_poly = geom.Polygon(np.dstack((x, y))[0])

        return cell_poly.area


    def get_centroid(self, x=None, y=None):
        """Get the perimeter-based centroid. This is important if the spacing becomes non-uniform then the
        mean of node positions becomes biased to where nodes are most dense.

        :param x:  (Default value = None)  x-coordinates to use.  Defaults to stored cortex x variable.
        :type x:  np.array
        :param y:  (Default value = None) y-coordinates to use.  Defaults to stored cortex y variable.
        :type y:  np.array
        :return:  The (x,y) coords of the centroid.
        :rtype:  tuple

        """

        x = self.x if x is None else x
        y = self.y if y is None else y

        spacing = self.get_xy_segment_lengths()
        x_spacing = x[:-1] + x[1:]
        x_spacing = np.append(x_spacing, [x[0] + x[-1]])
        y_spacing = y[:-1] + y[1:]
        y_spacing = np.append(y_spacing, [y[0] + y[-1]])
        centroid = (0.5 * np.array([np.sum(x_spacing * spacing), np.sum(y_spacing * spacing)])) / self.get_length()

        return centroid


    def get_length_of_cortex_with_active_contractility(self):
        """Get the total length of the cortex that has active contractility

        :return:  The summed cortex segments that have active contractility.
        :rtype:  float
        """

        # Get the prestrain
        prestrains = self.get_prestrains()
        # get lengths of segments
        segments = self.get_xy_segment_lengths()

        return np.sum(segments[prestrains < 1])


    def pickle_self(self, SAVE_DIR=None, name=None):
        """Pickles instance of this class

        :param SAVE_DIR:  (Default value = None)  Location to save.
        :type SAVE_DIR:  string
        :param name:  (Default value = None)  Name of the pickled file.
        :type name:  string

        """

        self.verboseprint("Saving cortex %s" % self.identifier, object(), 1)

        if SAVE_DIR == None:
            SAVE_DIR = CURRENT_DIR
        # Make sure the directory exists
        if not os.path.exists(SAVE_DIR):
            os.makedirs(SAVE_DIR)
        # Filename
        if name == None:
            name = 'cell_' + self.identifier + '_omega0_' + str(self.omega0) + '_' + '_pressure_' + str(self.pressure)

        # saveloc
        saveloc = SAVE_DIR + '/' + name
        # Pickle
        with open(saveloc, 'wb') as s:
            dill.dump(self, s)


    def plot_cortex_variables(self, ax=None, linestyle='-', plot_legend=True, plot_strain=False):
        """Function to plot the data on 3 plot_3panels

        :param ax:  (Default value = None)
        :param linestyle:  (Default value = '-')
        :param plot_legend:  (Default value = True)
        :param plot_strain:  (Default value = False)

        """
        self.verboseprint("Plotting cortex variables for cell %s" % self.identifier, object(), 1)

        if ax is None:
            if plot_strain:
                fig, ax = plt.subplots(2, 1, figsize=(13, 8))
            else:
                fig, ax = plt.subplots(figsize=(12, 5))

        if plot_legend == True:
            labels = [r'$\tilde{c}$', r'$\tilde{\gamma}$', r'$\theta$', r'$\varepsilon = \gamma-1$',
                      r'$\gamma$']
        else:
            labels = ['', '', '', '', '']

        # Get the prestrains
        assert not (self.adhesion_point_coords is None), \
            "Error, must 'update_adhesion_points_between_all_cortices' on eptm."
        prestrain = self.get_prestrains()

        if not plot_strain:
            tau = (1 - self.gamma / prestrain) + self.kappa * self.C * self.C
            ax.plot(self.s, self.D, linestyle, color='C3', label=r'$\tilde{c}^\prime$')
            ax.plot(self.s, self.C, linestyle, color='C0', label=labels[0])
            ax.plot(self.s, tau, linestyle, color='C1', label=r'$\tau$')
            angles = np.degrees(self.theta % (2*np.pi))
            angles[angles > 180] -= 180
            angles[angles > 90] -= 90
            ax.plot(self.s, angles / 100, linestyle, color='C2', label=labels[2])
            ax.grid(alpha=0.5)
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            # Put a legend to the right of the current axis
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        else:
            # ax[0].plot(self.s, np.gradient(self.theta, self.s, edge_order=2), linestyle, color='C4', label="test")
            ax[0].plot(self.s, self.D, linestyle, color='C3', label=r'$\tilde{c}^\prime$')
            ax[0].plot(self.s, self.C, linestyle, color='C0', label=labels[0])
            ax[0].plot(self.s, self.gamma * prestrain, linestyle, color='C1', label=labels[1])
            ax[0].plot(self.s, np.degrees(self.theta) / 100, linestyle, color='C2', label=labels[2])
            ax[0].grid(alpha=0.5)
            box = ax[0].get_position()
            ax[0].set_position([box.x0, box.y0, box.width * 0.8, box.height])
            # Put a legend to the right of the current axis
            ax[0].legend(loc='center left', bbox_to_anchor=(1, 0.5))

            # Get gamma
            g = self.gamma * self.get_prestrains()
            ax[1].plot(self.s, g - 1, linestyle, color='C3', label=labels[3])
            ax[1].plot(self.s, g, linestyle, color='C4', label=labels[4])
            ax[1].set_xlabel(r'$\tilde{S}$')
            ax[1].grid(alpha=0.5)
            box = ax[1].get_position()
            ax[1].set_position([box.x0, box.y0, box.width * 0.8, box.height])
            # Put a legend to the right of the current axis
            ax[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))


    def plot_xy_on_trijunction(self, ax=None, col='k', equalAx=True, plotAdhesion=True,
                               cortexwidth=2, plot_adhesion_forces=True, plot_adhesion_at_specific_locations=False,
                               lagrangian_tracking=False, plot_pressure=False, plot_tension=False,
                               label=None, plot_stress=False, plot_stress_axis=False, sim_type='single',
                               plot_shape=False, label_size=26):
        """Plot the ys x for the junction and reflect/rotate if necessary.

        :param ax:  (Default value = None)
        :param col:  (Default value = 'k')
        :param equalAx:  (Default value = True)
        :param plotAdhesion:  (Default value = True)
        :param cortexwidth:  (Default value = 2)
        :param plot_adhesion_forces:  (Default value = True)
        :param plot_adhesion_at_specific_locations:  (Default value = False)
        :param lagrangian_tracking:  (Default value = False)
        :param plot_pressure:  (Default value = False)
        :param plot_tension:  (Default value = False)
        :param label:  (Default value = None)
        :param plot_stress:  (Default value = False)
        :param plot_stress_axis:  (Default value = False)
        :param sim_type:  (Default value = 'single')
        :param plot_shape:  (Default value = False)
        :param label_size:  (Default value = 26)

        """

        if ax is None:
            f, ax = plt.subplots()

        x, y = self.x, self.y

        # fill the bulk:
        if plot_stress:
            self.plot_stress(ax=ax, plot_stress_axis=plot_stress_axis, sim_type=sim_type)
        elif plot_pressure:
            self.plot_medial_pressure(ax=ax)
        else:
            # col = 'C0'
            face_alpha = 1 if col == "white" else .2
            ax.fill(x, y, col, alpha=face_alpha)

        # cell outline:
        self.plot_cortex(ax=ax, x=x, y=y, plot_tension=plot_tension, cortex_width=cortexwidth, col=col)

        # Principal axes of shape
        if plot_shape:
            self.plot_principal_axes_of_shape(ax=ax)

        # Adhesions
        if plotAdhesion:
            self.plot_adhesion_points(ax=ax, plot_forces=plot_adhesion_forces,
                                      plot_adhesion_at_specific_locations=plot_adhesion_at_specific_locations)

        # Protrusion
        if self.protrusion_force != 0:
            self.plot_protrusion_force(ax=ax, colour=col)

        # Lagrangian points
        if lagrangian_tracking:
            # Scale the colour of the cortex to make slightly darker lagrangian points
            col_scale = 0.75
            try:
                c = matplotlib.colors.cnames[col]
            except:
                c = col
            c = colorsys.rgb_to_hls(*matplotlib.colors.to_rgb(c))
            c = colorsys.hls_to_rgb(c[0], max(0, min(1, col_scale * c[1])), c[2])
            # Plot
            ax.plot(self.x[::20], self.y[::20], 'o', c=c, ms=max(.5, cortexwidth))

        # Plot the cell label
        label = r'$\mathcal{%s}$' % self.identifier if label is None else label
        xshift = 5 if plot_stress_axis else 0
        yshift = 10 if plot_stress_axis else 0
        centroid = self.get_centroid()
        label_pos = centroid if not plot_stress \
            else [centroid[0] + xshift, centroid[1] + yshift]
        kw = dict(xycoords='data', textcoords='data', va="center", ha="center", fontsize=label_size)
        ax.annotate(label, xy=label_pos, **kw)

        # Axes limits
        minX, maxX = ax.get_xlim()
        minY, maxY = ax.get_ylim()
        minY = min(self.y) - self.domain[1] * .1 if min(self.y) - self.domain[1] * .1 < minY else minY
        maxY = max(self.y) + self.domain[1] * .1 if max(self.y) + self.domain[1] * .1 > maxY else maxY
        minX = min(self.x) - self.domain[1] * .1 if min(self.x) - self.domain[1] * .1 < minX else minX
        maxX = max(self.x) + self.domain[1] * .1 if max(self.x) + self.domain[1] * .1 > maxX else maxX
        ax.set_xlim([minX, maxX])
        ax.set_ylim([minY, maxY])

        if equalAx:
            ax.set_aspect('equal', 'box')


    def plot_cortex(self, ax=None, x=None, y=None, plot_tension=False, cortex_width=2, col='k', max_strain = .002):
        """Plot the cortex (outline) of the cell.

        :param ax:  (Default value = None)
        :param x:  (Default value = None)
        :param y:  (Default value = None)
        :param plot_tension:  (Default value = False)
        :param cortex_width:  (Default value = 2)
        :param col:  (Default value = 'k')
        :param max_strain:  (Default value = .002)

        """
        if ax is None:
            f, ax = plt.subplots()

        x = self.x if x is None else x
        y = self.y if y is None else y

        # Get the edges
        edges = [[(x[i], y[i]), (x[i + 1], y[i + 1])] for i in range(0, int(self.n) - 1)]
        if plot_tension:

            if self.adhesion_connections_identities is None:
                self.update_adhesion_distances_identifiers_and_indices()

            # Plot the strain (nondim tension) with a heatmap
            default_cmap = plt.get_cmap('seismic')
            cNorm = matplotlib.colors.Normalize(vmin=-max_strain, vmax=max_strain)
            scaled_cmap = matplotlib.cm.ScalarMappable(norm=cNorm, cmap=default_cmap)
            # Get tangential component of force gradient
            # f_n, tau = self.get_normal_and_tangential_components_of_cortex_forces()
            tau = self.gamma - 1
            colour_cmap = scaled_cmap.to_rgba(tau)
            lc = LineCollection(edges, colors=colour_cmap, linewidths=cortex_width)

            # Add the outline
            if cortex_width < 5:
                lc_outline = LineCollection(edges, colors='k', linewidths=.5, zorder=13)  # Centre line
            else:
                lc_outline = LineCollection(edges, colors='k', linewidths=cortex_width + 1)  # Outline
            ax.add_collection(lc_outline)

            # Plot the prestrain next to the cortex
            prestrains = self.get_prestrains() if len(self.prestrains) != self.s.size else self.prestrains
            if len(prestrains) < self.s.size:
                filler = np.ones(self.s.size - len(prestrains))
                prestrains = np.concatenate((prestrains, filler))
            elif len(prestrains) > self.s.size:
                prestrains = [prestrains[idx] for idx in range(self.x.size)]
            stretch = self.prestrain_plot_offset
            c_x, c_y = np.mean(self.x), np.mean(self.y)
            edges_myo = [[(stretch * (x[i] - c_x) + c_x, stretch * (y[i] - c_y) + c_y),
                          (stretch * (x[i + 1] - c_x) + c_x, stretch * (y[i + 1] - c_y) + c_y)]
                         for i in range(0, int(self.s.size) - 1) if prestrains[i] < 0.9995]

            myo_cbar_lims = 0.065 #  0.01  # 0.065 for gamma = 0.94 # 0.05 old
            cNorm_myo = matplotlib.colors.Normalize(vmin=-myo_cbar_lims, vmax=myo_cbar_lims)
            scaled_cmap_myo = matplotlib.cm.ScalarMappable(norm=cNorm_myo, cmap=default_cmap)
            myo_cols = [scaled_cmap_myo.to_rgba(pre_s - 1) for pre_s in prestrains if pre_s < 0.9995]
            lc_myo = LineCollection(edges_myo, colors=myo_cols, linewidths=cortex_width)
            ax.add_collection(lc_myo)

        else:
            # Plot normal line
            edges_col = 'k' if col == 'white' else col
            lc = LineCollection(edges, colors=edges_col, linewidths=cortex_width, alpha=0.8)

        # Add the edges
        ax.add_collection(lc)


    def plot_adhesion_points(self, ax=None, plot_forces=True, plot_adhesion_at_specific_locations=False):
        """Plot the adhesions on the cortex

        :param ax:  (Default value = None)
        :param plot_forces:  (Default value = True)
        :param plot_adhesion_at_specific_locations:  (Default value = False)

        """
        if ax is None:
            f, ax = plt.subplots()

        # Cell coords
        points = np.dstack((self.x, self.y))[0]

        # This bit plots all fast adhesions at only certain cortex nodes
        if plot_adhesion_at_specific_locations:
            edges = []
            for i in [500, 1165]:
                for node in self.adhesion_connections[i]:
                    edges.append([(node[0], node[1]), (points[i, 0], points[i, 1])])

                dirs = [ad - points[i] for ad in self.adhesion_connections[i]]
                dirs = np.array(dirs)
                dirs = np.divide(dirs, np.sqrt(inner1d(dirs, dirs))[:, np.newaxis])

                angles = np.degrees([np.arctan2(dir[1], dir[0]) for dir in dirs])
                current_tangent = [np.cos(self.x[i + 1] - self.x[i - 1]), np.sin(self.y[i + 1] - self.y[i - 1])]
                current_angle = np.arctan2(current_tangent[1], current_tangent[0])

                # Get spring extension
                d = np.array(self.adhesion_distances[i])
                e = d - self.delta
                force = self.omega0 * e

                # scaling_factors_by_distance using softmax function
                exp_factor = self.adhesion_beta_scale_factor
                dist_scaling = np.exp(-exp_factor * d)
                dist_scaling /= np.sum(dist_scaling)
                force *= dist_scaling

                vector_of_forces = dirs * force[:, np.newaxis]
                vector_force = np.sum(vector_of_forces, axis=0)
                ax.quiver(points[i, 0], points[i, 1], vector_force[0], vector_force[1], width=0.008, scale=.00075,
                          color='r', zorder=11)

                for idx in range(len(dirs)):
                    dir = dirs[idx] * force[idx] * 100000
                    ax.quiver(points[i, 0], points[i, 1], dir[0], dir[1], width=0.0025, scale=5,  # scale=.00010,
                              color='b', zorder=10)

            lc = LineCollection(edges, linewidths=0.5, color='k')
            ax.add_collection(lc)

        # Else, we just plot the nearest fast adhesions, or slow or sdk
        else:
            if plot_forces:
                total_force = -self.get_cortex_forces()
                ax.quiver(points[:, 0], points[:, 1], total_force[:, 0], total_force[:, 1], width=0.0025, scale=.0050,
                          color='k', zorder=10)

            if self.fast_adhesions_active:
                # Get adhesion nodes
                chosen_nodes = [self.adhesion_connections[i][0] if self.adhesion_connections[i] else points[i]
                                for i in range(0, points.shape[0])]
                # Create edges
                edges = [[(chosen_nodes[i][0], chosen_nodes[i][1]), (points[i, 0], points[i, 1])]
                         for i in range(0, points.shape[0])]
                # Colour by forces
                forces = self.get_total_adhesion_force_across_cortex()
                forces = [np.linalg.norm(f) for f in forces]
                cmap = self.make_heatmap_from_array(forces)
                lc = LineCollection(edges, colors=cmap, linewidths=1, zorder=0)
                ax.add_collection(lc)

            # Slow adhesions
            if len(self.slow_adhesions) > 0:
                slow_ad_edges = [np.array([[self.x[ad[0]], self.y[ad[0]]], [ad[1], ad[2]]]) for ad in
                                 self.slow_adhesions]
                # Colour by forces
                distances = [np.linalg.norm(d[1] - d[0]) for d in slow_ad_edges]
                forces = [self.omega0 * (d - 1) for d in distances]
                cmap = self.make_heatmap_from_array(forces)
                lc = LineCollection(slow_ad_edges, colors=cmap, linewidths=1, zorder=0)
                ax.add_collection(lc)

            if len(self.sidekick_adhesions) > 0:
                # Sdk adhesions
                sdk_col = '#8000ff'
                sdk_lw = 2.5
                if len(self.sidekick_adhesions):
                    sdk_ad_edges = [[[self.x[ad[0]], self.y[ad[0]]], [ad[1], ad[2]]] for ad in self.sidekick_adhesions]
                    lc = LineCollection(sdk_ad_edges, colors=sdk_col, linewidths=sdk_lw, zorder=10000, alpha=1)
                    ax.add_collection(lc)

    def plot_medial_pressure(self, ax=None):
        """Plot the adhesions on the cortex

        :param ax:  (Default value = None)

        """
        if ax is None:
            f, ax = plt.subplots()

        pressure = self.pressure

        # Make a heatmap
        max_pressure = 10
        default_cmap = plt.get_cmap('seismic')
        cNorm = matplotlib.colors.Normalize(vmin=-max_pressure, vmax=max_pressure)
        scaled_cmap = matplotlib.cm.ScalarMappable(norm=cNorm, cmap=default_cmap)
        colour_cmap = scaled_cmap.to_rgba(pressure)
        # colour_cmap = matplotlib.cm.seismic(cNorm(self.pressure), bytes=True)
        colour_cmap = matplotlib.colors.to_hex(colour_cmap)

        # Plot the faces
        ax.fill(self.x, self.y, colour_cmap, alpha=1)

        ##########

        # # # Get pressure
        # # if self.identifier in ['B', 'D']:
        # #     pressure = self.pressure
        # # else:
        # #     pressure = -self.pressure
        # pressure = -self.pressure
        #
        # # Figure out which ones have adhesions
        # # Extract the adhesion points
        # adhesionNodes = self.adhesion_point_coords
        #
        # # Get the discrete points on curve
        # slices = 20
        # cortex_x, cortex_y, theta = self.x[::slices], self.y[::slices], self.theta[::slices]
        # points = np.dstack((cortex_x, cortex_y))[0]
        # # Find closest distance to adhesion points
        # dists = cdist(points, adhesionNodes)
        # # #  Smallest distances:
        # smallestDists = [a.min() for a in dists]
        # smallestDists = np.array(smallestDists)
        #
        # # Get the normal
        # n = np.array([-np.sin(theta), np.cos(theta)]) * pressure
        #
        # # Filter only those that have actually joined. (within min dist).
        # n = n[:, smallestDists <= self.max_adhesion_length]
        # xs = cortex_x[smallestDists <= self.max_adhesion_length]
        # ys = cortex_y[smallestDists <= self.max_adhesion_length]
        #
        # # # Also remove the force at the fixed boundaries
        # # n[0, 0:int(self.s.size / 20)] *= 0
        # # n[0, -int(self.s.size / 20):] *= 0
        # # n[1, 0:int(self.s.size / 20)] *= 0
        # # n[1, -int(self.s.size / 20):] *= 0
        # # Plot them
        # # Scale = 100 if scale by spacing, or 2 if not
        # ax.quiver(xs, ys, n[0], n[1], width=0.002, scale=.08, color='#004c00', zorder=10)
        # # ax.plot(np.dstack((self.x, chosen_nodes[:, 0]))[0].T,
        # #         np.dstack((self.y, chosen_nodes[:, 1]))[0].T, '-', lw=.25, c='k')

    def plot_stress(self, plot_stress_axis=False, ax=None, sim_type='single'):
        """Plot the adhesions on the cortex

        :param plot_stress_axis:  (Default value = False)
        :param ax:  (Default value = None)
        :param sim_type:  (Default value = 'single')

        """
        if ax is None:
            f, ax = plt.subplots()

        # Get stress tensor
        stress = self.get_stress_tensor()
        pressure = -0.5 * stress.trace()
        pressure *= -1  # RdBu Colmap has blue for positive so just invert pressure

        # Make a heatmap
        if sim_type == 'single':
            max_pressure = 5e-4  # For singe junction (was 6)
            p_axis_scale = 2.5e4  #  7.5e3  # 50000
        elif sim_type == 'cable':
            max_pressure = 1.2e-3 / 1  # For cables
            p_axis_scale = 7.5e3  # 20000
        elif sim_type == 'whole':
            max_pressure = 4e-3  # For whole cells
            p_axis_scale = 1e1  # 2000
        default_cmap = plt.get_cmap('RdBu')
        #
        # Truncate the colourmap to use lighter colours
        lower, upper = -max_pressure * (5 / 6), max_pressure * (5 / 6)
        minColor = 0.5 - 0.5 * (upper / max_pressure)
        maxColor = 0.5 + 0.5 * (upper / max_pressure)
        truncated_map = self.truncate_colormap(default_cmap, minColor, maxColor)
        # Normalise about new points
        norm = matplotlib.colors.Normalize(lower, upper)
        # Create scaled cmap
        scaled_cmap = matplotlib.cm.ScalarMappable(norm=norm, cmap=truncated_map)
        colour_cmap = scaled_cmap.to_rgba(pressure)
        colour_cmap = matplotlib.colors.to_hex(colour_cmap)

        # Plot the faces
        ax.fill(self.x, self.y, colour_cmap, alpha=1)

        if plot_stress_axis:
            eigvals, eigvecs = np.linalg.eig(stress)
            indexList = np.argsort(-abs(eigvals))  # Abs to account for contractile forces.
            eigvals = eigvals[indexList]
            eigvecs = eigvecs[:, indexList]
            # Make vec2 orthogonal to 1 (hack if they won't be in equilibrium...)
            # eigvecs[:, 1] = np.array([eigvecs[1, 0], -eigvecs[0, 0]])
            evec1, evec2 = eigvecs[:, 0] * eigvals[0], eigvecs[:, 1] * eigvals[1]

            # Get perimeter centroid
            centroid = self.get_centroid()

            evec2 *= p_axis_scale
            evec1 *= p_axis_scale

            # Principal axis
            ax.plot([centroid[0], centroid[0] + evec1[0]], [centroid[1], centroid[1] + evec1[1]],
                    color='black', linewidth=2, alpha=1, zorder=12)
            ax.plot([centroid[0], centroid[0] - evec1[0]], [centroid[1], centroid[1] - evec1[1]],
                    color='black', linewidth=2, alpha=1, zorder=12)
            # Minor axis
            # ax.plot([centroid[0], centroid[0] + evec2[0]], [centroid[1], centroid[1] + evec2[1]],
            #         color='black', linewidth=2, alpha=1, zorder=12)
            # ax.plot([centroid[0], centroid[0] - evec2[0]], [centroid[1], centroid[1] - evec2[1]],
            #         color='black', linewidth=2, alpha=1, zorder=12)

            # Arrows
            # opt = {'length_includes_head': True, 'width': .4, 'head_width': 2.}
            # if pressure < 0:
            #     plt.arrow(centroid[0], centroid[1], evec1[0], evec1[1], alpha=1, fc='k', ec='k', zorder=12, **opt)
            #     plt.arrow(centroid[0], centroid[1], -evec1[0], -evec1[1], alpha=1, fc='k', ec='k', zorder=12, **opt)
            # else:
            #     plt.arrow(centroid[0] - evec1[0], centroid[1] - evec1[1], evec1[0], evec1[1],
            #               alpha=1, fc='k', ec='k', zorder=12, **opt)
            #     plt.arrow(centroid[0] + evec1[0], centroid[1] + evec1[1], -evec1[0], -evec1[1],
            #               alpha=1, fc='k', ec='k', zorder=12, **opt)


    def plot_principal_axes_of_shape(self, ax=None):
        """Plot pricipal axes of shape

        :param ax:  (Default value = None)

        """
        if ax is None:
            f, ax = plt.subplots()

        # Get stress tensor
        shape = self.get_shape_tensor()

        eigvals, eigvecs = np.linalg.eig(shape)
        indexList = np.argsort(-abs(eigvals))  # Abs to account for contractile forces.
        eigvals = eigvals[indexList]
        eigvecs = eigvecs[:, indexList]
        # Make vec2 orthogonal to 1 (hack if they won't be in equilibrium...)
        evec1, evec2 = eigvecs[:, 0] * eigvals[0], eigvecs[:, 1] * eigvals[1]

        centroid = np.array([np.mean(self.x), np.mean(self.y)])

        p_axis_scale = .05
        evec2 *= p_axis_scale
        evec1 *= p_axis_scale
        # Lines
        ax.plot([centroid[0], centroid[0] + evec1[0]], [centroid[1], centroid[1] + evec1[1]],
                color='red', linewidth=2, alpha=1, zorder=12)
        ax.plot([centroid[0], centroid[0] - evec1[0]], [centroid[1], centroid[1] - evec1[1]],
                color='red', linewidth=2, alpha=1, zorder=12)


    def plot_protrusion_force(self, ax=None, colour='C0'):
        """Plot the protrusion on the cortex

        :param ax:  (Default value = None)
        :param colour:  (Default value = 'C0')

        """
        if ax is None:
            f, ax = plt.subplots()

        # Get the protrusion force
        protrusion_forces = self.get_protrusion_force(self.x, self.y)

        # Get index of force
        index = [True if i[0] != 0 and i[1] != 0 else False for i in protrusion_forces]
        if any(index):
            # Get force
            force = protrusion_forces[index][0]

            # Get location
            location = [self.x[index][0], self.y[index][0]]

            # Plot it
            ax.quiver(location[0], location[1], force[0], force[1], scale=10, zorder=10, color=colour)

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
    def make_heatmap_from_array(input_array, cmap='coolwarm'):
        """Given an array of values, generate a heatmap

        :param input_array:  Input array of values to base the cmap on.
        :type input_array: list
        :param cmap:  (Default value = 'coolwarm')  Name of a Matplotlib cmap.
        :type cmap:  sring
        :return:  An rgba colourmap.
        :rtype:  mpl cmap

        """
        # Make a heatmap
        max_val = np.max(input_array)
        # Use a seismic pallette
        cmap1 = plt.get_cmap(cmap, 1000)
        cNorm = matplotlib.colors.Normalize(vmin=-max_val, vmax=max_val)
        cmap = matplotlib.cm.ScalarMappable(norm=cNorm, cmap=cmap1)
        colour_cmap = cmap.to_rgba(input_array)

        return colour_cmap

#
#
#
#
#
#
#
#
