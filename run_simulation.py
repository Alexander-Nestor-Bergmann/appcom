#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Author  : Alexander Nestor-Bergmann
# Released: 08/03/2021
# =============================================================================
"""

Template file that can be used to run all of the simulations published in XXX

The possible choices for active contractlity are:
    - contractility in bicellular junctions.  The contractile junctions are specified by the idienties of the cell pairs
      that share the junction. (cell_prestretch_pairs).
    - Whole-cortex contractility.  Contractility in the entire cell cortex, regardeless of the identity of its
      neighbours.  (whole_prestretch_cells)

There is also functionality to specify changes in adhesion density along junctions and pressure forces from medial
myosin.

"""

import dill
import matplotlib
import os
import time
import numpy as np
from datetime import datetime

matplotlib.rcParams.update({'font.size': 22})

CURRENT_DIR = os.path.dirname(__file__)

# save dir
SAVE_DIR = ''
# Name of file to start from
name = '14_cells'
location = 'pickled_tissues'
# Any additional info wanted to append to the save-filename
name_postfix = ''

############ Set params

# General simulation parameters
verbose = False
num_simulation_steps = 400

############

# Prestretch properties
global_prestretch = 0.9998  # Background contractilty in all cells. Set to 1 if not wanted.
# Junctional contractility
cell_prestretch_pairs = [['A', 'B']]  # Pairs of cell.indentifier e.g. [['A', 'B']]
prestretch = 1 - 0.04  # Prestretch in active juncs
num_junc_prestretch_steps = 1  # In how many steps should we reach max prestretch
conserve_initial_myosin = False  # Keep same amount of myosin, so density increases as shrinking
max_junc_prestretch = 1 - 0.15  # Maximum density that can be held in a junction where it accumuates
unipolar = False  # Apply prestrian to only one side of the apposed cortices?
prestretch_type = 'min'  # how the magnitude of prestrain evaluated, base on neighbour ids.  'min' is neareast neighbour

# Whole-cortex contractility
whole_prestretch_cells = []  # Cells to apply whole-cortex contractility
whole_prestretch = 1 - 0.0  # Prestretch in active juncs
num_whole_prestretch_steps = 1  # In how many steps should we reach max prestretch

############

# Active adhesion properties
# junction-specific adhesion
juncs_to_scale_omega = []  # List of junctions that will have an adhesion scaling
new_junc_omega = 5e-2  # New omega for those junctions
num_junc_adhesion_steps = 1  # How many steps to reach desired scaling

# Whole-cortex adhesion
cells_to_scale_omega = []  # List of cells that will have a new omega
new_whole_omega = 5e-2  # New omega for whole cells
num_whole_adhesion_steps = 1  # How many steps to reach desired scaling

############

# Passive adhesion properties
search_radius = 4  # radius to apply pre_stretch
max_ad_len = 4  # Adhesions longer than this break
max_num_adhesions = 5
use_fast_adhesions = False
use_sidekick_adhesions = False
sidekick_removal_junc_len = 10  # Remove sdk from a junction when its length falls below
use_slow_adhesions = True
slow_ad_lifespan = 10
slow_ad_shear_unbinding_factor = 0  # Increases probability of unbinding with shear angle

############

# Medial myosin properties N.B. Haven't implemented medial myosin in this file
medial_on = False
visc_timescale = 0  # Viscocity of cortex: 0 for junctional simulations.
runaway_junctional = False  # Compound junctional over medial pulses.
max_medial_pressure = 3e-4  # 2e-5 for viscous pulse
positive_pressure_cells = []  # Cell refs to add positive pressure
negative_pressure_cells = []  # Negative pressure

############

# Boundary conditions
boundary_condition = 'fixed'  # = fixed, viscous, germband
boundary_stiffness = 5e-2  # Stiffess of fixed wall.
posterior_pull_shift = 0  # .1  # strength of pull on posterior side.
boundary_strain = 0  # 0.005  # Move the whole boundary

############

# Solving conditions
adaptive = True
max_relaxations = 100
elastic_tol = 1e-2

##########################################################################
####################### Simulation code runs below #######################
##########################################################################

# Load the tissue
open_dir = os.path.join(location, name)
with open(open_dir, 'rb') as s:
    eptm = dill.load(s)

# Set conditions
eptm.set_verbose(verbose)
eptm.set_prestrain_type(prestretch_type)
eptm.activate_fast_adhesions(use_fast_adhesions)
eptm.activate_sidekick_adhesions(use_sidekick_adhesions)
eptm.activate_slow_adhesions(use_slow_adhesions)
eptm.slow_adhesion_lifespan = slow_ad_lifespan
eptm.adhesion_shear_unbinding_factor = slow_ad_shear_unbinding_factor
eptm.set_adhesion_type('fixed_radius')
eptm.set_adhesion_search_radius(search_radius)
eptm.update_all_max_adhesion_lengths(max_ad_len)
eptm.set_max_num_adhesions_for_fast_force_calculation(max_num_adhesions)
eptm.set_pressure_on_off(medial_on)
eptm.max_elastic_relax_steps = max_relaxations
eptm.elastic_relax_tol = elastic_tol
eptm.viscous_timescale = visc_timescale
eptm.boundary_stiffness_x = boundary_stiffness
eptm.boundary_stiffness_y = boundary_stiffness
eptm.posterior_pull_shift = posterior_pull_shift
eptm.boundary_bc = boundary_condition

eptm.use_mesh_refinement = adaptive
eptm.use_mesh_coarsening = adaptive

eptm.update_all_missing_member_variables()
eptm.remove_redundant_member_variables()

#################
# Storing the simulation details in the name
#################

# Apply name postfix
name = '_'.join([name, name_postfix])

# Postfix global prestress
name = '_'.join([name, 'gamma0', str(global_prestretch)])

# Adhesion type
if eptm.cells[0].fast_adhesions_active:
    name = '_'.join([name, 'fast'])
if eptm.slow_adhesions_active:
    name = '_'.join([name, 'slow_ads', str(eptm.slow_adhesion_lifespan)])
    if slow_ad_shear_unbinding_factor > 0:
        name = '_'.join([name, 'shearbonds', str(slow_ad_shear_unbinding_factor)])
if eptm.sidekick_active:
    name = '_'.join([name, 'sdk', 'len', str(sidekick_removal_junc_len)])
# Adhesion info
name = '_'.join([name, 'd0', str(max_ad_len)])
name = '_'.join([name, 'dgamma', str(search_radius)])

# Junction-specific adhesion scaling
if len(juncs_to_scale_omega) > 0:
    junc_cells = '_'.join([''.join(c) for c in juncs_to_scale_omega])
    name = '_'.join([name, 'adhesion_pairs', junc_cells])
    name = '_'.join([name, 'omega', str(new_junc_omega)])
    if num_junc_adhesion_steps > 1:
        name = '_'.join([name, 'steps', str(num_junc_adhesion_steps)])

# Whole-cortex adhesion scaling
if len(cells_to_scale_omega) > 0:
    ad_scaling_cells = '_'.join(cells_to_scale_omega)
    name = '_'.join([name, 'omega_scaling', ad_scaling_cells, str(new_whole_omega)])
    if num_whole_adhesion_steps > 1:
        name = '_'.join([name, 'steps', str(num_whole_adhesion_steps)])

# Junction pairs
if len(cell_prestretch_pairs) > 0:
    junc_cells = '_'.join([''.join(c) for c in cell_prestretch_pairs])
    name = '_'.join([name, 'cell_pairs', junc_cells])
    name = '_'.join([name, 'gamma', str(prestretch)])
    if num_junc_prestretch_steps > 1:
        name = '_'.join([name, 'steps', str(num_junc_prestretch_steps)])
    if unipolar:
        name += '_unipolar'
# Whole-cell contractility
if len(whole_prestretch_cells) > 0:
    whole_cells = '_'.join(whole_prestretch_cells)
    name = '_'.join([name, 'whole_contractility', whole_cells])
    name = '_'.join([name, 'gamma', str(whole_prestretch)])
    if num_whole_prestretch_steps > 1:
        name = '_'.join([name, 'steps', str(num_whole_prestretch_steps)])

# Global boundary stretch
if boundary_condition != 'fixed':
    save_loc = "_".join([name, boundary_condition, 'stiff', str(boundary_stiffness), 'pull', str(boundary_strain)])

# Postfix if adaptive
if not adaptive:
    name = '_'.join([name, 'not_adaptive'])
# Constant density?
if conserve_initial_myosin:
    name = '_'.join([name, 'conserve_myo'])
    initial_length = eptm.get_length_of_shared_junction('A', 'B')

# Save location
SAVE_DIR = os.path.join(SAVE_DIR, name)

# Create the directory if it doesn't exist.
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)
# File storing time taken
open(os.path.join(SAVE_DIR, "times.txt"), 'w').close()

#################
# Finished naming file.  Set up active property lists.
#################

# Add the global prestress
for cell in eptm.cells:
    cell.prestrain_dict = dict.fromkeys(cell.prestrain_dict, global_prestretch)

# Initialise the incremental increase in pretretch
junc_prestretch_list = np.linspace(1, prestretch, num_junc_prestretch_steps + 1)
whole_prestretch_list = np.linspace(1, whole_prestretch, num_whole_prestretch_steps + 1)
# And adhesion
junc_adhesion_list = np.linspace(eptm.cells[0].omega0, new_junc_omega, num_junc_adhesion_steps + 1)
whole_adhesion_list = np.linspace(eptm.cells[0].omega0, new_whole_omega, num_whole_adhesion_steps + 1)

# Initial junction length, used if conserving junc myosin
if conserve_initial_myosin:
    eptm.update_adhesion_points_between_all_cortices(apply_to=['A', 'B'])
    initial_length = eptm.get_length_of_shared_junction('A', 'B')

#################
# Run simulation
#################

for sim_step in range(num_simulation_steps):
    print('simulation step:', sim_step+1)
    t = time.time()

    ########### Active contractility ###########

    # Apply junctional prestretch
    if 0 < sim_step <= num_junc_prestretch_steps:
        current_prestretch = junc_prestretch_list[sim_step]
        for cell_refs in cell_prestretch_pairs:
            # If we are conserving the initial myosin density
            if conserve_initial_myosin:
                eptm.update_adhesion_points_between_all_cortices(apply_to=[cell_refs[0], cell_refs[1]])
                current_junc_len = eptm.get_length_of_shared_junction(cell_refs[0], cell_refs[1])
                current_junc_len = 1e-4 if current_junc_len == 0 else current_junc_len
                current_prestretch = 1 - (1 - current_prestretch) * (initial_length / current_junc_len)
                # Set the maximum myosin that can be on a junction
                current_prestretch = max_junc_prestretch if 1 - current_prestretch > 1 - max_junc_prestretch \
                    else current_prestretch

            # Add the prestrech
            eptm.apply_prestretch_to_cell_identity_pairs(current_prestretch, cell_refs, unipolar=unipolar)

    # Apply whole-cortex prestretch
    if 0 < sim_step <= num_whole_prestretch_steps:
        eptm.apply_prestretch_to_whole_cells(whole_prestretch_list[sim_step], whole_prestretch_cells)

    ########### Active adhesion ###########

    # Apply junctional adhesion
    if 0 < sim_step <= num_junc_adhesion_steps:
        for cell_refs in juncs_to_scale_omega:
            eptm.cellDict[cell_refs[0]].adhesion_density_dict[cell_refs[1]] = junc_adhesion_list[sim_step]
            eptm.cellDict[cell_refs[1]].adhesion_density_dict[cell_refs[0]] = junc_adhesion_list[sim_step]

    # Apply whole-cortex adhesion changes   
    if 0 < sim_step <= num_whole_adhesion_steps:
        for cell in eptm.cells:
            for nabo_ref in cell.adhesion_density_dict.keys():
                if nabo_ref in cells_to_scale_omega:
                    cell.adhesion_density_dict[nabo_ref] = whole_adhesion_list[sim_step]
                    eptm.cellDict[nabo_ref].adhesion_density_dict[cell.identifier] = whole_adhesion_list[sim_step]

    ########### Run simulation step ###########

    eptm.run_simulation_timestep()

    ########### Save ###########

    # current step
    save_name = 'step_%d' % sim_step
    eptm.pickle_self(name=save_name, SAVE_DIR=SAVE_DIR)
    # last animation.
    save_name = 'last_animation'
    eptm.pickle_self(name=save_name, SAVE_DIR=SAVE_DIR)

    # File storing time taken
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    with open(os.path.join(SAVE_DIR, "times.txt"), "a") as times_file:
        times_file.write('time to complete last step:' + str((time.time() - t) / 60.) + " " + current_time + "\n")
        times_file.write('which was' + str(eptm.last_num_internal_relaxes) + "relaxes\n")








