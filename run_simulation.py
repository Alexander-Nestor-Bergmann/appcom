#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Author  : Alexander Nestor-Bergmann
# Released: 08/03/2021
# =============================================================================
#Just a test
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

import os
import time
from datetime import datetime
from typing import List, Tuple

import dill
import matplotlib
import numpy as np

matplotlib.rcParams.update({'font.size': 22})

CURRENT_DIR = os.path.dirname(__file__)

# save dir
SAVE_DIR: str = ''
# Name of file to start from. If name = "continue", it will take the last "step_XX" file in the folder and run as a
# continued simulation.
name: str = '14_cells'
location: str = 'pickled_tissues'
# name: str = 'continue'
# location: str = '14_cells__gamma0_0.9998_slow_ads_1_d0_4_dgamma_4_tau_c_15_medial__AB_0.0003_tau_m_30_loading_AB_max_0.2_min_0.1'
# Any additional info wanted to append to the save-filename
name_postfix: str = ''

############ Set params ############

# General simulation parameters
verbose: bool = False
num_simulation_steps: int = 300
# Solving conditions
max_relaxations: int = 25  # How many elastic relaxation steps to apply in each solving iteration
elastic_tol: float = 1e-2  # Tolerance on much cells have moved to establish equilibrium.
adaptive: bool = True  # Adaptively update the cortex meshes

############

# Timescales (will be normalised to cortex_timescale, so all that matters is value / cortex_timescale)
simulation_timestep: float = 1  # Discretised time to step forward.
cortex_timescale: float = 15  # Turnover time of cortex: 0 for completely viscous simulations.
slow_ad_timescale: int = 15  # Average turnover time (lifespan; sampled from exp distribution) of slow adhesions.
medial_pulse_period: int = 30  # Duration of a medial pulse, from 0 - peak - 0 (no trough).
tau_gamma = 1 * medial_pulse_period  # Prestress de-loading timescale ONLY works for medial loading atm \todo

############

# Prestretch properties
cell_prestretch_pairs: List[Tuple[str, str]] = []  # Pairs of cell.identifier e.g. [('A', 'B')]
global_prestretch: float = 0.9998  # Background contractilty in all cells. Set to 1 if not wanted.
# Junctional contractility
prestretch: float = 1 - 0.3  # Prestretch in active juncs
num_junc_prestretch_steps: int = 40  # In how many steps should we reach max prestretch
prestretch_type: str = 'min'  # how the magnitude of prestrain evaluated, base on neighbour ids.  'min' is neareast neighbour

conserve_initial_myosin: bool = False  # Keep same amount of myosin, so density increases as shrinking
max_junc_prestretch: float = 1 - 0.15  # Maximum density that can be held in a junction where it accumuates

unipolar: bool = False  # Apply prestrian to only one side of the apposed cortices?

# Whole-cortex contractility
whole_prestretch_cells: List[str] = []  # Cells to apply whole-cortex contractility
whole_prestretch: float = 1 - 0.0  # Prestretch in active juncs
num_whole_prestretch_steps: int = 1  # In how many steps should we reach max prestretch

############

# Medial myosin properties
max_medial_pressure: float = 5e-4
antiphase_pressure_cells: List[str] = ['C', 'D']  # Cell refs to add positive pressure
inphase_pressure_cells: List[str] = ['A', 'B']  # Negative pressure
# Params for loading junctions using medial:
# 'max_epsilon_prestretch'/'min...' are the epsilon factors taken from 1 e.g. prestretch = 1 - "epsilon"
# 'junctional_cell_pairs' is list of tuple cell identity pairs to apply to e.g. [('A', 'B')] like cell_prestretch_pairs.
medial_loading_params: dict = {'max_epsilon_prestretch': 0.01, 'min_epsilon_prestretch': 0,
                               'junctional_cell_pairs': [('A', 'B')]}

############

# Passive adhesion properties
search_radius: float = 4  # radius to apply pre_stretch
max_ad_len: float = 4  # Adhesions longer than this break
max_num_adhesions: int = 5  # Maximum num fast adhesions to use for calculation meanfield at each cortex node.
ad_stiffness: float = 0.5  # Stiffness of a (cadherin) adhesion complex.
# Slow adhesions
use_slow_adhesions: bool = True  # Use adhesions that persist for multiple timesteps
slow_ad_shear_unbinding_factor: float = 0  # Increases probability of unbinding with shear angle
# Fast adhesions
use_fast_adhesions: bool = False  # Use a meanfield adhesion force
# Sidekick
use_sidekick_adhesions: bool = False  # Use adhesion that are fixed at vertices
sidekick_removal_junc_len: float = 0  # Remove sdk from a junction when its length falls below
sidekick_stiffness: float = 10

############

# Active adhesion properties
# junction-specific adhesion
juncs_to_scale_omega: List[Tuple[str, str]] = []  # List of junctions that will have an adhesion scaling
new_junc_omega: float = 5e-2  # New omega for those junctions
num_junc_adhesion_steps: int = 1  # How many steps to reach desired scaling

# Whole-cortex adhesion
cells_to_scale_omega: List[str] = []  # List of cells that will have a new omega
new_whole_omega: float = 5e-2  # New omega for whole cells
num_whole_adhesion_steps: int = 1  # How many steps to reach desired scaling

############

# Boundary conditions
boundary_condition: str = 'fixed'  # = fixed, viscous, germband
boundary_stiffness: float = 5e-2  # Stiffness of boundary stencil, used if not fixed.
posterior_pull_shift: float = 0  # .1  # strength of pull on posterior side.
boundary_strain: float = 0  # 0.005  # Move the whole boundary

##########################################################################
####################### Simulation code runs below #######################
##########################################################################

# if we are continuing a file, file the most recent
if name == 'continue':
    print(f"Continuing existing simulation {location}")
    # Get all sim step files in folder
    steps_so_far = [int(f.split("_")[-1]) for f in os.listdir(location) if os.path.isfile(os.path.join(location, f))
                    and "step" in f]
    if len(steps_so_far) == 0:
        raise ValueError("Asked to continue simulation (name = 'continue', but no step_XX files in folder")
    last_simulation_step = max(steps_so_far)
    name = f"step_{last_simulation_step}"
    starting_simulation_step = last_simulation_step + 1
else:
    print("Starting new simulation")
    starting_simulation_step = 0

# Load the tissue
open_dir = os.path.join(location, name)
with open(open_dir, 'rb') as s:
    eptm = dill.load(s)

# Set conditions
eptm.verbose = verbose
eptm.set_prestrain_type(prestretch_type)
eptm.activate_fast_adhesions(use_fast_adhesions)
eptm.activate_sidekick_adhesions(use_sidekick_adhesions)
eptm.set_sdk_stiffness(sidekick_stiffness)
eptm.activate_slow_adhesions(use_slow_adhesions)
eptm.slow_adhesion_lifespan = slow_ad_timescale
eptm.update_adhesion_stiffness_for_cells(ad_stiffness)
eptm.adhesion_shear_unbinding_factor = slow_ad_shear_unbinding_factor
eptm.set_adhesion_type('fixed_radius')
eptm.set_adhesion_search_radius(search_radius)
eptm.update_all_max_adhesion_lengths(max_ad_len)
eptm.set_max_num_adhesions_for_fast_force_calculation(max_num_adhesions)
eptm.max_elastic_relax_steps = max_relaxations
eptm.elastic_relax_tol = elastic_tol
eptm.set_cortical_timescale(cortex_timescale)
eptm.boundary_stiffness_x = boundary_stiffness
eptm.boundary_stiffness_y = boundary_stiffness
eptm.posterior_pull_shift = posterior_pull_shift
eptm.boundary_bc = boundary_condition

eptm.use_mesh_refinement = adaptive
eptm.use_mesh_coarsening = adaptive

#################
# Storing the simulation details in the name
#################

# Apply name postfix
name = '_'.join([name, name_postfix])

# Postfix global prestress
name = '_'.join([name, 'gamma0', str(global_prestretch)])

# Adhesion type
if eptm.cells[0].fast_adhesions_active:
    print("Fast adhesions active")
    name = '_'.join([name, 'fast'])
if eptm.slow_adhesions_active:
    print(f"Slow adhesions active, with timescale {eptm.slow_adhesion_lifespan} and stiffness {ad_stiffness}")
    name = '_'.join([name, 'tau_ad', str(eptm.slow_adhesion_lifespan)])
    if slow_ad_shear_unbinding_factor > 0:
        print(f"\t with shear unbinding factor {slow_ad_shear_unbinding_factor}")
        name = '_'.join([name, 'shearbonds', str(slow_ad_shear_unbinding_factor)])
if eptm.sidekick_active:
    print(f"Sidkekick adhesions active, with stiffness {sidekick_stiffness} and removal len {sidekick_removal_junc_len}")
    name = '_'.join([name, 'sdk', 'len', str(sidekick_removal_junc_len), 'stiff', str(sidekick_stiffness)])
# Adhesion info
name = '_'.join([name, 'omega', str(ad_stiffness)])
name = '_'.join([name, 'd0', str(max_ad_len)])
name = '_'.join([name, 'dgamma', str(search_radius)])

# Junction-specific adhesion scaling
if len(juncs_to_scale_omega) > 0:
    junc_cells = '_'.join([''.join(c) for c in juncs_to_scale_omega])
    print(f"Scaling adhesion strength on {junc_cells} by a factor of {new_junc_omega}")
    name = '_'.join([name, 'adhesion_pairs', junc_cells])
    name = '_'.join([name, 'omega', str(new_junc_omega)])
    if num_junc_adhesion_steps > 1:
        name = '_'.join([name, 'steps', str(num_junc_adhesion_steps)])

# Whole-cortex adhesion scaling
if len(cells_to_scale_omega) > 0:
    ad_scaling_cells = '_'.join(cells_to_scale_omega)
    print(f"Scaling adhesion strength on cells {ad_scaling_cells} by a factor of {new_whole_omega}")
    name = '_'.join([name, 'omega_scaling', ad_scaling_cells, str(new_whole_omega)])
    if num_whole_adhesion_steps > 1:
        name = '_'.join([name, 'steps', str(num_whole_adhesion_steps)])

# Junction pairs
if len(cell_prestretch_pairs) > 0:
    junc_cells = '_'.join([''.join(c) for c in cell_prestretch_pairs])
    print(f"Adding prestress to junctions {junc_cells}, magnitude {prestretch}, in {num_junc_prestretch_steps} steps")
    name = '_'.join([name, 'cell_pairs', junc_cells])
    name = '_'.join([name, 'gamma', str(prestretch)])
    if num_junc_prestretch_steps > 1:
        name = '_'.join([name, 'steps', str(num_junc_prestretch_steps)])
    if unipolar:
        print("\t That's unipolar")
        name += '_unipolar'
    # Constant density?
    if conserve_initial_myosin:
        print("\t Conserving total initial myosin")
        name = '_'.join([name, 'conserve_myo'])
        initial_length = eptm.get_length_of_shared_junction('A', 'B')
# Whole-cell contractility
if len(whole_prestretch_cells) > 0:
    whole_cells = '_'.join(whole_prestretch_cells)
    print(f"Adding prestress to cells {whole_cells}, magnitude {whole_prestretch}, in {num_whole_prestretch_steps} steps")
    name = '_'.join([name, 'whole_contractility', whole_cells])
    name = '_'.join([name, 'gamma', str(whole_prestretch)])
    if num_whole_prestretch_steps > 1:
        name = '_'.join([name, 'steps', str(num_whole_prestretch_steps)])
# Cortex timescale
if cortex_timescale != 0:
    print(f"Cortex is viscoelastic, with timescale {cortex_timescale}")
    name = '_'.join([name, 'tau_c', str(cortex_timescale)])
else:
    print("Cortex is fully viscous")

# Medial myosin
if len(antiphase_pressure_cells) > 0 or len(inphase_pressure_cells) > 0:
    medial_cells = "".join(inphase_pressure_cells + ['_'] + antiphase_pressure_cells)
    print(f"Adding medial pulses to cells {medial_cells}, magnitude {max_medial_pressure}, period {medial_pulse_period}")
    name = '_'.join([name, 'medial', medial_cells, str(max_medial_pressure)])
    name = '_'.join([name, 'tau_m', str(medial_pulse_period)])

    if len(medial_loading_params['junctional_cell_pairs']) > 0:
        junc_cells = '_'.join([''.join(c) for c in medial_loading_params['junctional_cell_pairs']])
        print(f"\t and loading junctional myosin on cells {junc_cells}, from "
              f"{medial_loading_params['min_epsilon_prestretch']} to "
              f"{medial_loading_params['max_epsilon_prestretch']}, with timescale "
              f"{tau_gamma}")
        name = '_'.join([name, 'loading', junc_cells, 'max', str(medial_loading_params['max_epsilon_prestretch']),
                         'min', str(medial_loading_params['min_epsilon_prestretch']), 'tau_g', str(tau_gamma)])

# Global boundary stretch
if boundary_condition != 'fixed':
    save_loc = "_".join([name, boundary_condition, 'stiff', str(boundary_stiffness), 'pull', str(boundary_strain)])

# Postfix if adaptive
if not adaptive:
    name = '_'.join([name, 'not_adaptive'])

# Save location
SAVE_DIR = os.path.join(SAVE_DIR, name) if starting_simulation_step == 0 else os.path.join(SAVE_DIR, location)

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

for sim_step in range(starting_simulation_step, num_simulation_steps):
    print('simulation step:', sim_step)
    t = time.time()

    #######################################################
    ########### Active junctional contractility ###########
    #######################################################

    # Apply junctional prestretch
    if 0 < sim_step <= num_junc_prestretch_steps:
        current_prestretch = junc_prestretch_list[sim_step]

        # Iterate over cell pairs and add prestrain to shared interfaces
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

    #####################################
    ########### Medial myosin ###########
    #####################################

    if len(antiphase_pressure_cells) > 0 or len(inphase_pressure_cells) > 0:
        # Reset the pressures to zero
        eptm.update_pressures(0)
        # Get current magnitude of medial pressure
        position_in_pulse = (sim_step % medial_pulse_period) / medial_pulse_period
        current_medial_pressure = max_medial_pressure * 0.5 * (1 - np.cos(position_in_pulse * 2 * np.pi))
        if sim_step >= 0.5 * medial_pulse_period:
            antiphase_medial_pressure = max_medial_pressure * 0.5 * (1 - np.cos(position_in_pulse * 2 * np.pi - np.pi))
        else:
            antiphase_medial_pressure = 0
        # Add to cells
        eptm.update_pressures(-antiphase_medial_pressure, antiphase_pressure_cells)
        eptm.update_pressures(-current_medial_pressure, inphase_pressure_cells)

        # Medial loading of junctional myosin
        if len(medial_loading_params['junctional_cell_pairs']) > 0:
            # Calculate how medial relates to junctional
            prestretch_scale = (medial_loading_params['max_epsilon_prestretch'] -
                                medial_loading_params['min_epsilon_prestretch']) / max_medial_pressure
            # Add the prestrech on specified junctions
            for cell_pair in medial_loading_params['junctional_cell_pairs']:
                # If no timescale of myosin deloading, just add what's coming from medial
                if tau_gamma == 0:
                    prestretch_magnitude = 1 - (prestretch_scale * current_medial_pressure +
                                                medial_loading_params['min_epsilon_prestretch'])
                # Elif there is a tau_gamma > 0, then: \dot(gamma) = -gamma(t) / tau_gamma + scale * medial
                else:
                    if sim_step == starting_simulation_step:
                        if sim_step == 0:
                            last_epsilon = 0
                        else:
                            eptm.update_adhesion_points_between_all_cortices()
                            last_epsilon = max([max(1 - eptm.cellDict[c].get_prestrains()) for c in cell_pair])
                    # Solve discretised ode for prestress to get next value
                    next_epsilon = last_epsilon * (1 - simulation_timestep / tau_gamma) + \
                        simulation_timestep * prestretch_scale * current_medial_pressure
                    prestretch_magnitude = 1 - next_epsilon
                    # Store
                    last_epsilon = next_epsilon
                # Apply
                eptm.apply_prestretch_to_cell_identity_pairs(prestretch_magnitude, cell_pair, unipolar=unipolar)

    #######################################
    ########### Active adhesion ###########
    #######################################

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

    ###########################################
    ########### Run simulation step ###########
    ###########################################

    eptm.run_simulation_timestep(dt=simulation_timestep)

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
        times_file.write('which was ' + str(eptm.last_num_internal_relaxes) + " relaxes\n")
