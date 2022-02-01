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

import os
import time
from datetime import datetime
from typing import List, Tuple
from os import listdir
from os.path import isfile, join
import dill
import matplotlib
import numpy as np
import re
import matplotlib.pyplot as plt

matplotlib.rcParams.update({'font.size': 22})


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]


CURRENT_DIR = os.path.dirname(__file__)

# save dir
SAVE_DIR: str = ''
# Name of file to start from
name: str = 'step_0'
location: str = 'pickled_tissues'
# Any additional info wanted to append to the save-filename
name_postfix: str = 'Sim_110'
txt_suffix: str = '.txt'
result_file_name = "".join((name_postfix, txt_suffix))

##### Recuperation mode #####
Recuperation_mode = False
Last_step = -1  # initialisation of Last step
###############################
##### Run parameters ##########
tolerance = 1e-4
relaxes = 200
cortex_time = 0  # np.infty
adhesion_time = 0  # np.infty
mm1 = 0
mm2 = -10
ad_len = 10
############ Set params

# General simulation parameters
verbose: bool = True
num_simulation_steps: int = 250
serious = True

############### Specific parameters for periodic case
M1: float = mm1  # multiplicator of stretch
M2: float = mm2  # num_simulation_steps/30 #multiplicator of shear
maximum_number_of_steps = num_simulation_steps
ratio: float = 1  # 1/np.sqrt(5) #ratio of m1<1 :m1>1
tile_type: str = 'Parallelogram'  # Parallelogram or Rhombus
Phi: float = 0  # phi the angle of the transformation in degrees
############

# Prestretch properties
cell_prestretch_pairs: List[Tuple[str, str]] = []  # Pairs of cell.identifier e.g. [('A', 'B')]
global_prestretch: float = 0.9998  # Background contractilty in all cells. Set to 1 if not wanted.
# Junctional contractility
prestretch: float = 1 - 0.04  # Prestretch in active juncs
num_junc_prestretch_steps: int = 1  # In how many steps should we reach max prestretch
conserve_initial_myosin: bool = False  # Keep same amount of myosin, so density increases as shrinking
max_junc_prestretch: float = 1 - 0.15  # Maximum density that can be held in a junction where it accumuates
unipolar: bool = False  # Apply prestrian to only one side of the apposed cortices?
prestretch_type: str = 'min'  # how the magnitude of prestrain evaluated, base on neighbour ids.  'min' is neareast neighbour

# Whole-cortex contractility
whole_prestretch_cells: List[str] = []  # Cells to apply whole-cortex contractility
whole_prestretch: float = 1  # Prestretch in active juncs
num_whole_prestretch_steps: int = 1  # In how many steps should we reach max prestretch

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

# Passive adhesion properties
search_radius: float = 4  # radius to apply pre_stretch
max_ad_len: float = ad_len  # Adhesions longer than this break
max_num_adhesions: int = 5  # Maximum num fast adhesions to use for calculation meanfield at each cortex node.
use_fast_adhesions: bool = False  # Use a meanfield adhesion force
use_sidekick_adhesions: bool = False  # Use adhesion that are fixed at vertices
sidekick_removal_junc_len: float = 6  # Remove sdk from a junction when its length falls below
use_slow_adhesions: bool = True  # Use adhesions that persist for multiple timesteps
slow_ad_lifespan: int = adhesion_time  # Average max age of slow adhesions
slow_ad_shear_unbinding_factor: float = 0  # Increases probability of unbinding with shear angle

############

# Timescales
cortex_timescale: int = cortex_time  # Viscosity of cortex: 0 for junctional simulations.

############

# Medial myosin properties
max_medial_pressure: float = 50e-4
medial_pulse_period: int = 2000000  # Duration of a medial pulse, from 0 - peak - 0
positive_pressure_cells: List[str] = []  # Cell refs to add positive pressure
negative_pressure_cells: List[str] = []  # Negative pressure
negative_and_positive_cycle: bool = False  # Make cells go positive and negative.

############

# Boundary conditions
boundary_condition: str = 'periodic'  # = fixed, viscous, germband
boundary_stiffness: float = 1e-2  # Stiffness of fixed wall.
posterior_pull_shift: float = 0  # .1  # strength of pull on posterior side.
boundary_strain: float = 0  # 0.005  # Move the whole boundary

############

# Solving conditions
adaptive: bool = False  # Adaptively update the mesh
if serious == True:
    max_relaxations: int = relaxes  # How many elastic relaxation steps to apply in each solving iteration
    elastic_tol: float = tolerance  # Tolerance on much cells have moved to establish equilibrium.
if serious == False:
    max_relaxations: int = 25  # How many elastic relaxation steps to apply in each solving iteration
    elastic_tol: float = 1e-2  # Tolerance on much cells have moved to establish equilibrium.

##########################################################################
####################### Simulation code runs below #######################
##########################################################################

# Load the tissue
if Recuperation_mode == False:
    open_dir = os.path.join(location, name)
    with open(open_dir, 'rb') as s:
        eptm = dill.load(s)

# Apply name postfix
name = '_'.join([name, name_postfix])
#### Read the file in name and get me the last step

if Recuperation_mode:
    Last_animation = "last_animation"
    print('\n Rami the recuperation mode is ON')
    cwd = os.getcwd()
    newlocation = cwd
    name_recup = name
    newlocation = "/".join((newlocation, name_recup))
    print(' \nRami the new location is :' + str(newlocation))
    open_dir = os.path.join(newlocation, Last_animation)
    with open(open_dir, 'rb') as s:
        eptm = dill.load(s)
    slash = '/'
    name = "".join((name_recup, slash))
    List_files = [f for f in listdir(name) if isfile(join(name, f)) and f.endswith('.txt') == False]
    index = List_files.index('last_animation')
    del List_files[index]
    step_list = List_files
    step_list.sort(key=natural_keys)
    print(step_list)
    List_of_norm = []
    for idx, item in enumerate(step_list):
        item = "/".join((name_recup, item))
        file_in = open(item, 'rb')
        sim_in = dill.load(file_in)
        print(' ' + str(item) + ':   ' + str(sim_in.final_dist_norm))
        List_of_norm.append(sim_in.final_dist_norm)
    print(step_list)
    max_norm = max(List_of_norm)
    max_index = List_of_norm.index(max_norm)
    print(f"The step with the maximum norm is {max_index} ")
    step_max = int(step_list[-1].split('_')[1])
    Last_step = step_max + 1
    affiche_last_step = step_max + 2
    new_name = "_".join(('step', str(affiche_last_step - 1)))
    new_name = os.path.join(newlocation, new_name)
    os.rename(open_dir, new_name)
    print('\n Rami the last step was :' + str(affiche_last_step))

# Set conditions

eptm.verbose = verbose
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
eptm.max_elastic_relax_steps = max_relaxations
eptm.elastic_relax_tol = elastic_tol
eptm.set_cortical_timescale(cortex_timescale)
eptm.boundary_stiffness_x = boundary_stiffness
eptm.boundary_stiffness_y = boundary_stiffness
eptm.posterior_pull_shift = posterior_pull_shift
eptm.boundary_bc = boundary_condition

eptm.use_mesh_refinement = adaptive
eptm.use_mesh_coarsening = adaptive

# Specific parameters for periodic case
eptm.m1 = M1
eptm.m2 = M2
eptm.maximum_number_of_steps = maximum_number_of_steps
eptm.ratio = ratio
eptm.tile_type = tile_type  # Parallelogram or Rhombus
eptm.phi = Phi
eptm.name = name
#################
# Storing the simulation details in the name
#################


##########################
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
Last_step = Last_step + 1
# print('HEre the LAst step' + str(Last_step))
for sim_step in range(Last_step, num_simulation_steps):
    print('simulation step:', sim_step, flush=True)
    print('\n\n ################# Simulation name :' + str(name_postfix) + '#################', flush=True)
    t = time.time()
    # Periodic case only
    # check equilibrium at first step
    # get the first transformation
    if boundary_condition == 'periodic':
        if sim_step == Last_step:
            eptm.solve_bvps_in_parallel()
            First_trans = eptm.set_a_first_transformation()
            eptm.trans = First_trans

            print('The first transformation is :  ' + str(First_trans))
        # I=eptm.get_initial_area_and_energy()
        # with open(os.path.join(SAVE_DIR, "times_result_initials.txt"), "a") as result_file:
        #	result_file.write(str(I[0]) + ";" + str(I[1])+ ";" + str(M1) +  ";" + str(M2)+"\n")

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

    ########### Medial myosin ###########

    if len(positive_pressure_cells) > 0 or len(negative_pressure_cells) > 0:
        # Reset the pressures to zero
        eptm.update_pressures(0)
        # Get current magnitude of medial pressure
        position_in_pulse = (sim_step % medial_pulse_period) / medial_pulse_period
        current_medial_pressure = max_medial_pressure * 0.5 * (1 - np.cos(position_in_pulse * 2 * np.pi))
        # Add to cells
        eptm.update_pressures(current_medial_pressure, positive_pressure_cells)
        eptm.update_pressures(-current_medial_pressure, negative_pressure_cells)

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

    T = eptm.run_simulation_timestep(step_num=sim_step)

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
    with open(os.path.join(SAVE_DIR, str(result_file_name)), "a") as result_file:
        result_file.write(str(T[0]) + ";" + str(T[1]) + ";" + str(T[2]) + ";" + str(T[3]) + ";" + str(T[4]) + ";" + str(
            T[5]) + ";" + str(T[6]) + "\n")
    with open(os.path.join(SAVE_DIR, "times.txt"), "a") as times_file:
        times_file.write('time to complete last step:' + str((time.time() - t) / 60.) + " " + current_time + "\n")
        times_file.write('which was' + str(eptm.last_num_internal_relaxes) + "relaxes\n")
        # Junctional contractility
        times_file.write('############ Junctional contractility #################' + "\n" + "\n")
        times_file.write('List of pairs on prestretch=' + " " + str(cell_prestretch_pairs) + "\n")
        times_file.write('Global prestretch=' + " " + str(global_prestretch) + "\n")
        times_file.write('Prestretch=' + " " + str(global_prestretch) + "\n")
        times_file.write('Number of steps to reach max prestretch=' + " " + str(num_junc_prestretch_steps) + "\n")
        times_file.write('Conserve initial myosin=' + " " + str(conserve_initial_myosin) + "\n")
        times_file.write('Maximum density that can be held in a junction where it accumuates=' + " " + str(
            max_junc_prestretch) + "\n")
        times_file.write('Apply prestrian to only one side of the apposed cortices?=' + " " + str(unipolar) + "\n")
        times_file.write(
            'how the magnitude of prestrain evaluated, base on neighbour ids. =' + " " + str(prestretch_type) + "\n")
        # Whole-cortex contractility
        times_file.write('############ Whole cortex contractility ###############' + "\n" + "\n")
        times_file.write('Cells to apply whole cortex contractility=' + " " + str(whole_prestretch_cells) + "\n")
        times_file.write('Prestrectch in active juncs=' + " " + str(whole_prestretch) + "\n")
        times_file.write('Number of steps to reach max prestretch=' + " " + str(num_whole_prestretch_steps) + "\n")
        # Active adhesion properties
        times_file.write('############ Active adhesion properties ##############' + "\n" + "\n")
        # junction-specific adhesion
        times_file.write('############ Junction-Specific Adhesion ##############' + "\n")
        times_file.write(
            'List of junctions that will have an adhesion scaling=' + " " + str(juncs_to_scale_omega) + "\n")
        times_file.write('New omega for those junctions=' + " " + str(new_junc_omega) + "\n")
        times_file.write('How many steps to reach desired scaling=' + " " + str(num_junc_adhesion_steps) + "\n")
        # Whole-cortex adhesion
        times_file.write('############ Whole-Cortex Adhesion ##############' + "\n")
        times_file.write('List of cells that will have a new omega=' + " " + str(cells_to_scale_omega) + "\n")
        times_file.write('New omega for whole cells=' + " " + str(new_whole_omega) + "\n")
        times_file.write('How many steps to reach desired scaling=' + " " + str(num_whole_adhesion_steps) + "\n")
        # Passive adhesion properties
        times_file.write('############Passive adhesion properties ##############' + "\n" + "\n")
        times_file.write('Radius to apply pre_stretch=' + " " + str(search_radius) + "\n")
        times_file.write('Adhesions longer than this break=' + " " + str(max_ad_len) + "\n")
        times_file.write('Maximum num fast adhesions to use for calculation meanfield at each cortex node=' + " " + str(
            max_num_adhesions) + "\n")
        times_file.write('Use a meanfield adhesion force=' + " " + str(use_fast_adhesions) + "\n")
        times_file.write('Use adhesion that are fixed at vertices=' + " " + str(use_sidekick_adhesions) + "\n")
        times_file.write(
            'Remove sdk from a junction when its length falls below=' + " " + str(sidekick_removal_junc_len) + "\n")
        times_file.write('Use adhesions that persist for multiple timesteps=' + " " + str(use_slow_adhesions) + "\n")
        times_file.write('Average max age of slow adhesions=' + " " + str(slow_ad_lifespan) + "\n")
        times_file.write(
            'Increases probability of unbinding with shear angle=' + " " + str(slow_ad_shear_unbinding_factor) + "\n")
        # Timescales
        times_file.write('############ Timescales ##############' + "\n" + "\n")
        times_file.write('Viscosity of cortex: 0 for junctional simulations=' + " " + str(cortex_timescale) + "\n")
        # Medial myosin properties
        times_file.write('############ Medial myosin properties ##############' + "\n" + "\n")
        times_file.write('Maximum medial pressure=' + " " + str(max_medial_pressure) + "\n")
        times_file.write('Duration of a medial pulse, from 0 - peak - 0=' + " " + str(medial_pulse_period) + "\n")
        times_file.write('Cell refs to add positive pressure=' + " " + str(positive_pressure_cells) + "\n")
        times_file.write('Negative pressure=' + " " + str(negative_pressure_cells) + "\n")
        times_file.write('Make cells go positive and negative=' + " " + str(negative_and_positive_cycle) + "\n")
        # Boundary conditions
        times_file.write('############ Boundary conditions ##############' + "\n" + "\n")
        times_file.write('Boundary conditions=' + " " + str(boundary_condition) + "\n")
        times_file.write('Stiffness of fixed wall=' + " " + str(boundary_stiffness) + "\n")
        times_file.write('Strength of pull on posterior side=' + " " + str(posterior_pull_shift) + "\n")
        times_file.write('Move the whole boundary=' + " " + str(boundary_strain) + "\n")
        # Solving conditions
        times_file.write('############ Solving conditions ##############' + "\n" + "\n")
        times_file.write('Adaptively update the mesh=' + " " + str(adaptive) + "\n")
        times_file.write(
            'How many elastic relaxation steps to apply in each solving iteration=' + " " + str(max_relaxations) + "\n")
        times_file.write('Tolerance on much cells have moved to establish equilibrium=' + " " + str(elastic_tol) + "\n")









