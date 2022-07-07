import matplotlib.pyplot as plt
import numpy as np
import shapely.geometry as geom

from scipy.spatial import Voronoi

from src.cell_class import Cell
from src.eptm_class import Epithelium

#### \START: DEMO, a way to get cell boundaries ####

# Specify number of cells
NUM_CELLS: int = 20
NODE_SPACING: float = 0.2
TISSUE_WIDTH = 100

# Create a random set of voronoi cells
random_seed_points = np.random.rand(NUM_CELLS, 2) * TISSUE_WIDTH
vor = Voronoi(random_seed_points)

# # For plotting, if wanted
# f, ax = plt.subplots()

# Loop over each voronoi region and build the cell boundary
cell_boundaries = []
for region in vor.regions:
    # Finite regions only
    if -1 not in region and len(region) > 2:
        cell_vertices = np.array([vor.vertices[i] for i in region])
        # Make sure within boundary
        if (cell_vertices < 0).any() or (cell_vertices > TISSUE_WIDTH).any():
            continue

        # create a polygon from the vertices
        cell_poly = geom.LinearRing(cell_vertices)
        # ax.plot(*cell_poly.xy, '-')  # Have a look

        # Interpolate equally spaced points around the boundary
        num_discretisation_nodes = int(cell_poly.length / NODE_SPACING)
        interpolation_points = np.linspace(0, cell_poly.length, num_discretisation_nodes)
        discretised_cortex = [(cell_poly.interpolate(p).xy[0][0], cell_poly.interpolate(p).xy[1][0])
                              for p in interpolation_points]
        # for nodes in discretised_cortex:
        #     ax.plot(*nodes, 'o')

         # Store the coordinates
        cell_boundaries.append(discretised_cortex)
# plt.show()

#### \END: DEMO, a way to get cell boundaries ####

######################################################

# Given lists of coords defining the boundary for each cell [(x1, y1), (x2, y2),... ], construct tissues with
# singe cells within these boundaries

# We will make a list of temporary epithelia, each with 1 of these cells. Then we'll zip each cell up and create the
# 'final' epithelium, which contains all of the cells from these temporary epithelia.

temp_epithelia = []
for idx, cell_boundary in enumerate(cell_boundaries):
    # Create a shapely polygon, which has some useful functions
    cell_poly = geom.Polygon(cell_boundary)

    # Calculate how big the cell can be by getting the distance from the centroid to the boundary
    min_dist_to_boundary = cell_poly.centroid.distance(cell_poly.exterior)
    # The cell radius is assumed to be for a hexagon, so convert that
    cell_radius = min_dist_to_boundary / (np.sqrt(3) * 0.5)

    # Create a new tissue, with a single cell in it
    mini_eptm = Epithelium(tissue_type='from_stencil')  #  I think that's not implemented in current code
    # ### \START NOTE If you can't do 'from_stencil' (as I suspect) do this:
    # mini_eptm = Epithelium(tissue_type='within_hexagon')
    # mini_eptm.cells = []
    # mini_eptm.cellDict = {}
    # ### \END NOTE
    # Build the cell in the tissue
    mini_eptm._build_single_cell(radius=cell_radius, n=len(cell_boundary), verbose=False, identifier=f"{idx}")
    # The cell defaults to (0,0), so translate it to the correct location
    mini_eptm.cells[-1].x += cell_poly.centroid.coords.xy[0]
    mini_eptm.cells[-1].y += cell_poly.centroid.coords.xy[1]

    # Give the epithelium it's boundary stencil, which is the original cell boundary
    cell_boundary_x, cell_boundary_y = np.array(list(zip(*cell_boundary)))
    # The boundary is actually coded as another cell -- the 'boundary cell'.  Build that.
    boundary_cell_dict = {'D': [], 'C': [], 'gamma': [], 'theta': [], 'x': cell_boundary_x, 'y': cell_boundary_y}
    mini_eptm.boundary_cell = Cell(initial_guesses=boundary_cell_dict, param_dict=mini_eptm.param_dict,
                                   identifier=f"boundary")
    mini_eptm.boundary_cell.update_deformed_mesh_spacing()
    mini_eptm.cellDict['boundary'] = mini_eptm.boundary_cell
    mini_eptm.boundary_coordinates = np.array([cell_boundary_x, cell_boundary_y])
    # Make sure the cell is at least distance 1 away from the boundary
    # mini_eptm.scale_cells_to_be_within_global_boundary_stencil()

    #  Store the temporary epithelium to the list.
    temp_epithelia.append(mini_eptm)

# Plot if you want to see them
f, ax = plt.subplots()
for eptm in temp_epithelia:
    eptm.plot_self(ax=ax)
plt.show()

######################################################

# Now we have a list of epithelia with 1 cell in them.  We will fit each one separately and then aggregate them in
# a new tissue.

#  First, se the parameters for each cell
for eptm in temp_epithelia:

    # Define the stiffness of the cortex
    eptm.cells[0].kappa = 1e-4
    # Purely visccous cortex (so rest length updates to dissipate all stretching energy after every relax step)
    eptm.set_cortical_timescale(0)

    # Adhesion properties
    # Use 'slow' adhesions only.
    eptm.activate_fast_adhesions(False)
    eptm.activate_slow_adhesions(True)
    # But get a fresh set of adhesions between relax steps
    eptm.slow_adhesion_lifespan = 0
    # Allow the adhesions to be long, because the cell is far from the boundary right now.
    eptm.set_adhesion_search_radius(50)
    eptm.update_all_max_adhesion_lengths(1e4)

    # How many times to try to get to elastic equilibrium per relax step
    eptm.max_elastic_relax_steps = 10
    # Equilibrium tolerance
    eptm.elastic_relax_tol = 1e-2

    # Update geometric properties.
    eptm.update_all_rest_lengths_and_areas()

# define the steps in adhesion stiffness, starting small to help the solver and then increasing to the final value
ADHESION_STIFFNESS_INCREMENTS = [1e-5, 1e-4, 1e-3, 1e-2]
NUM_RELAXES_BETWEEN_ADHESION_INCREASE = 30
RELAXED_EPTM_SAVE_LOCATION = 'tissue_fitting/temporary_eptms'

# Now, loop over each cell and fit it to the boundary (THIS CAN BE PARALLELISED)
for eptm in temp_epithelia:

    # Zip up adhesions, gradually increasing
    for omega in ADHESION_STIFFNESS_INCREMENTS:
        eptm.cells[0].omega0 = omega

        # Keep relaxing for a certain number of timesteps (this performs viscous length update at the end)
        for relax_step in range(NUM_RELAXES_BETWEEN_ADHESION_INCREASE):
            eptm.run_simulation_timestep()
            # Make sure the cells didn't move out of the boundary
            eptm.scale_cells_to_be_within_global_boundary_stencil()
            # If we were successful, save the intermediate step to return to incase it's disrupted
            if eptm.last_solve_success:
                cell = eptm.cells[0]
                eptm.pickle_self(name=f'{cell.identifier}_{cell.omega0}', SAVE_DIR=RELAXED_EPTM_SAVE_LOCATION)

######################################################

# Finally, create a new tissue, built  from all of the temporary ones
final_tissue = Epithelium(tissue_type='from_stencil')
final_tissue.cells = []
final_tissue.cellDict = {}
for eptm in temp_epithelia:

    cell_to_add = eptm.cells[-1]

    final_tissue.cells.append(cell_to_add)
    final_tissue.cellDict[cell_to_add.identifier] = cell_to_add

# Then create the boundary of the eptm, which could  be a concave hull of all of the cells.
### ... left for you


