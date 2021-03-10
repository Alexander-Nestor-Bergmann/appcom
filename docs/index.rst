.. AppCoM documentation master file, created by
   sphinx-quickstart on Mon Mar  8 08:53:13 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to AppCoM's documentation!
==================================

The **App**\ osed-**Co**\ rtex **M**\ odel library provides an interface to implement a
biomechanical model of an epithelial tissue.  In this model, each cell cortex is represented
as an active, continuum morphoelastic rod with resistance to bending and extension.  Adhesions
are modelleld as agents explicitly coupling neighbouring cell cortices.

.. note::
    Take a look at the theoretical development of the model here: TBA.

Quickstart demo
===============

Load a tissue with 14 cells:

.. code-block:: python
    :linenos:

    with open('initial_conditions/14_cells', 'rb') as new_tissue:
        eptm = dill.load(new_tissue)


Update the adhesions (passing the connectivity data to the cells) and apply some active
contractility (a prestretch) on the bicellular junction shared by cells A and B:

.. code-block:: python
    :linenos:

    eptm.update_adhesion_points_between_all_cortices()
    prestrech_magnitude = 1 - 0.01
    eptm.apply_prestretch_to_cell_identity_pairs(prestrech_magnitude, [['A','B']])


Perform 10 simulation timesteps (update restlengths and relax to equilibrium) and save the 
result:

.. code-block:: python
    :linenos:

    num_timesteps = 10
    for time_step in range(num_timesteps):
        eptm.run_simulation_timestep()
    eptm.pickle_self(name='my_test_file')

Have a look at the result, colouring the cortex with tension and bulk with cell pressure:

.. code-block:: python
    :linenos:

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 9))
    eptm.plot_xy_on_trijunction(ax=ax, plot_stress=True, plot_tension=True)
    plt.show()


More coming soon!


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   modules

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
