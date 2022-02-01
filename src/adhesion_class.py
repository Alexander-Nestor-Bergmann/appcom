#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Author  : Alexander Nestor-Bergmann
# Released: 08/03/2021
# =============================================================================
"""Implementation of a class to represent the cell-cell adhesions."""

from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from src.cell_class import Cell

class Adhesion(object):
    """ A class to represent an adhesion agent in a tissue.  This is a useful object to keep track of which cells the
    adhesion is attached to and the coordinates and locations in the deformed and undeformed cortex configurations.
    """

    def __init__(self, cells: List[Cell], s_coords: List[int], average_lifespan: int,
                 adhesion_type: str = 'cadherin'):
        self.cell_1: Cell = cells[0]
        self.cell_2: Cell = cells[1]
        self.cell_1_s: int = s_coords[0]
        self.cell_2_s: int = s_coords[1]

        self.update_local_cell_indices_with_s()

        self.adhesion_type: str = adhesion_type

        if adhesion_type == 'cadherin':
            self.delta: float = min([self.cell_1.delta, self.cell_2.delta])
            self.omega: float = min([self.cell_1.omega0, self.cell_2.omega0])
        elif adhesion_type == 'sidekick':
            self.delta: float = min([self.cell_1.sdk_restlen, self.cell_2.sdk_restlen])
            self.omega: float = min([self.cell_1.sdk_stiffness, self.cell_2.sdk_stiffness])
        else:
            raise NotImplementedError('"adhesion_type" must be cadherin or sidekick')

        self.max_bonding_length: float = min([self.cell_1.max_adhesion_length, self.cell_2.max_adhesion_length])

        self.age: int = 0
        self.lifespan: int = np.random.exponential(average_lifespan)

    def update_local_cell_indices_with_s(self):
        """Updates the local cortex indices by the s values for the cells.
        """
        # self.cell_1_index = self.cell_1.s_index_dict[self.cell_1_s]
        # self.cell_2_index = self.cell_2.s_index_dict[self.cell_2_s]
        self.cell_1_index = np.where(self.cell_1.s == self.cell_1_s)[0][0]
        self.cell_2_index = np.where(self.cell_2.s == self.cell_2_s)[0][0]

    def update_s_by_local_cell_indices(self):
        """Updates the local cortex indices by the s values for the cells.
        """
        self.cell_1_s = self.cell_1.s[self.cell_1_index]
        self.cell_2_s = self.cell_2.s[self.cell_2_index]

    def cell_index_by_id(self, id):
        """ Get the local index on a cortex that the adhesion is attached to.

        :param id: The identity of the cell to look on.
        :type id: string
        :return: The local index that the adhesion is connected to for the given cell.
        :rtype: string

        """
        if id == self.cell_1.identifier:
            # return self.cell_1.s_index_dict[self.cell_1_s]
            return self.cell_1_index
        elif id == self.cell_2.identifier:
            # return self.cell_2.s_index_dict[self.cell_2_s]
            return self.cell_2_index
        else:
            raise ValueError('given cell does not belong to adhesion (should be cell id)')
        # return self.cell_dict[id].s_index_dict[self.cell_2_s]

    def get_xy(self):
        """Get the [(x_1, y_1), (x_2, y_2)] coords of the cell connections

        :return: The coords of the adhesion connections.
        :rtype: list

        """
        return [[self.cell_1.x[self.cell_1_index], self.cell_1.y[self.cell_1_index]],
                [self.cell_2.x[self.cell_2_index], self.cell_2.y[self.cell_2_index]]]

    def get_spacing_at_other_end(self, this_cell_id):
        """Returns the spacing at the other side of the adhesion, given that we are in cell with ``this_cell_id``

        :param this_cell_id: The identity of the cell that we know/are in.
        :type this_cell_id: string
        :return: The discretised spacing on the other cortex.
        :rtype: float

        """
        if this_cell_id == self.cell_1.identifier:
            return self.cell_2.deformed_mesh_spacing[self.cell_2_index]
        elif this_cell_id == self.cell_2.identifier:
            return self.cell_1.deformed_mesh_spacing[self.cell_1_index]
        else:
            raise ValueError('given cell does not belong to adhesion (should be cell id)')

    def get_cell_id_at_other_end(self, this_cell_id):
        """Returns the identity at the other side of the adhesion, given that we are in ``this_cell``

        :param this_cell_id: The identity of the cell that we know.
        :type this_cell_id: string
        :return: The identity of the cell on the other side.
        :rtype: string

        """
        if this_cell_id == self.cell_1.identifier:
            return self.cell_2.identifier
        elif this_cell_id == self.cell_2.identifier:
            return self.cell_1.identifier
        else:
            raise ValueError('given cell does not belong to adhesion (should be cell id)')

    def get_xy_at_other_end(self, this_cell_id):
        """Returns the spacing at the other side of the adhesion, given that we are in this_cell

        :param this_cell_id: The identifier of the cell we know.
        :type this_cell_id: string
        :return: The (x,y) coords that the adhesion is connected to on the other end.
        :rtype: list

        """
        if this_cell_id == self.cell_1.identifier:
            return [self.cell_2.x[self.cell_2_index], self.cell_2.y[self.cell_2_index]]
        elif this_cell_id == self.cell_2.identifier:
            return [self.cell_1.x[self.cell_1_index], self.cell_1.y[self.cell_1_index]]
        else:
            raise ValueError('given cell does not belong to adhesion (should be cell id)')

    def get_xy_at_this_end(self, this_cell_id):
        """Returns the discrete cortex spacing where the adhesion is attached to a given cell

        :param this_cell_id:  The identity of the cell to get the spacing.
        :type this_cell_id: string
        :return: The (x,y) coords that the adhesion is connected to on the given cell.
        :rtype: list

        """
        if this_cell_id == self.cell_1.identifier:
            return [self.cell_1.x[self.cell_1_index], self.cell_1.y[self.cell_1_index]]
        elif this_cell_id == self.cell_2.identifier:
            return [self.cell_2.x[self.cell_2_index], self.cell_2.y[self.cell_2_index]]
        else:
            raise ValueError('given cell does not belong to adhesion (should be cell id)')

    def get_length(self, cell_id_for_new_xy='None', new_xy=(0, 0)):
        """Get the length of the adhesion. Can change the xy location of one of the cells.

        :param cell_id_for_new_xy:  (Default value = 'None')  Optional identifier for cell that we want to change the position of.
        :type cell_id_for_new_xy: string
        :param new_xy:  (Default value = (0, 0)  The new (x,y) position of the given cell.
        :type new_xy: tuple
        :return: The length of the adhesion.
        :rtype: float

        """
        if cell_id_for_new_xy == self.cell_1.identifier and new_xy != (0, 0):
            length = np.sqrt((new_xy[0] - self.cell_2.x[self.cell_2_index]) ** 2 +
                             (new_xy[1] - self.cell_2.y[self.cell_2_index]) ** 2)
        elif cell_id_for_new_xy == self.cell_2.identifier and new_xy != (0, 0):
            length = np.sqrt((new_xy[0] - self.cell_1.x[self.cell_1_index]) ** 2 +
                             (new_xy[1] - self.cell_1.y[self.cell_1_index]) ** 2)
        else:
            length = np.sqrt((self.cell_1.x[self.cell_1_index] - self.cell_2.x[self.cell_2_index]) ** 2 +
                             (self.cell_1.y[self.cell_1_index] - self.cell_2.y[self.cell_2_index]) ** 2)
        return length

    def get_force_magnitude(self, cell_id_for_new_xy='None', new_xy=(0, 0)):
        """Force acting on adhesion

        :param cell_id_for_new_xy:  (Default value = 'None')  Optional identifier for cell that we want to change the position of.
        :type cell_id_for_new_xy: string
        :param new_xy:  (Default value = (0, 0)  The new (x,y) position of the given cell.
        :type new_xy: tuple
        :return: The magnitude of (spring) force in the adhesion.
        :rtype: float

        """
        # Get length first
        length = self.get_length(cell_id_for_new_xy=cell_id_for_new_xy, new_xy=new_xy)

        e = length - self.delta if length < self.max_bonding_length else 0
        force = self.omega * e
        # if cell_id_for_new_xy == self.cell_1.identifier:
        #     force *= self.cell_1.deformed_mesh_spacing[self.cell_1_index]
        # if cell_id_for_new_xy == self.cell_2.identifier:
        #     force *= self.cell_2.deformed_mesh_spacing[self.cell_2_index]
        # else:
        #     raise ValueError('given cell does not belong to adhesion (should be cell id)')

        return force
    def get_energy_magnitude(self, cell_id_for_new_xy='None', new_xy=(0, 0)):
        """Force acting on adhesion

        :param cell_id_for_new_xy:  (Default value = 'None')  Optional identifier for cell that we want to change the position of.
        :type cell_id_for_new_xy: string
        :param new_xy:  (Default value = (0, 0)  The new (x,y) position of the given cell.
        :type new_xy: tuple
        :return: The magnitude of (spring) force in the adhesion.
        :rtype: float

        """
        # Get length first
        length = self.get_length(cell_id_for_new_xy=cell_id_for_new_xy, new_xy=new_xy)

        e = length - self.delta if length < self.max_bonding_length else 0
        energy = self.omega * e**2

        return energy

    def get_unit_direction(self, cell_id_for_new_xy='None', new_xy=(0, 0)):
        """Get a vector describing the direction of the adhesion

        :param cell_id_for_new_xy:  (Default value = 'None')  Optional identifier for cell that we want to change the position of.
        :type cell_id_for_new_xy: string
        :param new_xy:  (Default value = (0, 0)  The new (x,y) position of the given cell.
        :type new_xy: tuple
        :return: The direction of the adhesion, from cell1 to cell2
        :rtype: list

        """
        if cell_id_for_new_xy == self.cell_1.identifier:
            cell_from = self.cell_1
            cell_from_id = self.cell_1_index
            cell_to = self.cell_2
            cell_to_id = self.cell_2_index

        elif cell_id_for_new_xy == self.cell_2.identifier:
            cell_from = self.cell_2
            cell_from_id = self.cell_2_index
            cell_to = self.cell_1
            cell_to_id = self.cell_1_index
        else:
            raise ValueError('given cell does not belong to adhesion (should be cell object, not id)')

        if new_xy == (0, 0):
            from_x, from_y = cell_from.x[cell_from_id], cell_from.y[cell_from_id]
        else:
            from_x, from_y = new_xy[0], new_xy[1]

        direction = [cell_to.x[cell_to_id] - from_x, cell_to.y[cell_to_id] - from_y]
        magnitude = np.sqrt(direction[0] ** 2 + direction[1] ** 2)
        direction[0] /= magnitude
        direction[1] /= magnitude

        return direction

    def get_vector_force(self, from_cell_id, new_xy_for_from_cell=(0, 0)):
        """Get a vector for the adhesion force, acting on from_cell_id

        :param cell_id_for_new_xy:  (Default value = 'None')  Optional identifier for cell that we want to change the position of.
        :type cell_id_for_new_xy: string
        :param new_xy:  (Default value = (0, 0)  The new (x,y) position of the given cell.
        :type new_xy: tuple
        :return: The vector force exerted by the adhesion on cell 1.
        :rtype: list

        """
        magnitude = self.get_force_magnitude(cell_id_for_new_xy=from_cell_id, new_xy=new_xy_for_from_cell)
        direction = self.get_unit_direction(cell_id_for_new_xy=from_cell_id, new_xy=new_xy_for_from_cell)

        # Multiply spacing spacing on cortices
        magnitude *= self.cell_2.deformed_mesh_spacing[self.cell_2_index] * self.cell_1.deformed_mesh_spacing[
            self.cell_1_index]
        # if self.cell_1.identifier == from_cell_id:
        #     magnitude *= self.cell_1.deformed_mesh_spacing[self.cell_1_index] * self.cell_2.deformed_mesh_spacing[self.cell_2_index]
        # elif self.cell_2.identifier == from_cell_id:
        #     magnitude *= self.cell_2.deformed_mesh_spacing[self.cell_2_index] * self.cell_1.deformed_mesh_spacing[self.cell_1_index]
        # else:
        #     raise ValueError('given cell does not belong to adhesion (should be cell object, not id)')

        direction[0] *= magnitude
        direction[1] *= magnitude

        return direction

    def get_angle_relative_to_cortices(self):
        """Get the angle the adhesion makes relative to the tangent along both cortices

        :return: The angle that the adhesion makes relative to the tangent along the connected cortices.
        :rtype: (float, float)

        """
        # adhesion vector
        ad_vector = self.get_unit_direction(cell_id_for_new_xy=self.cell_1.identifier)
        # Get tangent along cortices
        if self.cell_1.identifier != 'boundary':
            tangent_1 = [np.cos(self.cell_1.theta[self.cell_1_index]), np.sin(self.cell_1.theta[self.cell_1_index])]
        else:
            positive_idx = self.cell_1_index + 1 if self.cell_1_index < self.cell_1.x.size - 1 else 0
            tangent_1 = [self.cell_1.x[positive_idx] - self.cell_1.x[self.cell_1_index - 1],
                         self.cell_1.y[positive_idx] - self.cell_1.y[self.cell_1_index - 1]]
        if self.cell_2.identifier != 'boundary':
            tangent_2 = [np.cos(self.cell_2.theta[self.cell_2_index]), np.sin(self.cell_2.theta[self.cell_2_index])]
        else:
            positive_idx = self.cell_2_index + 1 if self.cell_2_index < self.cell_2.x.size - 1 else 0
            tangent_2 = [self.cell_2.x[positive_idx] - self.cell_2.x[self.cell_2_index - 1],
                         self.cell_2.y[positive_idx] - self.cell_2.y[self.cell_2_index - 1]]

        # Angles
        angle_1 = np.arccos(ad_vector[0] * tangent_1[0] + ad_vector[1] * tangent_1[1])
        angle_2 = np.arccos(ad_vector[0] * tangent_2[0] + ad_vector[1] * tangent_2[1])

        angle_1 = np.pi - angle_1 if angle_1 > np.pi / 2 else angle_1
        angle_2 = np.pi - angle_2 if angle_2 > np.pi / 2 else angle_2

        return angle_1, angle_2

    def plot(self, ax=None, **plot_params):
        """

        :param ax:  (Default value = None)  Axis object to plot on.
        :type ax: mpl axis
        :param plot_params:  Optional plotting arguments.
        :type plot_params: dict

        """

        if ax == None:
            f, ax = plt.subplots()

        if self.adhesion_type == 'sdk':
            linestyle = plot_params.get('linestyle', '-')
            colour = plot_params.get('colour', '#8000ff')
            lw = plot_params.get('lw', '2.5')
        elif self.adhesion_type == 'cadherin':
            linestyle = plot_params.get('linestyle', '-')
            colour = plot_params.get('colour', 'y')
            lw = plot_params.get('lw', '1')
        else:
            raise ValueError('self.type must be "cadherin" or "sdk"')

        coords = self.get_xy()

        ax.plot([coords[0][0], coords[1][0]], [coords[0][1], coords[1][1]], linestyle, c=colour, lw=lw)
