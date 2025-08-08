import torch
from src.utils import bmv, bdot

class CrackGeometry:
    """
    Defines the geometry of a crack in a 2D plane.
    """
    def __init__(self, crack_type='center', crack_param=None):
        """
        Initializes the crack geometry.

        Args:
            crack_type (str): The type of crack ('center', 'edge', 'inclined').
            crack_param (dict): A dictionary of parameters defining the crack.
                                For 'center' and 'edge': {'length': float}
                                For 'inclined': {'length': float, 'angle': float}
        """
        self.crack_type = crack_type
        self.crack_param = crack_param

    def sdf(self, points):
        """
        Calculates the signed distance function (SDF) for the crack.

        Args:
            points (torch.Tensor): A tensor of points (x, y) where the SDF is to be evaluated.

        Returns:
            torch.Tensor: The SDF values at the given points.
        """
        if self.crack_type == 'center':
            return self._center_crack_sdf(points)
        elif self.crack_type == 'edge':
            return self._edge_crack_sdf(points)
        elif self.crack_type == 'inclined':
            return self._inclined_crack_sdf(points)
        else:
            raise ValueError(f"Unknown crack type: {self.crack_type}")

    def _center_crack_sdf(self, points):
        """
        Calculates the SDF for a central crack.
        """
        crack_length = self.crack_param['length']
        x = points[:, 0]
        y = points[:, 1]
        
        # SDF for a horizontal crack centered at the origin
        sdf = torch.abs(y)
        # Set SDF to 0 for points on the crack
        on_crack = (torch.abs(x) <= crack_length / 2) & (torch.abs(y) < 1e-9)
        sdf[on_crack] = 0
        return sdf

    def _edge_crack_sdf(self, points):
        """
        Calculates the SDF for an edge crack.
        """
        crack_length = self.crack_param['length']
        x = points[:, 0]
        y = points[:, 1]

        # SDF for a horizontal crack starting from the left edge
        sdf = torch.abs(y)
        on_crack = (x >= 0) & (x <= crack_length) & (torch.abs(y) < 1e-9)
        sdf[on_crack] = 0
        return sdf

    def _inclined_crack_sdf(self, points):
        """
        Calculates the SDF for an inclined crack.
        """
        crack_length = self.crack_param['length']
        angle = self.crack_param['angle']
        x = points[:, 0]
        y = points[:, 1]

        # Rotate points to align the crack with the x-axis
        cos_a = torch.cos(angle)
        sin_a = torch.sin(angle)
        x_rot = x * cos_a + y * sin_a
        y_rot = -x * sin_a + y * cos_a

        # SDF for a horizontal crack centered at the origin
        sdf = torch.abs(y_rot)
        on_crack = (torch.abs(x_rot) <= crack_length / 2) & (torch.abs(y_rot) < 1e-9)
        sdf[on_crack] = 0
        return sdf

    def get_crack_embedding(self, points):
        """
        Generates the crack embedding based on the SDF.

        Args:
            points (torch.Tensor): A tensor of points where the embedding is to be calculated.

        Returns:
            torch.Tensor: The crack embedding values.
        """
        sdf_values = self.sdf(points)
        return torch.sign(sdf_values)