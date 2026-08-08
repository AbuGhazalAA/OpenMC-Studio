import logging
import openmc
import os

logger = logging.getLogger("OpenMC Studio")


class ProjectManager:
    """
    The Central Brain of OpenMC Studio.
    Handles the translation between the PySide6 UI and the OpenMC Python API.
    Maintains the state of the current simulation project.
    """

    def __init__(self):
        # Initialize native OpenMC collections
        self.materials = openmc.Materials()
        self.geometry = openmc.Geometry()
        self.settings = openmc.Settings()
        self.tallies = openmc.Tallies()

        # Dictionaries to keep track of UI objects linked to OpenMC objects
        self._material_dict = {}

        logger.info("ProjectManager initialized with empty OpenMC structures.")

    def add_material(self, name: str, density: float, density_units: str) -> openmc.Material:
        """
        Creates an OpenMC Material object and stores it in the project.
        """
        mat = openmc.Material(name=name)
        mat.set_density(density_units, density)

        self.materials.append(mat)
        self._material_dict[name] = mat

        logger.debug(f"Added Material '{name}' to ProjectManager.")
        return mat

    def add_nuclide_to_material(self, mat_name: str, nuclide_name: str, fraction: float, fraction_type: str):
        """
        Adds a nuclide or element to an existing material.
        """
        if mat_name in self._material_dict:
            mat = self._material_dict[mat_name]
            # OpenMC API expects 'ao' or 'wo'
            mat.add_nuclide(nuclide_name, fraction, percent_type=fraction_type)
            logger.debug(f"Added nuclide {nuclide_name} to {mat_name}.")
        else:
            logger.error(f"Material {mat_name} not found in ProjectManager.")

    def export_xml(self, export_dir: str = "."):
        """
        Exports all current configurations to OpenMC XML files.
        """
        if not os.path.exists(export_dir):
            os.makedirs(export_dir)

        try:
            if self.materials:
                self.materials.export_to_xml(os.path.join(export_dir, "materials.xml"))

            if self.geometry:
                self.geometry.export_to_xml(os.path.join(export_dir, "geometry.xml"))

            if self.settings:
                self.settings.export_to_xml(os.path.join(export_dir, "settings.xml"))

            # تصدير ملف الـ plots.xml لتمكين محرك الرسم من العمل
            if hasattr(self, 'plots') and self.plots:
                self.plots.export_to_xml(os.path.join(export_dir, "plots.xml"))

            return True
        except Exception as e:
            logger.error(f"Failed to export XML files: {e}")
            return False