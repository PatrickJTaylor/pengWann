Geometry (:code:`pengwann.geometry`)
====================================
.. currentmodule:: pengwann.geometry

.. automodule:: pengwann.geometry

Geometry manipulation functions
-------------------------------
.. autosummary::
   :toctree: generated

   build_geometry
   assign_wannier_centres
   find_interactions

Interaction dataclasses
-----------------------
These are the data structures that are used to parse/store data related to interactions
between pairs of Wannier functions (:py:class:`~pengwann.geometry.WannierInteraction`)
and pairs of atoms (:py:class:`~pengwann.geometry.AtomicInteraction`).

.. autosummary::
   :template: dataclass_template.rst
   :toctree: generated

   AtomicInteraction
   WannierInteraction
