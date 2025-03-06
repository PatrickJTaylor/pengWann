//! A PyO3 module that allows Python to call the Rust extension.

// Copyright (C) 2024-2025 Patrick J. Taylor

// This file is part of pengWann.
//
// pengWann is free software: you can redistribute it and/or modify it under the terms
// of the GNU General Public License as published by the Free Software Foundation, either
// version 3 of the License, or (at your option) any later version.
//
// pengWann is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
// without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
// PURPOSE. See the GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License along with pengWann.
// If not, see <https://www.gnu.org/licenses/>.

use numpy::{IntoPyArray, PyArray2, PyArray3, PyReadonlyArray2};
use pyo3::prelude::*;

use crate::build_distance_and_image_matrices;

/// The PyO3 module itself, exported as _geometry which maturin turns into a submodule:
/// pengwann._geometry.
#[pymodule(name = "_geometry")]
fn pengwann(m: &Bound<'_, PyModule>) -> PyResult<()> {
    /// Translation function handling the conversion to and from numpy and ndarray in
    /// order to call the build_distance_and_image_matrices function from Python.
    #[pyfn(m)]
    #[pyo3(name = "_build_distance_and_image_matrices")]
    fn py_build_distance_and_image_matrices<'py>(
        py: Python<'py>,
        py_coords: PyReadonlyArray2<'py, f64>,
        py_cell: PyReadonlyArray2<'py, f64>,
    ) -> (Bound<'py, PyArray2<f64>>, Bound<'py, PyArray3<i32>>) {
        let frac_coords = py_coords.as_array();
        let cell = py_cell.as_array();

        let (distance_matrix, image_matrix) =
            build_distance_and_image_matrices(&frac_coords, &cell);

        (
            distance_matrix.into_pyarray(py),
            image_matrix.into_pyarray(py),
        )
    }

    Ok(())
}
