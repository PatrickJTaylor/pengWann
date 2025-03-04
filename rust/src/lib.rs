//! A small rust extension for pengwann.
//!
//! Provides a few Rust implementations for functions that would be prohibitively slow
//! in Python, mainly due to loops that cannot be unrolled by the interpreter.

use std::iter::repeat_n;

use itertools::Itertools;
use ndarray::{Array1, Array2, Array3, ArrayView2, Axis, s};
use ndarray_stats::QuantileExt;

mod python_module;

/// Given a set of fractional coordinates and cell vectors, calculate a distance matrix
/// and the accompanying image matrix. For each element of the distance matrix D_ij, the
/// corresponding element of the image matrix I_ij is the Bravais lattice vector that
/// yields the periodic image of atom j that is closest to atom i.
///
/// This function currently implements no error handling, because it is expected to only
/// ever be called internally (never directly by the user) with arguments that will
/// never cause a panic.
fn build_distance_and_image_matrices(
    frac_coords: &ArrayView2<f64>,
    cell: &ArrayView2<f64>,
) -> (Array2<f64>, Array3<i32>) {
    let (num_sites, num_dim) = frac_coords.dim();

    let image_vectors = enum_image_vectors(num_dim);
    let mut distance_matrix = Array2::<f64>::zeros((num_sites, num_sites));
    let mut image_matrix = Array3::<i32>::zeros((num_sites, num_sites, num_dim));

    for i in 0..num_sites {
        for j in i + 1..num_sites {
            let i_coords = frac_coords.slice(s![i, ..]);
            let j_coords = frac_coords.slice(s![j, ..]);

            let v_0 = (&i_coords - &j_coords).round();
            let trans_j_coords = &j_coords + &v_0;

            let frac_vectors = &trans_j_coords + &image_vectors - i_coords;
            let cart_vectors = cell.dot(&frac_vectors.t());

            let distances = normalise(&cart_vectors, Axis(0));
            let min_idx = distances.argmin().unwrap();

            distance_matrix[[i, j]] = distances[min_idx];
            distance_matrix[[j, i]] = distances[min_idx];

            let image_i = (&image_vectors.slice(s![min_idx, ..]) + &v_0)
                .round()
                .mapv(|x| x as i32);
            let image_j = -1 * &image_i;
            image_matrix.slice_mut(s![i, j, ..]).assign(&image_i);
            image_matrix.slice_mut(s![j, i, ..]).assign(&image_j);
        }
    }

    (distance_matrix, image_matrix)
}

/// Enumerate the Bravais lattice vectors in n-dimensions for all unit cells
/// neighbouring the home cell.
fn enum_image_vectors(num_dim: usize) -> Array2<f64> {
    let num_images = 3_usize.pow(u32::try_from(num_dim).unwrap());

    let mut image_vectors = Array2::<f64>::zeros((num_images, num_dim));
    for (i, bl_vector) in repeat_n(-1..2, num_dim)
        .multi_cartesian_product()
        .enumerate()
    {
        let bl_vector = Array1::from_vec(bl_vector).mapv(f64::from);

        image_vectors.slice_mut(s![i, ..]).assign(&bl_vector);
    }

    image_vectors
}

/// Normalise a 2-dimensional matrix along a given axis.
fn normalise(matrix: &Array2<f64>, axis: Axis) -> Array1<f64> {
    (matrix * matrix).sum_axis(axis).sqrt()
}
