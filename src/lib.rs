// Include readme in doc
#![doc = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/Readme.md"))]

pub mod error;

use std::fmt::Debug;
use std::iter::Sum;
use std::ops::AddAssign;

use kdtree::KdTree;
use ndarray::{Array1, Array2, ScalarOperand};
use num_traits::{Float, FloatConst};

use crate::error::Error;

/// Type alias for distance function
/// Due to the kdtree implementation we can not return a Error in case the dimensions do not match
///
#[allow(type_alias_bounds)]
pub type DistanceFn<T: Float + FloatConst + ScalarOperand + AddAssign + Sum + Debug> =
    fn(&[T], &[T]) -> T;

/// Manhattan distance
///
/// # Arguments
/// * `a` - First point
/// * `b` - Second point
///
pub fn manhattan<T>(a: &[T], b: &[T]) -> T
where
    T: Float + FloatConst + ScalarOperand + AddAssign + Sum + Debug,
{
    debug_assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (*x - *y).abs())
        .fold(T::zero(), ::std::ops::Add::add)
}

/// Euclidean distance
///
/// # Arguments
/// * `a` - First point
/// * `b` - Second point
///
pub fn euclidean<T>(a: &[T], b: &[T]) -> T
where
    T: Float + FloatConst + ScalarOperand + AddAssign + Sum + Debug,
{
    debug_assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (*x - *y).powi(2))
        .fold(T::zero(), ::std::ops::Add::add)
        .sqrt()
}

/// Local Outlier Probability (LoOP) according to
/// > Kriegel, H.-P.; Kröger, P.; Schubert, E. & Zimek, A. (2009),
/// > LoOP: local outlier probabilities.,
/// > in David Wai-Lok Cheung; Il-Yeol Song; Wesley W. Chu; Xiaohua Hu & Jimmy Lin, ed.,
/// > 'CIKM' , ACM, , pp. 1649-1652 .
///
/// # Arguments
/// * `data` - 2D array of data points N x M (N rows & M columns).
///   **It is important to note that the data is expected to be in contiguous memory.**
/// * `k` - Float + FloatConst + ScalarOperand + AddAssign + Sum + Debug of nearest neighbors to consider
/// * `lambda` - Scaling factor for the probabilistic set distance & Probabilistic Local Outlier (PLod)
/// * `distance_fn` - Optional distance function to use (default is Manhattan distance)
///
pub fn local_outlier_probabilities<T>(
    data: Array2<T>,
    k: usize,
    lambda: u8,
    distance_fn: Option<DistanceFn<T>>,
) -> Result<Array1<T>, Error>
where
    T: Float + FloatConst + ScalarOperand + AddAssign + Sum + Debug,
{
    // Unwrap the distance function or use the default
    let distance_fn = distance_fn.unwrap_or(manhattan);

    // Build the KDTree for finding nearest neighbors
    let mut tree = KdTree::new(data.shape()[1]);

    // Get and array of references to the underlying data in the array
    // and add them to the KDTree. This should safe some memory on larger features arrays
    let point_refs = data
        .outer_iter()
        .map(|point| {
            point
                .to_slice()
                .ok_or(Error::SliceNotContiguous("collecting points references"))
        })
        .collect::<Result<Vec<_>, Error>>()?;

    for (idx, point) in point_refs.iter().enumerate() {
        tree.add(*point, idx)?;
    }

    // Build the neighbor list. We need to manually remove the queried point from the list of neighbors
    // therefore the map()-part is a bit more complicated and we increase the `k by 1
    let neighbors_list = point_refs
        .iter()
        .enumerate()
        .map(|(point_idx, point)| {
            match tree.nearest(point, k + 1, &distance_fn) {
                Ok(neighbors) => {
                    // kdtree::nearest includes the point itself, so we need to add 1 to k and remove the point
                    let filtered = neighbors
                        .into_iter()
                        .filter(|neighbor| {
                            // filter out the point itself
                            *neighbor.1 != point_idx
                        })
                        .collect::<Vec<_>>();
                    Ok(filtered)
                }
                Err(err) => Err(Error::KdTreeError(err)),
            }
        })
        .collect::<Result<Vec<Vec<(T, &usize)>>, Error>>()?;

    // Let's prepare some to Floats to work with
    let k_float = T::from(k).ok_or(Error::KCastingError("k".to_string()))?;
    let lambda_float = T::from(lambda).ok_or(Error::KCastingError("lambda".to_string()))?;
    let two_squared = T::from(2.0).ok_or(Error::KCastingError("2.0".to_string()))?;

    // Calculate the probabilistic distance for each point
    let pdists = neighbors_list
        .iter()
        .map(|neighbors| calc_sigma(neighbors, k_float))
        .collect::<Array1<_>>()
        * lambda_float;

    // Calculate the Probabilistic Outlier Factor for each point
    let plofs = neighbors_list
        .iter()
        .zip(pdists.iter())
        .map(|(nearest_neighbors, pdist)| calc_plof(nearest_neighbors, *pdist, &pdists))
        .collect::<Array1<_>>();

    // Aggregate the Probabilistic Outlier Factor (nPLOF)
    let nplof = calc_nplof(&plofs, lambda_float);

    // Calculate the local outlier probability
    let local_outlier_prob =
        (plofs / (nplof * two_squared.sqrt())).map(|x| erf_approx(*x).max(T::zero()));

    Ok(local_outlier_prob)
}

/// Calculate the sigma used in the probabilistic distance function
///
/// # Arguments
/// * `neighbors` - A slice of tuples containing the distance and index of the neighbors
/// * `k` - Number of neighbors to consider
///
fn calc_sigma<T>(neighbors: &[(T, &usize)], k: T) -> T
where
    T: Float + FloatConst + ScalarOperand + AddAssign + Sum + Debug,
{
    (neighbors.iter().map(|(dist, _)| dist.powi(2)).sum::<T>() / k).sqrt()
}

/// Probabilistic Outlier Factor (PLOF) for a point
///
/// # Arguments
/// * `neighbors` - A slice of tuples containing the distance and index of the neighbors
/// * `pdist` - Probabilistic distance of the point
/// * `pdists` - Probabilistic distance of the points
///
fn calc_plof<T>(nearest_neighbors: &[(T, &usize)], pdist: T, pdists: &Array1<T>) -> T
where
    T: Float + FloatConst + ScalarOperand + AddAssign + Sum + Debug,
{
    let nn_mean = nearest_neighbors
        .iter()
        .map(|(_, idx)| pdists[**idx])
        .sum::<T>()
        / T::from(nearest_neighbors.len()).unwrap();

    pdist / nn_mean - T::one()
}

/// Aggregate Probabilistic Outlier Factor (nPLOF)
///
/// # Arguments
/// * `plofs` - Array of PLOF values
///
fn calc_nplof<T>(plofs: &Array1<T>, lambda: T) -> T
where
    T: Float + FloatConst + ScalarOperand + AddAssign + Sum + Debug,
{
    let plofs_squared_mean =
        plofs.iter().map(|x| x.powi(2)).sum::<T>() / T::from(plofs.len()).unwrap();

    lambda * plofs_squared_mean.sqrt()
}

/// Approximate the error function (erf) according to
/// > Abramowitz, Milton, and Irene A. Stegun, eds.
/// > Handbook of mathematical functions with formulas, graphs, and mathematical tables. Vol. 55. US Government printing office, 1948.
/// > Equation 7.1.25
///
/// # Arguments
/// * `x` - Input value
///
fn erf_approx<T>(x: T) -> T
where
    T: Float + FloatConst + ScalarOperand + AddAssign + Sum + Debug,
{
    T::one()
        - T::one()
            / (T::one()
                + T::from(0.278393).unwrap() * x
                + T::from(0.230389).unwrap() * x.powi(2)
                + T::from(0.000972).unwrap() * x.powi(3)
                + T::from(0.078108).unwrap() * x.powi(4))
            .powi(4)
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;
    use ndarray_stats::DeviationExt;
    use polars::prelude::*;

    #[test]
    fn test_loop() {
        // Test file contains a peptide spectrum matches form a proteomics-MS experiment with a precalculated loop score (PyNomaly)
        // based on the features xcorr, ions_matched_ratio and mass_diff, using lambda=3 (default) and k=1000
        let df_file_path = PathBuf::from("test_files/scored_psms.tsv");

        // Read the TSV file into a DataFrame
        let df = CsvReadOptions::default()
            .with_has_header(true)
            .with_parse_options(CsvParseOptions::default().with_separator(b'\t'))
            .try_into_reader_with_file_path(Some(df_file_path))
            .unwrap()
            .finish()
            .unwrap();

        // Select the relevant columns
        let feature_df = df
            .select(["xcorr", "ions_matched_ratio", "mass_diff"])
            .unwrap();

        // Convert the DataFrame to a 2D array
        // The data is expected to be in contiguous memory therefore we use the C-order
        let array = feature_df.to_ndarray::<Float64Type>(IndexOrder::C).unwrap();

        // Calculate the local outlier probabilities
        let loop_score = local_outlier_probabilities(array, 1000, 3, None).unwrap();

        assert_eq!(loop_score.len(), df.height());

        let loop_score_py = df
            .column("loop_score")
            .unwrap()
            .f64()
            .unwrap()
            .into_no_null_iter()
            .collect::<Array1<_>>();

        let rmse = loop_score.root_mean_sq_err(&loop_score_py).unwrap();

        // RMSE under 0.02 should be good enough
        assert!(rmse < 0.02, "RMSE > 0.02");
    }
}
