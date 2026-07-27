// Include readme in doc
#![doc = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/Readme.md"))]

pub mod error;

use std::borrow::Cow;
use std::fmt::Debug;
use std::iter::Sum;
use std::ops::AddAssign;

use kdtree::KdTree;
use num_traits::{Float, FloatConst};

use crate::error::Error;

/// A source of points that can be queried by index, without requiring the caller
/// to convert their data into a specific owned container first.
///
/// A blanket implementation is provided for `[R] where R: AsRef<[T]>` (covers
/// `Vec<Vec<T>>`, `&[[T; M]]`, etc.), which borrows every point at zero cost.
/// To feed in a type from another crate (e.g. `ndarray::Array2<T>`) without adding
/// that crate as a dependency here, implement this trait for a thin wrapper around
/// it in your own code, e.g.:
///
/// ```ignore
/// struct NdarrayPoints<'a, T>(&'a ndarray::Array2<T>);
///
/// impl<'a, T: Clone> PointSource<T> for NdarrayPoints<'a, T> {
///     fn len(&self) -> usize {
///         self.0.nrows()
///     }
///     fn dim(&self) -> usize {
///         self.0.ncols()
///     }
///     fn point(&self, idx: usize) -> Result<std::borrow::Cow<'_, [T]>, Error> {
///         // `Array2::row` is a zero-copy view; `to_slice` only fails for non-contiguous arrays.
///         let row = self.0.row(idx);
///         Ok(std::borrow::Cow::Borrowed(row.to_slice().expect("contiguous row")))
///     }
/// }
/// ```
#[allow(clippy::len_without_is_empty)]
pub trait PointSource<T: Clone> {
    /// Number of points.
    fn len(&self) -> usize;
    /// Number of dimensions every point is expected to have.
    fn dim(&self) -> usize;
    /// Borrow (or, if the underlying storage isn't row-major, gather) the point at `idx`.
    fn point(&self, idx: usize) -> Result<Cow<'_, [T]>, Error>;
}

impl<T: Clone, R> PointSource<T> for [R]
where
    R: AsRef<[T]>,
{
    fn len(&self) -> usize {
        <[R]>::len(self)
    }

    fn dim(&self) -> usize {
        self.first().map(|row| row.as_ref().len()).unwrap_or(0)
    }

    fn point(&self, idx: usize) -> Result<Cow<'_, [T]>, Error> {
        Ok(Cow::Borrowed(self[idx].as_ref()))
    }
}

impl<T: Clone, R> PointSource<T> for Vec<R>
where
    R: AsRef<[T]>,
{
    fn len(&self) -> usize {
        PointSource::<T>::len(self.as_slice())
    }

    fn dim(&self) -> usize {
        PointSource::<T>::dim(self.as_slice())
    }

    fn point(&self, idx: usize) -> Result<Cow<'_, [T]>, Error> {
        PointSource::<T>::point(self.as_slice(), idx)
    }
}

/// Type alias for distance function
/// Due to the kdtree implementation we can not return a Error in case the dimensions do not match
///
#[allow(type_alias_bounds)]
pub type DistanceFn<T: Float + FloatConst + AddAssign + Sum + Debug> = fn(&[T], &[T]) -> T;

/// Manhattan distance
///
/// # Arguments
/// * `a` - First point
/// * `b` - Second point
///
pub fn manhattan<T>(a: &[T], b: &[T]) -> T
where
    T: Float + FloatConst + AddAssign + Sum + Debug,
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
    T: Float + FloatConst + AddAssign + Sum + Debug,
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
/// * `data` - Any [`PointSource`], N points of M dimensions each (all points must have the
///   same dimension). A blanket impl covers `&[R] where R: AsRef<[T]>` (e.g. `&[Vec<T>]`) for free;
///   implement `PointSource` for your own wrapper to support other containers without converting them.
/// * `k` - Number of nearest neighbors to consider
/// * `lambda` - Scaling factor for the probabilistic set distance & Probabilistic Local Outlier (PLod)
/// * `distance_fn` - Optional distance function to use (default is Manhattan distance)
///
pub fn local_outlier_probabilities<T, S>(
    data: &S,
    k: usize,
    lambda: u8,
    distance_fn: Option<DistanceFn<T>>,
) -> Result<Vec<T>, Error>
where
    T: Float + FloatConst + AddAssign + Sum + Debug,
    S: PointSource<T> + ?Sized,
{
    // Unwrap the distance function or use the default
    let distance_fn = distance_fn.unwrap_or(manhattan);

    if data.len() == 0 {
        return Err(Error::EmptyInput);
    }
    let dim = data.dim();

    // Borrow (or gather) every point once up front
    let points = (0..data.len())
        .map(|idx| data.point(idx))
        .collect::<Result<Vec<Cow<'_, [T]>>, Error>>()?;

    for (row, point) in points.iter().enumerate() {
        if point.len() != dim {
            return Err(Error::DimensionMismatch {
                expected: dim,
                found: point.len(),
                row,
            });
        }
    }

    // Let's prepare some Floats to work with
    let k_float = T::from(k).ok_or(Error::CastingError("k".to_string()))?;
    let lambda_float = T::from(lambda).ok_or(Error::CastingError("lambda".to_string()))?;
    let two_squared = T::from(2.0).ok_or(Error::CastingError("2.0".to_string()))?;

    // Build the KDTree, then for every point derive its probabilistic distance (pdist) and the
    // indices of its k nearest neighbors in one pass. Distances are only needed transiently to
    // compute sigma, so we never store a (distance, index) pair per neighbor - only the index
    // survives, which the PLOF step below needs to look up other points' pdists. The tree itself
    // is dropped at the end of this block since nothing after it needs the point coordinates.
    let (pdists, neighbor_indices) = {
        let mut tree = KdTree::new(dim);
        for (idx, point) in points.iter().enumerate() {
            tree.add(point.as_ref(), idx)?;
        }

        let mut pdists = Vec::with_capacity(points.len());
        let mut neighbor_indices = Vec::with_capacity(points.len());

        for (point_idx, point) in points.iter().enumerate() {
            // kdtree::nearest includes the point itself, so we query k + 1 and filter it out below
            let neighbors = tree.nearest(point.as_ref(), k + 1, &distance_fn)?;

            let mut sum_sq = T::zero();
            let mut indices = Vec::with_capacity(k);
            for (dist, idx) in neighbors {
                if *idx == point_idx {
                    continue;
                }
                sum_sq += dist.powi(2);
                indices.push(*idx);
            }

            pdists.push((sum_sq / k_float).sqrt() * lambda_float);
            neighbor_indices.push(indices);
        }

        (pdists, neighbor_indices)
    };

    // Calculate the Probabilistic Outlier Factor for each point
    let plofs = neighbor_indices
        .iter()
        .zip(pdists.iter())
        .map(|(indices, pdist)| calc_plof(indices, *pdist, &pdists))
        .collect::<Vec<T>>();

    // Aggregate the Probabilistic Outlier Factor (nPLOF)
    let nplof = calc_nplof(&plofs, lambda_float);

    // Calculate the local outlier probability
    let local_outlier_prob = plofs
        .iter()
        .map(|x| erf_approx(*x / (nplof * two_squared.sqrt())).max(T::zero()))
        .collect::<Vec<T>>();

    Ok(local_outlier_prob)
}

/// Probabilistic Outlier Factor (PLOF) for a point
///
/// # Arguments
/// * `neighbor_indices` - Indices of the point's k nearest neighbors
/// * `pdist` - Probabilistic distance of the point
/// * `pdists` - Probabilistic distance of the points
///
fn calc_plof<T>(neighbor_indices: &[usize], pdist: T, pdists: &[T]) -> T
where
    T: Float + FloatConst + AddAssign + Sum + Debug,
{
    let nn_mean = neighbor_indices.iter().map(|idx| pdists[*idx]).sum::<T>()
        / T::from(neighbor_indices.len()).unwrap();

    pdist / nn_mean - T::one()
}

/// Aggregate Probabilistic Outlier Factor (nPLOF)
///
/// # Arguments
/// * `plofs` - Array of PLOF values
///
fn calc_nplof<T>(plofs: &[T], lambda: T) -> T
where
    T: Float + FloatConst + AddAssign + Sum + Debug,
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
    T: Float + FloatConst + AddAssign + Sum + Debug,
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
    use ndarray::Array1;
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

        // Convert the 2D array into nested Vecs, the input type expected by `local_outlier_probabilities`
        let points = array
            .outer_iter()
            .map(|row| row.to_vec())
            .collect::<Vec<Vec<f64>>>();

        // Calculate the local outlier probabilities
        let loop_score = local_outlier_probabilities(&points, 1000, 3, None).unwrap();

        assert_eq!(loop_score.len(), df.height());

        let loop_score_py = df
            .column("loop_score")
            .unwrap()
            .f64()
            .unwrap()
            .into_no_null_iter()
            .collect::<Array1<_>>();

        let rmse = Array1::from(loop_score)
            .root_mean_sq_err(&loop_score_py)
            .unwrap();

        // RMSE under 0.02 should be good enough
        assert!(rmse < 0.02, "RMSE > 0.02");
    }
}
