use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("Slice of point not in contiguous memory, when {0}")]
    SliceNotContiguous(&'static str),
    #[error("KDTree error: {0}")]
    KdTreeError(#[from] kdtree::ErrorKind),
    #[error("Could not convert `{0}` to Float.")]
    KCastingError(String),
}
