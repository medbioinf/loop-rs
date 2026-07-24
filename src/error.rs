use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("Input data is empty")]
    EmptyInput,
    #[error("Expected point with {expected} dimensions but row {row} has {found}")]
    DimensionMismatch {
        expected: usize,
        found: usize,
        row: usize,
    },
    #[error("KDTree error: {0}")]
    KdTreeError(#[from] kdtree::ErrorKind),
    #[error("Could not convert `{0}` to Float.")]
    CastingError(String),
    #[error("Failed to extract point: {0}")]
    PointExtraction(String),
}
