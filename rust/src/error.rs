//! Errors of the core (native/include/hedonic_core.h, `hig_error_t`).

use std::fmt;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// Invalid argument or input outside the model's domain.
    Inval,
    /// Out of memory.
    NoMem,
    /// A size, or the arithmetic of the game, would overflow.
    Overflow,
    /// The interruption callback asked to stop.
    Interrupted,
    /// The random-number source failed.
    Callback,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Error {
    pub kind: Kind,
    pub message: String,
}

impl Error {
    pub fn new(kind: Kind, message: impl Into<String>) -> Self {
        Error { kind, message: message.into() }
    }

    pub fn inval(message: impl Into<String>) -> Self {
        Error::new(Kind::Inval, message)
    }

    pub fn overflow(message: impl Into<String>) -> Self {
        Error::new(Kind::Overflow, message)
    }

    pub fn nomem() -> Self {
        Error::new(Kind::NoMem, "Out of memory.")
    }

    pub fn interrupted(message: impl Into<String>) -> Self {
        Error::new(Kind::Interrupted, message)
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;

/// A vector of `count` copies of `value`, or an error instead of an abort
/// when the memory is not there (the C core's hig_alloc).
pub fn filled<T: Clone>(count: usize, value: T) -> Result<Vec<T>> {
    let mut v = Vec::new();
    v.try_reserve_exact(count).map_err(|_| Error::nomem())?;
    v.resize(count, value);
    Ok(v)
}

/// An empty vector with room for `count` elements.
pub fn with_capacity<T>(count: usize) -> Result<Vec<T>> {
    let mut v = Vec::new();
    v.try_reserve_exact(count).map_err(|_| Error::nomem())?;
    Ok(v)
}

/// Pushes with a recoverable error on allocation failure.
#[inline]
pub fn push<T>(v: &mut Vec<T>, value: T) -> Result<()> {
    if v.len() == v.capacity() {
        v.try_reserve(1).map_err(|_| Error::nomem())?;
    }
    v.push(value);
    Ok(())
}
