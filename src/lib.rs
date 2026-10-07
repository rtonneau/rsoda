//! rsoda: a Rust port of the LSODA ODE solver (automatic Adams/BDF switching).

pub mod coefficients;
pub mod error;
pub mod interpolate;
pub mod linalg;
pub mod lsoda;
pub mod norms;
pub mod options;
pub mod prja;
pub mod solsy;
pub mod state;
pub mod stoda;
pub mod system;

/// Machine constants, matching the values DLSODA obtains from `DUMACH`.
pub mod consts {
    /// Unit roundoff (`UROUND`): the smallest `u` with `1 + u > 1`.
    pub const UROUND: f64 = f64::EPSILON;
    /// Largest order of the nonstiff (Adams) method.
    pub const MORD_NONSTIFF: usize = 12;
    /// Largest order of the stiff (BDF) method.
    pub const MORD_STIFF: usize = 5;
    /// Default maximum number of internal steps per call (`MXSTP0`).
    pub const MXSTP0: usize = 10_000;
    /// Default maximum number of `T + H = T` warnings (`MXHNL0`).
    pub const MXHNL0: usize = 10;
}

pub use coefficients::Method;
pub use error::{InputError, LsodaError};
pub use lsoda::{Lsoda, MethodSwitch, Warning};
pub use options::{JacobianKind, Options, OptionsBuilder, Tol, Tolerance};
pub use system::{ClosureSystem, OdeSystem};
