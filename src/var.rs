use crate::{Param, Solver};
use core::{
    fmt::{self, Debug, Formatter},
    ops::{Deref, DerefMut},
};

/// A variable representing a degree of freedom in a physical system.
///
/// Variables combine a parameter value with its derivative and solver-specific
/// storage. They serve as the primary interface between systems (which define
/// physics) and solvers (which perform numerical integration).
///
/// # Type Parameters
/// * `P` - The parameter type (e.g., `f32`, `Vec2`, [`Rot2`](crate::Rot2)).
/// * `S` - The solver type (e.g., [`Euler`](crate::Euler), [`Rk4`](crate::Rk4)).
///
/// # Fields
/// - `value`: Current value of the variable.
/// - `deriv`: Current derivative (rate of change) of the variable.
/// - `storage`: Solver-specific storage for intermediate computations.
///
/// # Example
/// ```
/// use phy::{Var, Euler};
/// use glam::Vec2;
///
/// // Create a position variable initialized to (1.0, 2.0)
/// let mut position = Var::<Vec2, Euler>::new(Vec2::new(1.0, 2.0));
///
/// // Access the value via dereference
/// assert_eq!(*position, Vec2::new(1.0, 2.0));
///
/// // Modify the value directly
/// *position = Vec2::new(3.0, 4.0);
///
/// // Set the derivative (velocity)
/// position.deriv = Vec2::new(1.0, 0.0);
/// ```
pub struct Var<P: Param, S: Solver + ?Sized> {
    /// The current value of the variable.
    pub value: P,
    /// The derivative (rate of change) of the variable.
    pub deriv: P::Deriv,
    /// Solver-specific storage for intermediate computations.
    ///
    /// This storage is used by solvers like RK4 to hold temporary values
    /// between integration stages. For Euler method, this is `()`.
    pub storage: S::Storage<P>,
}

impl<P: Param, S: Solver> Clone for Var<P, S> {
    fn clone(&self) -> Self {
        Self {
            value: self.value.clone(),
            deriv: self.deriv.clone(),
            storage: self.storage.clone(),
        }
    }
}

impl<P: Param, S: Solver> Copy for Var<P, S>
where
    P: Copy,
    P::Deriv: Copy,
    S::Storage<P>: Copy,
{
}

impl<P: Param, S: Solver> Default for Var<P, S> {
    /// Create a variable with default values.
    ///
    /// The value and derivative are set to their type's default,
    /// and storage is initialized using `Default::default()`.
    fn default() -> Self {
        Self {
            value: P::default(),
            deriv: P::Deriv::default(),
            storage: S::Storage::<P>::default(),
        }
    }
}

impl<P: Param, S: Solver> Var<P, S> {
    /// Create a new variable with the given initial value.
    ///
    /// The derivative is set to its default value, and storage
    /// is initialized using `Default::default()`.
    ///
    /// # Arguments
    /// * `value` - Initial value for the variable.
    pub fn new(value: P) -> Self {
        Var {
            value,
            deriv: Default::default(),
            storage: Default::default(),
        }
    }
}

impl<P: Param, S: Solver> Deref for Var<P, S> {
    type Target = P;

    /// Provides immutable access to the variable's value.
    ///
    /// This allows treating `Var<P, S>` as if it were `P` for reading.
    ///
    /// # Example
    /// ```
    /// use phy::{Var, Euler};
    /// use glam::Vec2;
    ///
    /// let var = Var::<Vec2, Euler>::new(Vec2::new(1.0, 2.0));
    /// let value: &Vec2 = &*var; // Dereference to access the value
    /// assert_eq!(*value, Vec2::new(1.0, 2.0));
    /// ```
    fn deref(&self) -> &Self::Target {
        &self.value
    }
}

impl<P: Param, S: Solver> DerefMut for Var<P, S> {
    /// Provides mutable access to the variable's value.
    ///
    /// This allows treating `Var<P, S>` as if it were `P` for modification.
    ///
    /// # Example
    /// ```
    /// use phy::{Var, Euler};
    /// use glam::Vec2;
    ///
    /// let mut var = Var::<Vec2, Euler>::new(Vec2::new(1.0, 2.0));
    /// *var = Vec2::new(3.0, 4.0); // Modify through dereference
    /// assert_eq!(*var, Vec2::new(3.0, 4.0));
    /// ```
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.value
    }
}

impl<P: Param, S: Solver> Debug for Var<P, S>
where
    P: Debug,
    P::Deriv: Debug,
    S::Storage<P>: Debug,
{
    /// Formats the variable for debugging.
    ///
    /// Shows the value, derivative, and storage fields.
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Var {{ value: {:?}, deriv: {:?}, storage: {:?} }}",
            &self.value, &self.deriv, &self.storage
        )
    }
}
