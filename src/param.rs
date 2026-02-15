use core::ops::{AddAssign, MulAssign};

use glam::{Vec2, Vec3};

/// A system parameter representing a degree of freedom.
///
/// Parameters define how a physical quantity evolves over time when integrated
/// with a derivative. Each parameter type must specify its derivative type and
/// implement the stepping operation.
///
/// # Required Operations
/// - `Copy` and `Default` for value semantics and initialization.
/// - `Add` and `Mul<f32>` for derivative type to support weighted accumulation.
/// - `step()` method to integrate the derivative over a time step.
///
/// # Provided Implementations
/// - `f32`, `Vec2`, `Vec3` for scalar and vector quantities.
/// - [`Rot2`], [`Rot3`] for rotations (see [`rot`] module).
pub trait Param: Clone + Default {
    /// The type of derivative for this parameter.
    ///
    /// Must support addition and scalar multiplication by `f32` to enable
    /// numerical integration algorithms.
    type Deriv: Deriv;

    /// Advance the parameter by integrating its derivative over time.
    ///
    /// This is the fundamental operation for numerical integration:
    /// `param_{n+1} = param_n + deriv * dt` (or its generalized equivalent).
    ///
    /// # Arguments
    /// * `deriv` - The derivative (rate of change) of the parameter.
    /// * `dt` - Time step over which to integrate.
    ///
    /// # Returns
    /// The new parameter value after the time step.
    fn step(&mut self, deriv: &Self::Deriv, dt: f32);
}

pub trait Deriv: Clone + Default + MulAssign<f32> + for<'a> AddAssign<&'a Self> {}

// Implement Param and Deriv for basic numeric types

impl Param for f32 {
    type Deriv = f32;
    fn step(&mut self, deriv: &f32, dt: f32) {
        *self += *deriv * dt
    }
}

impl Param for Vec2 {
    type Deriv = Vec2;
    fn step(&mut self, deriv: &Vec2, dt: f32) {
        *self += *deriv * dt
    }
}

impl Param for Vec3 {
    type Deriv = Vec3;
    fn step(&mut self, deriv: &Vec3, dt: f32) {
        *self += *deriv * dt
    }
}

impl Deriv for f32 {}
impl Deriv for Vec2 {}
impl Deriv for Vec3 {}
