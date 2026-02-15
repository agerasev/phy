//! A generic, extensible framework for simulating temporal evolution of physical systems.
//!
//! This crate provides abstractions for solving first-order ordinary differential equations
//! (ODEs) using various numerical integration methods. It is designed to be `no_std` compatible
//! and focuses on performance and flexibility for physics simulations.
//!
//! # Core Concepts
//!
//! 1. **Parameters (`Param`)**: Degrees of freedom in a system (e.g., position, rotation).
//! 2. **Variables (`Var`)**: Containers that hold a parameter value, its derivative, and
//!    solver-specific storage.
//! 3. **Systems (`System`)**: Collections of variables with physics that define their evolution.
//! 4. **Solvers (`Solver`)**: Numerical integration algorithms (e.g., Euler, RK4).
//! 5. **Visitors (`Visitor`)**: Pattern for applying solver steps to variables.
//!
//! # Example Usage
//! ```
//! use phy::{Euler, Rk4, System, Var};
//! use glam::Vec2;
//!
//! struct Particle {
//!     position: Var<Vec2, Euler>,
//!     velocity: Var<Vec2, Euler>,
//! }
//!
//! impl System<Euler> for Particle {
//!     fn compute_derivs(&mut self, dt: f32) {
//!         // Simple kinematics: dx/dt = v
//!         self.position.deriv = *self.velocity;
//!         // dv/dt = 0 (no acceleration)
//!         self.velocity.deriv = Vec2::ZERO;
//!     }
//!
//!     fn visit_vars<V: phy::Visitor<Solver = Euler>>(&mut self, visitor: &mut V) {
//!         visitor.apply(&mut self.position);
//!         visitor.apply(&mut self.velocity);
//!     }
//! }
//! ```
//!
//! # Available Solvers
//! - [`Euler`]: First-order explicit Euler method (simple, low accuracy).
//! - [`Rk4`]: Fourth-order Runge-Kutta method (higher accuracy, more computation).
//!
//! # Available Parameters
//! - `f32`, `Vec2`, `Vec3` from `glam` for positions and linear quantities.
//! - [`Rot2`], [`Rot3`] from [`rot`] module for rotations.

#![cfg_attr(not(feature = "std"), no_std)]

mod euler;
mod rk4;
mod rot;
#[cfg(test)]
mod tests;
mod var;

pub use crate::{euler::Euler, rk4::Rk4, rot::*, var::Var};

use core::ops::{Add, Mul};
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
pub trait Param: Sized + Copy + Default {
    /// The type of derivative for this parameter.
    ///
    /// Must support addition and scalar multiplication by `f32` to enable
    /// numerical integration algorithms.
    type Deriv: Sized + Copy + Add<Output = Self::Deriv> + Mul<f32, Output = Self::Deriv> + Default;

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
    fn step(self, deriv: Self::Deriv, dt: f32) -> Self;
}

/// A visitor that applies solver-specific operations to variables.
///
/// The visitor pattern allows solvers to update variables in a system
/// without the system needing to know the solver's internal details.
/// Each solver defines its own visitor type that implements this trait.
pub trait Visitor {
    /// The solver type that uses this visitor.
    type Solver: Solver;

    /// Apply the visitor's operation to a single variable.
    ///
    /// This method is called by the system for each variable during
    /// a solver step. The visitor typically updates the variable's
    /// value using its derivative and solver-specific storage.
    fn apply<P: Param>(&mut self, v: &mut Var<P, Self::Solver>);
}

/// A physical system whose temporal evolution we want to simulate.
///
/// Systems contain variables (degrees of freedom) and define the physics
/// that governs their evolution through the `compute_derivs` method.
///
/// # Type Parameter
/// * `S` - The solver type used to integrate this system's equations.
pub trait System<S: Solver + ?Sized> {
    /// Compute derivatives for all variables in the system.
    ///
    /// This method defines the physics of the system by setting the
    /// `deriv` field of each variable based on the current state.
    ///
    /// # Arguments
    /// * `dt` - Time step for the upcoming integration.
    ///
    /// # Note on Time Dependence
    /// The `dt` parameter is primarily for numerical stability in
    /// algorithms that may need it (e.g., for handling constraints or
    /// stiff equations).
    fn compute_derivs(&mut self, dt: f32);

    /// Visit all variables in the system with the provided visitor.
    ///
    /// This method should call `visitor.apply()` for each variable
    /// in the system, allowing the solver to update them.
    fn visit_vars<V: Visitor<Solver = S>>(&mut self, visitor: &mut V);
}

/// A numerical integration algorithm for solving differential equations.
///
/// Solvers implement specific integration methods (e.g., Euler, RK4)
/// and define any additional storage required per variable.
pub trait Solver {
    /// Solver-specific storage type for variables of type `P`.
    ///
    /// This storage holds intermediate computations needed by the solver
    /// across integration steps. It must be `Copy` to allow efficient
    /// variable cloning and `Default` for initialization.
    ///
    /// # Examples
    /// - Euler method: `()` (no storage needed)
    /// - RK4 method: `Rk4Storage<P>` (stores initial value and accumulated derivatives)
    type Storage<P: Param>: Sized + Clone + Copy + Default;

    /// Perform one integration step for the given system.
    ///
    /// # Arguments
    /// * `system` - The system to integrate.
    /// * `dt` - Time step for the integration.
    fn solve_step<S: System<Self>>(&self, system: &mut S, dt: f32);
}

// Implement Param for basic numeric types

impl Param for f32 {
    type Deriv = f32;
    fn step(self, deriv: f32, dt: f32) -> Self {
        self + deriv * dt
    }
}

impl Param for Vec2 {
    type Deriv = Vec2;
    fn step(self, deriv: Vec2, dt: f32) -> Self {
        self + deriv * dt
    }
}

impl Param for Vec3 {
    type Deriv = Vec3;
    fn step(self, deriv: Vec3, dt: f32) -> Self {
        self + deriv * dt
    }
}
