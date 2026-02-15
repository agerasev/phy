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
//! use phy::{Rk4, Solver, System, Var};
//! use glam::Vec2;
//!
//! struct Particle<S: Solver> {
//!     position: Var<Vec2, S>,
//!     velocity: Var<Vec2, S>,
//! }
//!
//! impl<S: Solver> System<S> for Particle<S> {
//!     fn compute_derivs(&mut self, _: &S::Context) {
//!         // Simple kinematics: dx/dt = v
//!         self.position.deriv = *self.velocity;
//!         // dv/dt = 0 (no acceleration)
//!         self.velocity.deriv = Vec2::ZERO;
//!     }
//!
//!     fn visit_vars<V: phy::Visitor<S>>(&mut self, visitor: &mut V) {
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
mod param;
mod rk4;
mod rot;
mod var;

#[cfg(test)]
mod tests;

pub use crate::{euler::Euler, param::*, rk4::Rk4, rot::*, var::*};

/// A visitor that applies solver-specific operations to variables.
///
/// The visitor pattern allows solvers to update variables in a system
/// without the system needing to know the solver's internal details.
/// Each solver defines its own visitor type that implements this trait.
pub trait Visitor<S: Solver + ?Sized> {
    /// Apply the visitor's operation to a single variable.
    ///
    /// This method is called by the system for each variable during
    /// a solver step. The visitor typically updates the variable's
    /// value using its derivative and solver-specific storage.
    fn apply<P: Param>(&mut self, v: &mut Var<P, S>);
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
    fn compute_derivs(&mut self, ctx: &S::Context);

    /// Visit all variables in the system with the provided visitor.
    ///
    /// This method should call `visitor.apply()` for each variable
    /// in the system, allowing the solver to update them.
    fn visit_vars<V: Visitor<S>>(&mut self, visitor: &mut V);
}

pub trait Context<S: Solver + ?Sized> {
    /// Time step for upcoming integration.
    ///
    /// # Note on Time Dependence
    ///
    /// This time step may differ from time step passed to [`System::solve_step`].
    /// You should not use it for counting time or so on.
    ///
    /// This value should be used primarily for numerical stability in
    /// algorithms that may need it (e.g., for handling constraints or
    /// stiff equations).
    fn time_step(&self) -> f32;
}

/// A numerical integration algorithm for solving differential equations.
///
/// Solvers implement specific integration methods (e.g., Euler, RK4)
/// and define any additional storage required per variable.
pub trait Solver {
    type Context: Context<Self>;

    /// Solver-specific storage type for variables of type `P`.
    ///
    /// This storage holds intermediate computations needed by the solver
    /// across integration steps. It must be `Copy` to allow efficient
    /// variable cloning and `Default` for initialization.
    ///
    /// # Examples
    /// - Euler method: `()` (no storage needed)
    /// - RK4 method: `Rk4Storage<P>` (stores initial value and accumulated derivatives)
    type Storage<P: Param>: Sized + Clone + Default;

    /// Perform one integration step for the given system.
    ///
    /// # Arguments
    /// * `system` - The system to integrate.
    /// * `dt` - Time step for the integration.
    fn solve_step<S: System<Self>>(&self, system: &mut S, dt: f32);
}
