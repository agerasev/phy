//! Test suite for the physics simulation crate.
//!
//! This module contains unit and integration tests for all major components:
//! - Param trait implementations
//! - Var struct and its operations
//! - Euler and RK4 solvers
//! - Rotation types and utility functions
//! - System trait examples

mod euler;
mod param;
mod rk4;
mod rot;
mod system;
