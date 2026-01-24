#![no_std]

#[cfg(feature = "std")]
extern crate std;

mod euler;
mod rk4;
mod rot;
mod var;

pub use crate::{euler::Euler, rk4::Rk4, rot::*, var::Var};

use core::ops::{Add, Mul};
use glam::{Vec2, Vec3};

/// System parameter (degrees of freedom).
pub trait Param: Sized + Copy + Default {
    type Deriv: Sized + Copy + Add<Output = Self::Deriv> + Mul<f32, Output = Self::Deriv> + Default;
    fn step(self, deriv: Self::Deriv, dt: f32) -> Self;
}

pub trait Visitor {
    type Solver: Solver;
    fn apply<P: Param>(&mut self, v: &mut Var<P, Self::Solver>);
}

pub trait System<S: Solver + ?Sized> {
    fn compute_derivs(&mut self, dt: f32);
    fn visit_vars<V: Visitor<Solver = S>>(&mut self, visitor: &mut V);
}

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

/// Temporal differential equation solver.
pub trait Solver {
    type Storage<P: Param>: Sized + Clone + Copy + Default;
    fn solve_step<S: System<Self>>(&self, system: &mut S, dt: f32);
}
