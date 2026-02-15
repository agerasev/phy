//! Tests for the Euler solver.

use crate::{Euler, Solver, System, Var};
use glam::Vec2;

/// A simple test system with a single variable.
struct SimpleSystem {
    x: Var<f32, Euler>,
}

impl SimpleSystem {
    fn new(initial_value: f32) -> Self {
        Self {
            x: Var::new(initial_value),
        }
    }
}

impl System<Euler> for SimpleSystem {
    fn compute_derivs(&mut self, _dt: f32) {
        // Constant derivative: dx/dt = 1.0
        self.x.deriv = 1.0;
    }

    fn visit_vars<V: crate::Visitor<Solver = Euler>>(&mut self, visitor: &mut V) {
        visitor.apply(&mut self.x);
    }
}

/// Test Euler integration with constant derivative.
#[test]
fn test_euler_constant_derivative() {
    let mut system = SimpleSystem::new(0.0);
    let solver = Euler;
    let dt = 0.1;
    let steps = 10;

    // Integrate for 10 steps
    for _ in 0..steps {
        solver.solve_step(&mut system, dt);
    }

    // After 10 steps of dt=0.1 with dx/dt=1.0
    // x = x0 + 1.0 * (10 * 0.1) = 0.0 + 1.0 = 1.0
    let expected = 1.0;
    assert!((*system.x - expected).abs() < 1e-6);
}

/// A system with exponential growth: dx/dt = k*x
struct ExponentialSystem {
    x: Var<f32, Euler>,
    growth_rate: f32,
}

impl ExponentialSystem {
    fn new(initial_value: f32, growth_rate: f32) -> Self {
        Self {
            x: Var::new(initial_value),
            growth_rate,
        }
    }
}

impl System<Euler> for ExponentialSystem {
    fn compute_derivs(&mut self, _dt: f32) {
        self.x.deriv = self.growth_rate * *self.x;
    }

    fn visit_vars<V: crate::Visitor<Solver = Euler>>(&mut self, visitor: &mut V) {
        visitor.apply(&mut self.x);
    }
}

/// Test Euler integration with exponential growth.
#[test]
fn test_euler_exponential_growth() {
    let mut system = ExponentialSystem::new(1.0, 0.1); // dx/dt = 0.1*x
    let solver = Euler;
    let dt = 0.01;
    let steps = 100;

    // Integrate for 100 steps of dt=0.01 (total time = 1.0)
    for _ in 0..steps {
        solver.solve_step(&mut system, dt);
    }

    // Analytical solution: x(t) = x0 * exp(k*t)
    // t = 1.0, k = 0.1, x0 = 1.0
    let t_total = dt * steps as f32;
    let expected = 1.0 * (0.1 * t_total).exp();

    // Euler method is first-order, so with small dt we should be reasonably accurate
    let error = (*system.x - expected).abs();
    assert!(
        error < 0.005,
        "Error too large: {} (expected: {})",
        error,
        expected
    );
}

/// A system with multiple variables (2D position and velocity).
struct Particle2D {
    position: Var<Vec2, Euler>,
    velocity: Var<Vec2, Euler>,
}

impl Particle2D {
    fn new(pos: Vec2, vel: Vec2) -> Self {
        Self {
            position: Var::new(pos),
            velocity: Var::new(vel),
        }
    }
}

impl System<Euler> for Particle2D {
    fn compute_derivs(&mut self, _dt: f32) {
        // dx/dt = v
        self.position.deriv = *self.velocity;
        // dv/dt = 0 (no acceleration)
        self.velocity.deriv = Vec2::ZERO;
    }

    fn visit_vars<V: crate::Visitor<Solver = Euler>>(&mut self, visitor: &mut V) {
        visitor.apply(&mut self.position);
        visitor.apply(&mut self.velocity);
    }
}

/// Test Euler integration with multiple variables.
#[test]
fn test_euler_multiple_variables() {
    let mut system = Particle2D::new(Vec2::new(0.0, 0.0), Vec2::new(1.0, 2.0));
    let solver = Euler;
    let dt = 0.1;
    let steps = 10;

    // Integrate for 10 steps
    for _ in 0..steps {
        solver.solve_step(&mut system, dt);
    }

    // After 10 steps of dt=0.1 with constant velocity (1.0, 2.0)
    // position = initial + velocity * (dt * steps)
    let expected_pos = Vec2::new(0.0, 0.0) + Vec2::new(1.0, 2.0) * (dt * steps as f32);
    assert!((*system.position - expected_pos).length() < 1e-6);

    // Velocity should remain constant
    assert!((*system.velocity - Vec2::new(1.0, 2.0)).length() < 1e-6);
}

/// Test that derivatives are reset after each step.
#[test]
fn test_euler_derivative_reset() {
    let mut system = SimpleSystem::new(0.0);
    let solver = Euler;

    // First step
    solver.solve_step(&mut system, 0.1);

    // After step, derivative should be reset to default (0.0)
    assert!(system.x.deriv.abs() < 1e-6);

    // Second step - compute_derivs will set it to 1.0 again
    solver.solve_step(&mut system, 0.1);

    // After second step, derivative should be reset again
    assert!(system.x.deriv.abs() < 1e-6);
}

/// Test zero time step.
#[test]
fn test_euler_zero_dt() {
    let mut system = SimpleSystem::new(5.0);
    let solver = Euler;

    // With dt = 0, value should not change
    solver.solve_step(&mut system, 0.0);
    assert!((*system.x - 5.0).abs() < 1e-6);
}
