//! Example systems for testing the System trait implementations.

use crate::{Euler, Rk4, Solver, System, Var, Visitor};
use glam::Vec2;

/// A simple particle system with position and velocity.
pub struct SimpleParticle<S: Solver> {
    pub position: Var<Vec2, S>,
    pub velocity: Var<Vec2, S>,
}

impl<S: Solver> SimpleParticle<S> {
    pub fn new(pos: Vec2, vel: Vec2) -> Self {
        Self {
            position: Var::new(pos),
            velocity: Var::new(vel),
        }
    }
}

impl<S: Solver> System<S> for SimpleParticle<S> {
    fn compute_derivs(&mut self, _: &S::Context) {
        // dx/dt = v
        self.position.deriv = *self.velocity;
        // dv/dt = 0 (no acceleration)
        self.velocity.deriv = Vec2::ZERO;
    }

    fn visit_vars<V: Visitor<S>>(&mut self, visitor: &mut V) {
        visitor.apply(&mut self.position);
        visitor.apply(&mut self.velocity);
    }
}

/// A particle with drag force: dv/dt = -k * v
pub struct ParticleWithDrag<S: Solver> {
    pub position: Var<Vec2, S>,
    pub velocity: Var<Vec2, S>,
    pub drag_coefficient: f32,
}

impl<S: Solver> ParticleWithDrag<S> {
    pub fn new(pos: Vec2, vel: Vec2, drag_coefficient: f32) -> Self {
        Self {
            position: Var::new(pos),
            velocity: Var::new(vel),
            drag_coefficient,
        }
    }
}

impl<S: Solver> System<S> for ParticleWithDrag<S> {
    fn compute_derivs(&mut self, _: &S::Context) {
        // dx/dt = v
        self.position.deriv = *self.velocity;
        // dv/dt = -k * v (drag force proportional to velocity)
        self.velocity.deriv = -*self.velocity * self.drag_coefficient;
    }

    fn visit_vars<V: Visitor<S>>(&mut self, visitor: &mut V) {
        visitor.apply(&mut self.position);
        visitor.apply(&mut self.velocity);
    }
}

/// Test the simple particle system.
#[test]
fn test_simple_particle_system() {
    let mut particle = SimpleParticle::new(Vec2::new(0.0, 0.0), Vec2::new(1.0, 2.0));
    let solver = Euler;

    // Take one step
    solver.solve_step(&mut particle, 0.1);

    // After one step with dt=0.1 and velocity (1, 2)
    let expected_pos = Vec2::new(0.1, 0.2);
    assert!((*particle.position - expected_pos).length() < 1e-6);

    // Velocity should remain constant
    assert!((*particle.velocity - Vec2::new(1.0, 2.0)).length() < 1e-6);
}

/// Test particle with drag.
#[test]
fn test_particle_with_drag() {
    let mut particle = ParticleWithDrag::new(
        Vec2::new(0.0, 0.0),
        Vec2::new(10.0, 0.0),
        0.5, // k = 0.5
    );
    let solver = Euler;
    let dt = 0.01;
    let steps = 100;

    // Integrate for 1 second total
    for _ in 0..steps {
        solver.solve_step(&mut particle, dt);
    }

    // Analytical solution for dv/dt = -k*v:
    // v(t) = v0 * exp(-k*t)
    // x(t) = (v0/k) * (1 - exp(-k*t))
    let t_total = dt * steps as f32;
    let k = 0.5;
    let v0 = 10.0;

    let expected_vel = v0 * (-k * t_total).exp();
    let expected_pos = (v0 / k) * (1.0 - (-k * t_total).exp());

    // Euler is first-order, so expect some error
    let vel_error = (*particle.velocity - Vec2::new(expected_vel, 0.0)).length();
    let pos_error = (*particle.position - Vec2::new(expected_pos, 0.0)).length();

    assert!(vel_error < 0.1, "Velocity error too large: {}", vel_error);
    assert!(pos_error < 0.1, "Position error too large: {}", pos_error);
}

/// Test that systems can be used with different solvers.
#[test]
fn test_system_solver_genericity() {
    // Create a simple system that works with any solver
    struct GenericSystem<S: Solver> {
        x: Var<f32, S>,
    }

    impl<S: Solver> System<S> for GenericSystem<S> {
        fn compute_derivs(&mut self, _ctx: &S::Context) {
            self.x.deriv = 1.0;
        }

        fn visit_vars<V: Visitor<S>>(&mut self, visitor: &mut V) {
            visitor.apply(&mut self.x);
        }
    }

    // Test with Euler
    let mut system_euler = GenericSystem::<Euler> { x: Var::new(0.0) };
    let solver_euler = Euler;
    solver_euler.solve_step(&mut system_euler, 0.1);
    assert!((*system_euler.x - 0.1).abs() < 1e-6);

    // Test with RK4
    let mut system_rk4 = GenericSystem::<Rk4> { x: Var::new(0.0) };
    let solver_rk4 = Rk4;
    solver_rk4.solve_step(&mut system_rk4, 0.1);
    // RK4 should be exact for constant derivative
    assert!((*system_rk4.x - 0.1).abs() < 1e-6);
}
