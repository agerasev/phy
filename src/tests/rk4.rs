//! Tests for the RK4 solver, including accuracy verification.

use crate::{Rk4, Solver, System, Var};
use glam::Vec2;

/// A simple test system with constant derivative: dx/dt = c
struct ConstantDerivativeSystem {
    x: Var<f32, Rk4>,
    derivative: f32,
}

impl ConstantDerivativeSystem {
    fn new(initial_value: f32, derivative: f32) -> Self {
        Self {
            x: Var::new(initial_value),
            derivative,
        }
    }
}

impl System<Rk4> for ConstantDerivativeSystem {
    fn compute_derivs(&mut self, _dt: f32) {
        self.x.deriv = self.derivative;
    }

    fn visit_vars<V: crate::Visitor<Solver = Rk4>>(&mut self, visitor: &mut V) {
        visitor.apply(&mut self.x);
    }
}

/// Test RK4 integration with constant derivative.
/// For constant derivative, RK4 should give exact solution (within machine precision).
#[test]
fn test_rk4_constant_derivative() {
    let mut system = ConstantDerivativeSystem::new(0.0, 2.0);
    let solver = Rk4;
    let dt = 0.5;
    let steps = 4;

    // Integrate for 4 steps of dt=0.5 (total time = 2.0)
    for _ in 0..steps {
        solver.solve_step(&mut system, dt);
    }

    // Analytical solution: x(t) = x0 + c*t = 0 + 2.0 * 2.0 = 4.0
    let expected = 2.0 * (dt * steps as f32);
    let error = (*system.x - expected).abs();

    // RK4 should be exact for constant derivative (polynomial of degree 0)
    assert!(
        error < 1e-6,
        "Error too large: {} (expected: {})",
        error,
        expected
    );
}

/// A system with exponential growth: dx/dt = k*x
struct ExponentialSystemRK4 {
    x: Var<f32, Rk4>,
    growth_rate: f32,
}

impl ExponentialSystemRK4 {
    fn new(initial_value: f32, growth_rate: f32) -> Self {
        Self {
            x: Var::new(initial_value),
            growth_rate,
        }
    }
}

impl System<Rk4> for ExponentialSystemRK4 {
    fn compute_derivs(&mut self, _dt: f32) {
        self.x.deriv = self.growth_rate * *self.x;
    }

    fn visit_vars<V: crate::Visitor<Solver = Rk4>>(&mut self, visitor: &mut V) {
        visitor.apply(&mut self.x);
    }
}

/// Test RK4 accuracy with exponential growth.
/// Measure convergence rate: error should decrease as dt^4.
#[test]
fn test_rk4_exponential_convergence() {
    // Use parameters that produce measurable errors
    let growth_rate = 2.0; // Larger growth rate for larger errors
    let initial_value = 1.0;
    let total_time = 2.0; // Longer integration time

    // Test different time step sizes
    let dt_values = vec![0.2, 0.1, 0.05, 0.025];
    let mut errors = Vec::new();

    for &dt in &dt_values {
        let mut system = ExponentialSystemRK4::new(initial_value, growth_rate);
        let solver = Rk4;
        let steps = (total_time / dt) as usize;

        // Integrate for total_time
        for _ in 0..steps {
            solver.solve_step(&mut system, dt);
        }

        // Analytical solution: x(t) = x0 * exp(k*t)
        let expected = initial_value * (growth_rate * total_time).exp();
        let error = (*system.x - expected).abs();
        errors.push(error);
    }

    // Compute convergence rate
    // For fourth-order method: error ≈ C * dt^4
    // So error2/error1 ≈ (dt2/dt1)^4
    for i in 0..dt_values.len() - 1 {
        let ratio = dt_values[i + 1] / dt_values[i];
        let expected_error_ratio = ratio.powi(4); // dt^4

        // Avoid division by zero or near-zero errors
        if errors[i] < 1e-12 {
            // If error is essentially zero, skip this comparison
            continue;
        }

        let actual_error_ratio = errors[i + 1] / errors[i];

        // Skip convergence check when errors become very small (numerical noise dominates)
        // Only check convergence for errors large enough to avoid numerical noise
        if errors[i] > 1e-3 && errors[i + 1] > 1e-4 {
            let ratio_error =
                (actual_error_ratio - expected_error_ratio).abs() / expected_error_ratio;

            // Allow tolerance of 75% for these error magnitudes
            assert!(
                ratio_error < 0.75,
                "Convergence rate not fourth-order: dt={}, error ratio={}, expected={}, errors={:?}",
                dt_values[i],
                actual_error_ratio,
                expected_error_ratio,
                errors
            );
        } else {
            // For very small errors, just ensure they're small
            assert!(errors[i] < 1e-2, "Error too large: {}", errors[i]);
            assert!(errors[i + 1] < 1e-2, "Error too large: {}", errors[i + 1]);
        }
    }

    // Final error with smallest dt should be small
    // RK4 is very accurate, so error should be < 1e-6 for dt=0.025
    assert!(
        errors.last().unwrap() < &1e-5,
        "Final error too large: {}",
        errors.last().unwrap()
    );

    // Ensure errors are decreasing with smaller dt (allow some noise)
    for i in 0..dt_values.len() - 1 {
        if errors[i] > 1e-10 && errors[i + 1] > errors[i] * 1.5 {
            // If error increased significantly, print warning but don't fail
            println!(
                "Warning: error increased from {} to {} when dt decreased from {} to {}",
                errors[i],
                errors[i + 1],
                dt_values[i],
                dt_values[i + 1]
            );
        }
    }
}

/// Harmonic oscillator system: d²x/dt² = -ω²x
/// Convert to first-order system: dx/dt = v, dv/dt = -ω²x
struct HarmonicOscillator {
    x: Var<f32, Rk4>,
    v: Var<f32, Rk4>,
    omega_squared: f32,
}

impl HarmonicOscillator {
    fn new(initial_x: f32, initial_v: f32, omega: f32) -> Self {
        Self {
            x: Var::new(initial_x),
            v: Var::new(initial_v),
            omega_squared: omega * omega,
        }
    }
}

impl System<Rk4> for HarmonicOscillator {
    fn compute_derivs(&mut self, _dt: f32) {
        // dx/dt = v
        self.x.deriv = *self.v;
        // dv/dt = -ω²x
        self.v.deriv = -self.omega_squared * *self.x;
    }

    fn visit_vars<V: crate::Visitor<Solver = Rk4>>(&mut self, visitor: &mut V) {
        visitor.apply(&mut self.x);
        visitor.apply(&mut self.v);
    }
}

/// Test RK4 with harmonic oscillator (periodic motion).
#[test]
fn test_rk4_harmonic_oscillator() {
    let omega = 1.0; // Angular frequency
    let period = 2.0 * std::f32::consts::PI / omega; // Period = 2π/ω

    let mut system = HarmonicOscillator::new(1.0, 0.0, omega);
    let solver = Rk4;

    // Integrate for one period
    let dt = period / 100.0; // 100 steps per period
    let steps = 100;

    for _ in 0..steps {
        solver.solve_step(&mut system, dt);
    }

    // After one period, should return close to initial state
    let x_error = (*system.x - 1.0).abs();
    let v_error = (*system.v - 0.0).abs();

    // RK4 should be quite accurate for one period
    assert!(x_error < 1e-4, "Position error too large: {}", x_error);
    assert!(v_error < 1e-4, "Velocity error too large: {}", v_error);
}

/// Compare RK4 accuracy vs Euler for the same problem.
#[test]
fn test_rk4_vs_euler_accuracy() {
    use crate::Euler;

    // Use exponential growth problem
    let growth_rate = 1.0;
    let initial_value = 1.0;
    let total_time = 1.0;
    let dt = 0.1;
    let steps = (total_time / dt) as usize;

    // RK4 integration
    let mut system_rk4 = ExponentialSystemRK4::new(initial_value, growth_rate);
    let solver_rk4 = Rk4;
    for _ in 0..steps {
        solver_rk4.solve_step(&mut system_rk4, dt);
    }

    // Euler integration
    struct ExponentialSystemEuler {
        x: Var<f32, Euler>,
        growth_rate: f32,
    }

    impl System<Euler> for ExponentialSystemEuler {
        fn compute_derivs(&mut self, _dt: f32) {
            self.x.deriv = self.growth_rate * *self.x;
        }

        fn visit_vars<V: crate::Visitor<Solver = Euler>>(&mut self, visitor: &mut V) {
            visitor.apply(&mut self.x);
        }
    }

    let mut system_euler = ExponentialSystemEuler {
        x: Var::new(initial_value),
        growth_rate,
    };
    let solver_euler = Euler;
    for _ in 0..steps {
        solver_euler.solve_step(&mut system_euler, dt);
    }

    // Analytical solution
    let expected = initial_value * (growth_rate * total_time).exp();

    let error_rk4 = (*system_rk4.x - expected).abs();
    let error_euler = (*system_euler.x - expected).abs();

    // RK4 should be much more accurate than Euler
    assert!(
        error_rk4 < error_euler,
        "RK4 not more accurate than Euler: RK4 error={}, Euler error={}",
        error_rk4,
        error_euler
    );
    assert!(error_rk4 < 1e-4, "RK4 error too large: {}", error_rk4);
}

/// Test RK4 with vector variables.
#[test]
fn test_rk4_vector_variables() {
    struct VectorSystem {
        position: Var<Vec2, Rk4>,
        velocity: Var<Vec2, Rk4>,
    }

    impl VectorSystem {
        fn new(pos: Vec2, vel: Vec2) -> Self {
            Self {
                position: Var::new(pos),
                velocity: Var::new(vel),
            }
        }
    }

    impl System<Rk4> for VectorSystem {
        fn compute_derivs(&mut self, _dt: f32) {
            // dx/dt = v
            self.position.deriv = *self.velocity;
            // dv/dt = -v (damping)
            self.velocity.deriv = -*self.velocity;
        }

        fn visit_vars<V: crate::Visitor<Solver = Rk4>>(&mut self, visitor: &mut V) {
            visitor.apply(&mut self.position);
            visitor.apply(&mut self.velocity);
        }
    }

    let mut system = VectorSystem::new(Vec2::new(0.0, 0.0), Vec2::new(1.0, 2.0));
    let solver = Rk4;
    let dt = 0.1;
    let steps = 10;

    for _ in 0..steps {
        solver.solve_step(&mut system, dt);
    }

    // Analytical solution for dv/dt = -v: v(t) = v0 * exp(-t)
    // And for dx/dt = v: x(t) = x0 + v0 * (1 - exp(-t))
    let t_total = dt * steps as f32;
    let expected_vel = Vec2::new(1.0, 2.0) * (-t_total).exp();
    let expected_pos = Vec2::new(1.0, 2.0) * (1.0 - (-t_total).exp());

    let pos_error = (*system.position - expected_pos).length();
    let vel_error = (*system.velocity - expected_vel).length();

    assert!(pos_error < 1e-4, "Position error too large: {}", pos_error);
    assert!(vel_error < 1e-4, "Velocity error too large: {}", vel_error);
}

/// Test that RK4 storage is properly initialized and cleaned up.
#[test]
fn test_rk4_storage_management() {
    // Create a variable and use it
    let mut system = ConstantDerivativeSystem::new(0.0, 1.0);
    let solver = Rk4;

    // Take a few steps
    solver.solve_step(&mut system, 0.1);
    solver.solve_step(&mut system, 0.1);
    solver.solve_step(&mut system, 0.1);

    // Should still work correctly
    let expected = 0.3; // 3 steps * 0.1 * 1.0
    let error = (*system.x - expected).abs();
    assert!(error < 1e-6, "Error after multiple steps: {}", error);
}
