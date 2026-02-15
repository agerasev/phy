//! Simple pendulum simulation.
//!
//! This example simulates a simple pendulum with nonlinear dynamics.
//! The pendulum consists of a mass attached to a rigid rod of length L.
//!
//! Physics equations (with m = 1, L = 1, g = 9.8):
//!   dθ/dt = ω
//!   dω/dt = -(g/L) * sin(θ)
//!
//! For small angles (θ ≪ 1), sin(θ) ≈ θ, giving simple harmonic motion.
//! For larger angles, the motion is nonlinear with period depending on amplitude.
//!
//! The visualization shows:
//!   - Numerical angle (θ) and angular velocity (ω)
//!   - Visual representation of pendulum bob position along an arc

use phy::{Rk4, Solver, System, Var, Visitor};
use std::fmt::{self, Display, Formatter};

struct Pendulum<S: Solver> {
    theta: Var<f32, S>, // angle from vertical (radians)
    omega: Var<f32, S>, // angular velocity (radians/s)
}

const G: f32 = 9.8; // gravitational acceleration (m/s²)
const L: f32 = 1.0; // pendulum length (m)

impl<S: Solver> System<S> for Pendulum<S> {
    fn compute_derivs(&mut self, _: &S::Context) {
        // Angle derivative: dθ/dt = ω
        self.theta.deriv = *self.omega;
        // Angular velocity derivative: dω/dt = -(g/L) * sin(θ)
        self.omega.deriv = -(G / L) * (*self.theta).sin();
    }

    fn visit_vars<V: Visitor<S>>(&mut self, visitor: &mut V) {
        visitor.apply(&mut self.theta);
        visitor.apply(&mut self.omega);
    }
}

impl<S: Solver> Display for Pendulum<S> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        // Print numerical state
        write!(f, "θ:{:>5.2}, ω:{:>5.2}|", *self.theta, *self.omega)?;

        // Visualize pendulum bob position along an arc
        // Convert angle to horizontal position for display
        // θ = 0 (vertical) maps to center of line, ±π/2 maps to edges
        let x_pos = (*self.theta).sin(); // horizontal position (-1 to 1)
        let scale = 0.03125; // units per character (64 chars for range -1 to 1)
        let offset = 1.0; // position offset so -1 maps to character 0

        let bob_char = ((x_pos + offset) / scale).floor() as usize;

        for i in 0..64 {
            if i == bob_char.min(63) {
                write!(f, "O")?; // pendulum bob
            } else {
                write!(f, " ")?;
            }
        }
        Ok(())
    }
}

fn main() {
    let solver = Rk4;

    // Initial conditions: start with 120° angle (2π/3) and zero angular velocity
    let mut system = Pendulum {
        theta: Var::new(2.0 * std::f32::consts::PI / 3.0), // 120 degrees
        omega: Var::new(0.0),
    };

    // Simulation loop: 40 frames with 10 RK4 steps per frame (dt=0.01 each)
    for _ in 0..40 {
        for _ in 0..10 {
            solver.solve_step(&mut system, 0.01);
        }
        println!("{}", system);
    }
}
