//! Duffing oscillator simulation.
//!
//! The Duffing oscillator is a nonlinear oscillator with a cubic stiffness term.
//! It exhibits complex behavior including period doubling, chaos, and multiple
//! equilibrium points.
//!
//! Physics equations (with damping and periodic forcing):
//!   dx/dt = v
//!   dv/dt = -δ*v - α*x - β*x³ + γ*cos(ω*t)
//!
//! Where:
//!   δ: damping coefficient
//!   α: linear stiffness (can be negative for bistable systems)
//!   β: nonlinear stiffness (positive for hardening spring)
//!   γ: forcing amplitude
//!   ω: forcing frequency
//!
//! This example shows a chaotic regime with strange attractor behavior.
//!
//! The visualization shows:
//!   - Numerical position and velocity
//!   - Visual trajectory showing chaotic oscillations

use phy::{Rk4, Solver, System, Var, Visitor};
use std::fmt::{self, Display, Formatter};

struct Duffing<S: Solver> {
    x: Var<f32, S>,    // position
    v: Var<f32, S>,    // velocity
    time: Var<f32, S>, // current time for driving force
}

const DELTA: f32 = 0.3; // damping coefficient
const ALPHA: f32 = -1.0; // linear stiffness (negative for bistable)
const BETA: f32 = 1.0; // nonlinear stiffness
const GAMMA: f32 = 0.5; // forcing amplitude
const OMEGA: f32 = 1.2; // forcing frequency

impl<S: Solver> System<S> for Duffing<S> {
    fn compute_derivs(&mut self, _: &S::Context) {
        // Position derivative: dx/dt = v
        self.x.deriv = *self.v;

        // Velocity derivative: dv/dt = -δ*v - α*x - β*x³ + γ*cos(ω*t)
        let forcing = GAMMA * (OMEGA * *self.time).cos();
        self.v.deriv =
            -DELTA * *self.v - ALPHA * *self.x - BETA * *self.x * *self.x * *self.x + forcing;

        // Update time for next step
        self.time.deriv = 1.0;
    }

    fn visit_vars<V: Visitor<S>>(&mut self, visitor: &mut V) {
        visitor.apply(&mut self.x);
        visitor.apply(&mut self.v);
        visitor.apply(&mut self.time);
    }
}

impl<S: Solver> Display for Duffing<S> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        // Print numerical state
        write!(
            f,
            "x:{:>6.3}, v:{:>6.3}, t:{:>4.1}|",
            *self.x, *self.v, *self.time
        )?;

        // Visualize position
        // Line represents positions from -2.0 to 2.0 (0.0625 units per character)
        let scale = 0.0625; // units per character
        let offset = 2.0; // position offset so -2 maps to character 0

        let pos_char = ((*self.x + offset) / scale).floor() as usize;

        for i in 0..64 {
            if i == pos_char.min(63) {
                write!(f, "*")?;
            } else {
                write!(f, " ")?;
            }
        }
        Ok(())
    }
}

fn main() {
    let solver = Rk4;

    // Initial conditions: start near one equilibrium point
    let mut system = Duffing {
        x: Var::new(1.0),
        v: Var::new(0.0),
        time: Var::new(0.0),
    };

    // Simulation loop: 100 frames with 10 RK4 steps per frame (dt=0.05 each)
    for _ in 0..100 {
        for _ in 0..10 {
            solver.solve_step(&mut system, 0.05);
        }
        println!("{}", system);
    }
}
