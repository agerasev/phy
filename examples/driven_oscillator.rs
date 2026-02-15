//! Driven harmonic oscillator simulation.
//!
//! This example simulates a damped harmonic oscillator with periodic external forcing.
//! The system exhibits resonance when the driving frequency matches the natural frequency.
//!
//! Physics equations (with m = 1, spring constant k, damping coefficient b):
//!   dx/dt = v
//!   dv/dt = -k*x - b*v + F0*cos(ω_d*t)
//!
//! Where:
//!   ω0 = √k      (natural frequency)
//!   ω_d          (driving frequency)
//!   F0           (driving amplitude)
//!
//! The visualization shows:
//!   - Numerical position and velocity
//!   - Visual trajectory showing oscillation amplitude changes

use phy::{Rk4, Solver, System, Var, Visitor};
use std::fmt::{self, Display, Formatter};

struct DrivenOscillator<S: Solver> {
    x: Var<f32, S>,    // position
    v: Var<f32, S>,    // velocity
    time: Var<f32, S>, // current time (evolves as d(time)/dt = 1)
}

const K: f32 = 25.0; // spring constant (natural frequency ω0 = 5)
const B: f32 = 0.1; // damping coefficient
const F0: f32 = 2.0; // driving amplitude
const OMEGA_D: f32 = 4.5; // driving frequency (slightly below resonance)

impl<S: Solver> System<S> for DrivenOscillator<S> {
    fn compute_derivs(&mut self, _: &S::Context) {
        // Position derivative: dx/dt = v
        self.x.deriv = *self.v;

        // Velocity derivative: dv/dt = -k*x - b*v + F0*cos(ω_d*t)
        let driving_force = F0 * (OMEGA_D * *self.time).cos();
        self.v.deriv = -K * *self.x - B * *self.v + driving_force;

        self.time.deriv = 1.0;
    }

    fn visit_vars<V: Visitor<S>>(&mut self, visitor: &mut V) {
        visitor.apply(&mut self.x);
        visitor.apply(&mut self.v);
        visitor.apply(&mut self.time);
    }
}

impl<S: Solver> Display for DrivenOscillator<S> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        // Print numerical state
        write!(
            f,
            "x:{:>6.3}, v:{:>6.3}, t:{:>4.1}|",
            *self.x, *self.v, *self.time
        )?;

        // Visualize oscillator position
        // Line represents positions from -1.0 to 1.0 (0.03125 units per character)
        let scale = 0.03125; // units per character
        let offset = 1.0; // position offset so -1 maps to character 0

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

    // Initial conditions: start at rest
    let mut system = DrivenOscillator {
        x: Var::new(0.0),
        v: Var::new(0.0),
        time: Var::new(0.0),
    };

    // Simulation loop: 120 frames with 10 RK4 steps per frame (dt=0.01 each)
    for _ in 0..120 {
        for _ in 0..10 {
            solver.solve_step(&mut system, 0.01);
        }
        println!("{}", system);
    }
}
