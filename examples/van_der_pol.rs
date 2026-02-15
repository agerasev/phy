//! Van der Pol oscillator simulation.
//!
//! The Van der Pol oscillator is a non-conservative oscillator with nonlinear damping.
//! It exhibits self-excited oscillations (limit cycle) where energy is dissipated
//! at large amplitudes and injected at small amplitudes.
//!
//! Physics equations:
//!   dx/dt = v
//!   dv/dt = μ*(1 - x²)*v - x
//!
//! Where μ is the damping parameter:
//!   - μ > 0: System exhibits self-excited oscillations
//!   - μ = 0: Reduces to simple harmonic oscillator
//!
//! The system always converges to a stable limit cycle regardless of initial conditions.
//!
//! The visualization shows:
//!   - Numerical position and velocity
//!   - Visual trajectory showing limit cycle behavior

use phy::{Rk4, Solver, System, Var, Visitor};
use std::fmt::{self, Display, Formatter};

struct VanDerPol<S: Solver> {
    x: Var<f32, S>, // position
    v: Var<f32, S>, // velocity
}

const MU: f32 = 2.0; // damping parameter

impl<S: Solver> System<S> for VanDerPol<S> {
    fn compute_derivs(&mut self, _: &S::Context) {
        // Position derivative: dx/dt = v
        self.x.deriv = *self.v;

        // Velocity derivative: dv/dt = μ*(1 - x²)*v - x
        let nonlinear_damping = MU * (1.0 - *self.x * *self.x) * *self.v;
        self.v.deriv = nonlinear_damping - *self.x;
    }

    fn visit_vars<V: Visitor<S>>(&mut self, visitor: &mut V) {
        visitor.apply(&mut self.x);
        visitor.apply(&mut self.v);
    }
}

impl<S: Solver> Display for VanDerPol<S> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        // Print numerical state
        write!(f, "x:{:>6.3}, v:{:>6.3}|", *self.x, *self.v)?;

        // Visualize position
        // Line represents positions from -3.0 to 3.0 (0.09375 units per character)
        let scale = 0.09375; // units per character
        let offset = 3.0; // position offset so -3 maps to character 0

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

    // Initial conditions: start with small displacement
    let mut system = VanDerPol {
        x: Var::new(0.1),
        v: Var::new(0.0),
    };

    // Simulation loop: 80 frames with 10 RK4 steps per frame (dt=0.02 each)
    for _ in 0..80 {
        for _ in 0..10 {
            solver.solve_step(&mut system, 0.02);
        }
        println!("{}", system);
    }
}
