//! Coupled harmonic oscillators simulation.
//!
//! This example simulates two masses connected by springs to fixed walls
//! and to each other, forming a system of coupled harmonic oscillators.
//!
//! The visualization shows:
//!   - Numerical positions and velocities for both masses
//!   - Visual trajectory with '*' for mass1 and '#' for mass2 on same line

use phy::{Rk4, Solver, System, Var, Visitor};
use std::fmt::{self, Display, Formatter};

struct CoupledOscillators<S: Solver> {
    m1: f32,         // mass 1
    x1: Var<f32, S>, // position of mass 1
    v1: Var<f32, S>, // velocity of mass 1
    m2: f32,         // mass 2
    x2: Var<f32, S>, // position of mass 2
    v2: Var<f32, S>, // velocity of mass 2
}

const WALL: f32 = 3.0; // walls offset

const K: f32 = 10.0; // spring constant
const L: f32 = 2.0; // spring equilibrium length

impl<S: Solver> System<S> for CoupledOscillators<S> {
    fn compute_derivs(&mut self, _: &S::Context) {
        // Mass 1: dx1/dt = v1
        self.x1.deriv = *self.v1;
        self.v1.deriv = -K / self.m1
            * [
                *self.x1 + WALL - L,     // left wall
                *self.x1 - *self.x2 + L, // mass 2
            ]
            .into_iter()
            .sum::<f32>();

        // Mass 2: dx2/dt = v2
        self.x2.deriv = *self.v2;
        self.v2.deriv = -K / self.m2
            * [
                *self.x2 - WALL + L,     // right wall
                *self.x2 - *self.x1 - L, // mass 1
            ]
            .into_iter()
            .sum::<f32>();
    }

    fn visit_vars<V: Visitor<S>>(&mut self, visitor: &mut V) {
        visitor.apply(&mut self.x1);
        visitor.apply(&mut self.v1);
        visitor.apply(&mut self.x2);
        visitor.apply(&mut self.v2);
    }
}

impl<S: Solver> Display for CoupledOscillators<S> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        // Print numerical state
        write!(
            f,
            "x1:{:>5.2}, v1:{:>5.2}, x2:{:>5.2}, v2:{:>5.2}",
            *self.x1, *self.v1, *self.x2, *self.v2
        )?;

        // Visualize both masses on a 64-character line
        let scale = 0.0625; // units per character
        let offset = WALL; // position offset so -WALL maps to character 0

        let pos1_char = ((*self.x1 + offset) / scale).floor() as usize;
        let pos2_char = ((*self.x2 + offset) / scale).floor() as usize;

        write!(f, "|")?;
        for i in 0..((2.0 * offset / scale) as usize + 1) {
            if i == pos1_char {
                write!(f, "*")?;
            } else if i == pos2_char {
                write!(f, "#")?;
            } else {
                write!(f, " ")?;
            }
        }
        write!(f, "|")?;
        Ok(())
    }
}

fn main() {
    let solver = Rk4;

    // Initial conditions: mass1 at left, mass2 at right with initial velocity
    let mut system = CoupledOscillators {
        m1: 1.0,
        x1: Var::new(-2.0),
        v1: Var::new(0.0),
        m2: 3.0,
        x2: Var::new(1.0),
        v2: Var::new(-3.0),
    };
    println!("{}", system);

    // Simulation loop: 80 frames with 10 RK4 steps per frame (dt=0.01 each)
    for _ in 0..40 {
        for _ in 0..10 {
            solver.solve_step(&mut system, 0.01);
        }
        println!("{}", system);
    }
}
