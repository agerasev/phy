//! Bouncing ball simulation with ground contact modeled as spring-damper.
//!
//! This example simulates a ball under gravity that bounces when it hits the ground.
//! The ground contact is modeled as a spring-damper system that activates only
//! when the ball is below the ground level (position < 0).
//!
//! Physics equations:
//!   dx/dt = v
//!   dv/dt = -g + (k - f*v) * max(0, -x)   (spring-damper activates when x < 0)
//!
//! Where:
//!   g = 9.8 m/s²  (gravity)
//!   k = 1000 N/m  (spring stiffness)
//!   f = 100 N·s/m (damping coefficient)
//!
//! The visualization shows:
//!   - Numerical position and velocity
//!   - Visual trajectory with '*' representing the ball's position along a line

use phy::{Rk4, Solver, System, Var, Visitor};
use std::fmt::{self, Display, Formatter};

struct BouncingBall<S: Solver> {
    pos: Var<f32, S>,
    vel: Var<f32, S>,
}

const G: f32 = 9.8; // gravitational acceleration (m/s²)
const K: f32 = 1000.0; // ground spring stiffness (N/m)
const F: f32 = 100.0; // ground damping coefficient (N·s/m)

impl<S: Solver> System<S> for BouncingBall<S> {
    fn compute_derivs(&mut self, _: &S::Context) {
        // Position derivative: dx/dt = v
        self.pos.deriv = *self.vel;

        // Velocity derivative: dv/dt = -g + contact_force
        // Contact force activates only when ball is below ground (pos < 0)
        // Force = (K - F*v) * (-pos)  [spring with velocity-dependent damping]
        self.vel.deriv += -G + (K - F * *self.vel) * (-self.pos.min(0.0));
    }

    fn visit_vars<V: Visitor<S>>(&mut self, visitor: &mut V) {
        visitor.apply(&mut self.pos);
        visitor.apply(&mut self.vel);
    }
}

impl<S: Solver> Display for BouncingBall<S> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        // Print numerical state: position and velocity
        write!(f, "pos:{:>4.1}, vel:{:>5.1}|", *self.pos, *self.vel)?;

        // Visualize ball position along a 64-character line
        // Each character represents 0.16 units of position
        for i in 0..64 {
            if i as f32 * 0.16 < *self.pos {
                write!(f, " ")?;
            } else {
                write!(f, "*")?;
                break;
            }
        }
        Ok(())
    }
}

fn main() {
    let solver = Rk4;

    // Initial conditions: ball starts at height 10 with zero velocity
    let mut system = BouncingBall {
        pos: Var::new(10.0),
        vel: Var::new(0.0),
    };

    // Simulation loop: 40 frames with 10 RK4 steps per frame (dt=0.01 each)
    for _ in 0..40 {
        for _ in 0..10 {
            solver.solve_step(&mut system, 0.01);
        }
        println!("{}", system);
    }
}
