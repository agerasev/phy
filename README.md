# phy

A generic extendable first-order differential equation solver.

[![Crates.io](https://img.shields.io/crates/v/phy)](https://crates.io/crates/phy)
[![Documentation](https://docs.rs/phy/badge.svg)](https://docs.rs/phy)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/agerasev/phy/actions/workflows/ci.yml/badge.svg)](https://github.com/agerasev/phy/actions/workflows/ci.yml)

## Overview

The crate provides a framework for solving first-order differential equations using various numerical integration methods. It's designed to be generic and extensible, supporting different parameter types and solvers.

## Features

- Different parameter types and their derivatives
- Generic solvers. For now there are Euler's method and Runge-Kutta 4th order (RK4)
- Built-in support for 2D and 3D rotations with proper angular mathematics
- Easy to add new solvers and parameter types
- Uses `#![no_std]` and `glam` crate for math operations

## Usage

Example usage with a simple system:

```rust
use phy::{Euler, Rk4, Var, System, Visitor};

// Define your system parameters
struct MySystem {
    position: Var<f32, Euler>,
    velocity: Var<f32, Euler>,
}

impl System<Euler> for MySystem {
    fn compute_derivs(&mut self, dt: f32) {
        // Compute derivatives for the system
        self.position.deriv = self.velocity.value;
        self.velocity.deriv = -9.81; // Gravity
    }

    fn visit_vars<V: Visitor<Solver = Euler>>(&mut self, visitor: &mut V) {
        visitor.apply(&mut self.position);
        visitor.apply(&mut self.velocity);
    }
}
```

## Examples

The crate includes several example simulations that demonstrate different physical systems. Each example produces a single-line output showing numerical state and visual trajectory:

- **Bouncing Ball** (`examples/bouncing_ball.rs`): A ball under gravity with spring-damper ground contact.
- **Coupled Oscillators** (`examples/coupled_oscillators.rs`): Two masses connected by springs to walls and each other, showing complex energy transfer patterns with different masses.
- **Simple Pendulum** (`examples/pendulum.rs`): Nonlinear pendulum with large-angle dynamics, starting at 60°.
- **Driven Harmonic Oscillator** (`examples/driven_oscillator.rs`): Damped oscillator with periodic forcing near resonance frequency.
- **Van der Pol Oscillator** (`examples/van_der_pol.rs`): Self-exciting nonlinear oscillator that converges to a stable limit cycle.
- **Duffing Oscillator** (`examples/duffing.rs`): Nonlinear oscillator with cubic stiffness showing chaotic behavior under periodic forcing.

Each example shows:
- Numerical state values (positions, velocities, angles)
- Visual trajectory using spaces and symbols (`*`, `O`, `#`)
- Single-line output format for clear animation

The examples demonstrate the crate's ability to simulate diverse 1D systems with non-trivial trajectories.
```
