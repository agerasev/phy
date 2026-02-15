use crate::{Context, Param, Solver, System, Var, Visitor};

/// The explicit Euler method for numerical integration.
///
/// Euler's method is a first-order numerical procedure for solving
/// ordinary differential equations (ODEs) with a given initial value.
/// It is simple but has relatively low accuracy and stability compared
/// to higher-order methods like RK4.
///
/// # Algorithm
///
/// For a differential equation dy/dt = f(t, y), Euler's method computes:
/// ```text
/// y_{n+1} = y_n + f(t_n, y_n) * dt
/// ```
///
/// where `dt` is the time step.
///
/// # Stability and Accuracy
///
/// Euler's method has truncation error O(dt²) per step and O(dt) globally.
/// It can be unstable for stiff equations or large time steps.
/// Consider using [`Rk4`](crate::Rk4) for higher accuracy requirements.
pub struct Euler;

/// Visitor that applies a single Euler step to variables.
pub struct EulerStep {
    dt: f32,
}

impl Context<Euler> for EulerStep {
    fn time_step(&self) -> f32 {
        self.dt
    }
}

impl Visitor<Euler> for EulerStep {
    /// Apply the Euler update to a variable.
    ///
    /// Updates the variable's value using:
    /// `value_{n+1} = value_n + deriv * dt`
    ///
    /// Then resets the derivative to prepare for the next step.
    fn apply<P: Param>(&mut self, var: &mut Var<P, Euler>) {
        // Euler integration: x_{n+1} = x_n + dx/dt * dt
        var.value.step(&var.deriv, self.dt);

        // Reset derivative for next computation
        var.deriv = P::Deriv::default();
    }
}

impl Solver for Euler {
    type Context = EulerStep;
    /// Euler's method requires no additional storage per variable.
    type Storage<P: Param> = ();

    /// Perform one Euler integration step for the given system.
    ///
    /// The algorithm consists of:
    /// 1. Computing derivatives for all variables (`compute_derivs`)
    /// 2. Updating each variable using the Euler formula (`visit_vars`)
    ///
    /// # Arguments
    /// * `system` - The system to integrate.
    /// * `dt` - Time step for the integration.
    fn solve_step<S: System<Self>>(&self, system: &mut S, dt: f32) {
        let mut step = EulerStep { dt };

        // Compute derivatives at current state
        system.compute_derivs(&step);

        // Apply Euler update to all variables
        system.visit_vars(&mut step);
    }
}
