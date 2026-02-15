use crate::{Context, Param, Solver, System, Var, Visitor};

/// The classical fourth-order Runge–Kutta method (RK4).
///
/// This solver implements the standard RK4 algorithm for solving
/// first-order ordinary differential equations of the form dy/dt = f(t, y).
///
/// # Algorithm
///
/// For a differential equation dy/dt = f(t, y), RK4 computes:
/// ```text
/// k1 = f(t_n, y_n)
/// k2 = f(t_n + h/2, y_n + h*k1/2)
/// k3 = f(t_n + h/2, y_n + h*k2/2)
/// k4 = f(t_n + h, y_n + h*k3)
/// y_{n+1} = y_n + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
/// ```
///
/// where `h` is the time step `dt`.
///
/// # Storage Requirements
///
/// RK4 requires additional storage per variable to hold intermediate
/// computations between stages. This is automatically managed by the
/// [`Rk4Storage`] type.
pub struct Rk4;

/// Storage required by the RK4 solver for each variable.
///
/// This stores the initial value `y_n` and accumulates weighted derivatives
/// incrementally across RK4 stages to compute the final update.
#[derive(Clone, Copy, Default, Debug)]
pub struct Rk4Storage<P: Param> {
    /// The initial value at the beginning of the RK4 step (y_n)
    init_value: P,
    /// Accumulated weighted derivatives: k1 + 2*k2 + 2*k3 + k4
    weighted_accum: P::Deriv,
}

/// The four stages of the RK4 algorithm.
#[derive(Clone, Copy)]
enum Rk4Stage {
    /// Stage 1: Compute k1 = f(t_n, y_n)
    Stage1,
    /// Stage 2: Compute k2 = f(t_n + h/2, y_n + h*k1/2)
    Stage2,
    /// Stage 3: Compute k3 = f(t_n + h/2, y_n + h*k2/2)
    Stage3,
    /// Stage 4: Compute k4 = f(t_n + h, y_n + h*k3) and final update
    Stage4,
}

/// Visitor that applies a single RK4 stage to variables.
pub struct Rk4Step {
    stage: Rk4Stage,
    dt: f32,
}

impl Context<Rk4> for Rk4Step {
    /// Returns the time step used in this stage.
    ///
    /// Used as the `dt` parameter passed to `System::compute_derivs`
    /// **before** the stage computations.
    fn time_step(&self) -> f32 {
        match self.stage {
            Rk4Stage::Stage1 => self.dt / 2.0,
            Rk4Stage::Stage2 => self.dt / 2.0,
            Rk4Stage::Stage3 => self.dt,
            Rk4Stage::Stage4 => self.dt,
        }
    }
}

impl Visitor<Rk4> for Rk4Step {
    fn apply<P: Param>(&mut self, var: &mut Var<P, Rk4>) {
        let y = &mut var.value;
        let dy_dt = &mut var.deriv;
        let init_y = &mut var.storage.init_value;
        let accum = &mut var.storage.weighted_accum;
        let dt = self.dt;

        match self.stage {
            Rk4Stage::Stage1 => {
                // k1 = f(t_n, y_n)
                // Save initial value y_n for use in subsequent stages
                init_y.clone_from(y);
                // Start accumulation with k1
                accum.clone_from(dy_dt);
                // Prepare state for stage 2: y = y_n + k1 * dt / 2
                y.step(dy_dt, 0.5 * dt);
            }
            Rk4Stage::Stage2 => {
                // k2 = f(t_n + dt/2, y_n + k1*dt/2)
                // Prepare state for stage 3: y = y_n + k2 * dt / 2
                y.clone_from(init_y);
                y.step(dy_dt, 0.5 * dt);
                // Accumulate: k1 + 2*k2
                *dy_dt *= 2.0;
                *accum += dy_dt;
            }
            Rk4Stage::Stage3 => {
                // k3 = f(t_n + dt/2, y_n + k2*dt/2)
                // Prepare state for stage 4: y = y_n + k3 * dt
                *y = init_y.clone();
                y.step(dy_dt, dt);
                // Accumulate: k1 + 2*k2 + 2*k3
                *dy_dt *= 2.0;
                *accum += dy_dt;
            }
            Rk4Stage::Stage4 => {
                // k4 = f(t_n + dt, y_n + k3*dt)
                // Complete accumulation: (k1 + 2*k2 + 2*k3 + k4) / 6
                *accum += dy_dt;
                *accum *= 1.0 / 6.0;
                // Final update: y_{n+1} = y_n + accum * dt
                *y = init_y.clone();
                y.step(accum, dt);
            }
        }

        // Reset derivative for next stage
        *dy_dt = P::Deriv::default();
    }
}

impl Solver for Rk4 {
    type Context = Rk4Step;
    type Storage<P: Param> = Rk4Storage<P>;

    fn solve_step<S: System<Self>>(&self, system: &mut S, dt: f32) {
        // Execute the four RK4 stages in sequence
        for stage in [
            Rk4Stage::Stage1,
            Rk4Stage::Stage2,
            Rk4Stage::Stage3,
            Rk4Stage::Stage4,
        ] {
            let mut step = Rk4Step { stage, dt };

            // Compute derivatives with the appropriate time step for this stage
            system.compute_derivs(&step);

            // Apply the RK4 stage to all variables
            system.visit_vars(&mut step);
        }
    }
}
