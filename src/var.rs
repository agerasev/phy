use crate::{Param, Solver};
use core::{
    fmt::{self, Debug, Formatter},
    ops::{Deref, DerefMut},
};

/// Independent variable.
pub struct Var<P: Param, S: Solver> {
    pub value: P,
    pub deriv: P::Deriv,
    pub storage: S::Storage<P>,
}

impl<P: Param, S: Solver> Clone for Var<P, S> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<P: Param, S: Solver> Copy for Var<P, S> {}
impl<P: Param, S: Solver> Default for Var<P, S> {
    fn default() -> Self {
        Self {
            value: P::default(),
            deriv: P::Deriv::default(),
            storage: S::Storage::<P>::default(),
        }
    }
}

impl<P: Param, S: Solver> Var<P, S> {
    pub fn new(value: P) -> Self {
        Var {
            value,
            deriv: Default::default(),
            storage: Default::default(),
        }
    }
}

impl<P: Param, S: Solver> Deref for Var<P, S> {
    type Target = P;
    fn deref(&self) -> &Self::Target {
        &self.value
    }
}
impl<P: Param, S: Solver> DerefMut for Var<P, S> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.value
    }
}

impl<P: Param, S: Solver> Debug for Var<P, S>
where
    P: Debug,
    P::Deriv: Debug,
    S::Storage<P>: Debug,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Var {{ value: {:?}, deriv: {:?}, storage: {:?} }}",
            &self.value, &self.deriv, &self.storage
        )
    }
}
