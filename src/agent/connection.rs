use std::{
    borrow::Borrow,
    fmt::{Debug, Display},
};

use super::NeuronID;

pub trait Propagator: Debug + Default {
    /// Will be received from [`Activator::Output`].
    type Input<'i>
    where
        Self: 'i;
    /// Will be send to [`Collector::Input`].
    type Output<'o>
    where
        Self: 'o;
    /// Additional data stored in the [`Connection`] used during propagation.
    type Gene: Debug + Clone;
    /// Additional global data used during propagation.
    type Config: Debug;
    /// Returns iterator over [`NeuronID`] of modulation inputs.
    fn modulation(&self) -> impl Iterator<Item: Borrow<NeuronID>>;
    /// Calculates the value send to the connection target.
    fn propagate(
        &mut self,
        input: Self::Input<'_>,
        modulation: &[Self::Input<'_>],
        gene: &Self::Gene,
        config: &Self::Config,
    ) -> Self::Output<'_>;
}

#[derive(Debug, Clone)]
pub struct Connection<P: Propagator> {
    pub from: NeuronID,
    pub to: NeuronID,
    pub propagator_gene: P::Gene,
}
impl<P> Display for Connection<P>
where
    P: Propagator<Gene: Display>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} -> {}: {}", self.from, self.to, self.propagator_gene)
    }
}

pub trait Collector: Default {
    /// Will be received from [`Propagator::Output`].
    type Input<'i>
    where
        Self: 'i;
    /// Will be send to [`Activator::Input`]
    type Output<'o>
    where
        Self: 'o;
    type Config;
    /// Adds new value into the [`Collector`].
    fn push(&mut self, input: Self::Input<'_>, config: &Self::Config);
    /// Returns the collected output.
    fn collect(&mut self, config: &Self::Config) -> Self::Output<'_>;
    /// Resets the state to be ready to collect a new value.
    fn clear(&mut self, config: &Self::Config);
}
