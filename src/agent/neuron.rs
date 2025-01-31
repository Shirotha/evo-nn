use std::fmt::{Debug, Display};

use super::NeuronID;

/// Per [`Neuron`] state during simulation.
pub trait Activator: Debug + Default {
    /// Will be received from [`Collector::Output`].
    type Input<'i>
    where
        Self: 'i;
    /// Will be send to [`Propagator::Input`].
    type Output<'o>
    where
        Self: 'o;
    /// Additional data stored in the [`Neuron`] used during activation.
    type Gene: Debug + Clone;
    /// Additional global data used during activation.
    type Config: Debug;
    /// Calculate the [`Neuron`] output using the weighted input of all connections.
    fn activate(&mut self, input: Self::Input<'_>, gene: &Self::Gene, config: &Self::Config);
    /// Retuns the current value of the [`Neuron`] output.
    /// The value should stay the same between calls to [`activate`].
    fn output(&self) -> Self::Output<'_>;
}

/// Neuron data used both as static data during simulation and as a direct gene.
#[derive(Debug, Clone)]
pub struct Neuron<A: Activator> {
    pub id: NeuronID,
    pub activator_gene: A::Gene,
}
impl<A> Display for Neuron<A>
where
    A: Activator<Gene: Display>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.id, self.activator_gene)
    }
}
