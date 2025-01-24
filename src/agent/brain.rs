use std::fmt::Debug;

use super::{NeuronID, NeuronOrder, connection::*, neuron::*};

#[derive(Clone, Debug)]
pub struct Brain<A, P>
where
    A: Activator,
    P: Propagator,
{
    neurons:     Vec<Neuron<A>>,
    connections: Vec<Connection<P>>,
    inputs:      Vec<NeuronID>,
    outputs:     Vec<NeuronID>,
    order:       NeuronOrder,
}
impl<A, P> Default for Brain<A, P>
where
    A: Activator,
    P: Propagator,
{
    fn default() -> Self {
        Self {
            neurons:     Vec::new(),
            connections: Vec::new(),
            inputs:      Vec::new(),
            outputs:     Vec::new(),
            order:       NeuronOrder::new(),
        }
    }
}
impl<A, P> Brain<A, P>
where
    A: Activator,
    P: Propagator,
{
    /// Creats an instance of [`Brain`] from existing data.
    /// # Safety
    /// Assumes that neurons are sorted in topological order.
    /// Assumes that connections are sorted by `Connection.to` using `Neuron.id` as a sort index.
    /// Asummes that inputs and outputs are sorted using `Neuron.id` as a sort index.
    pub unsafe fn new_unchecked(
        neurons: Vec<Neuron<A>>,
        connections: Vec<Connection<P>>,
        inputs: Vec<NeuronID>,
        outputs: Vec<NeuronID>,
        order: NeuronOrder,
    ) -> Self {
        Self { neurons, connections, inputs, outputs, order }
    }

    /// Creates an empty instance of [`Brain`]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn neurons(&self) -> &[Neuron<A>] {
        &self.neurons
    }

    pub fn connections(&self) -> &[Connection<P>] {
        &self.connections
    }

    pub fn inputs(&self) -> &[NeuronID] {
        &self.inputs
    }

    pub fn outputs(&self) -> &[NeuronID] {
        &self.outputs
    }

    pub fn order(&self) -> &NeuronOrder {
        &self.order
    }

    /// Gain direct access to the internal network data.
    /// This will always trigger a reordering of all data.
    pub fn raw(&mut self) -> RawBrainAccess<'_, A, P> {
        RawBrainAccess {
            neurons:     &mut self.neurons,
            connections: &mut self.connections,
            inputs:      &mut self.inputs,
            outputs:     &mut self.outputs,
            order:       &mut self.order,
        }
    }

    /// Create a wrapper around the ordered operations
    /// that caches the used and free [`NeuronID`]s.
    /// This will be faster when multiple operations are performed at once.
    pub fn pooled(&mut self) -> PooledBrainAccess<'_, A, P> {
        PooledBrainAccess::new(self)
    }
    // TODO: provide stable operations on `Brain`
}

mod pooled_brain_access {
    use super::*;
    use crate::agent::NeuronPool;

    pub struct PooledBrainAccess<'b, A, P>
    where
        A: Activator,
        P: Propagator,
    {
        brain: &'b mut Brain<A, P>,
        pool:  NeuronPool,
    }
    impl<'b, A, P> PooledBrainAccess<'b, A, P>
    where
        A: Activator,
        P: Propagator,
    {
        pub fn new(brain: &'b mut Brain<A, P>) -> Self {
            brain.order.truncate();
            // SAFETY: `largest_used` will always be correct after calling `truncate`.
            let pool = unsafe {
                NeuronPool::new_unchecked(brain.order.largest_used(), brain.order.iter_free())
            };
            Self { brain, pool }
        }
        // TODO: provide specialized implementations to stable operations on `brain` using `pool`
    }
}
pub use pooled_brain_access::*;

pub struct RawBrainAccess<'b, A, P>
where
    A: Activator,
    P: Propagator,
{
    pub neurons:     &'b mut Vec<Neuron<A>>,
    pub connections: &'b mut Vec<Connection<P>>,
    pub inputs:      &'b mut Vec<NeuronID>,
    pub outputs:     &'b mut Vec<NeuronID>,
    pub order:       &'b mut NeuronOrder,
}
impl<A, P> Debug for RawBrainAccess<'_, A, P>
where
    A: Activator<Gene: Debug> + Debug,
    P: Propagator<Gene: Debug> + Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DirectBrainAccess")
            .field("neurons", &self.neurons)
            .field("connections", &self.connections)
            .field("inputs", &self.inputs)
            .field("outputs", &self.outputs)
            .finish()
    }
}
impl<A, P> Drop for RawBrainAccess<'_, A, P>
where
    A: Activator,
    P: Propagator,
{
    fn drop(&mut self) {
        todo!("fix invariants before returning data")
        // --order neurons in topological order
        // --create NeuronOrder using remap with neurons
        // --apply remap to all NeuronIDs
        // --order connections, inputs, outputs by NeuronID
    }
}
