use std::fmt::Debug;

use thin_vec::ThinVec;

use super::{NeuronOrder, connection::*, neuron::*};

#[derive(Clone, Debug)]
pub struct Brain<A, P>
where
    A: Activator,
    P: Propagator,
{
    neurons:     ThinVec<Neuron<A>>,
    connections: ThinVec<Connection<P>>,
    order:       NeuronOrder,
}
impl<A, P> Default for Brain<A, P>
where
    A: Activator,
    P: Propagator,
{
    fn default() -> Self {
        Self {
            neurons:     ThinVec::new(),
            connections: ThinVec::new(),
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
        neurons: ThinVec<Neuron<A>>,
        connections: ThinVec<Connection<P>>,
        order: NeuronOrder,
    ) -> Self {
        Self { neurons, connections, order }
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

    pub fn order(&self) -> &NeuronOrder {
        &self.order
    }

    /// Gain direct access to the internal network data.
    /// This will always trigger a reordering of all data.
    pub fn raw(&mut self) -> RawBrainAccess<'_, A, P> {
        RawBrainAccess {
            neurons:     &mut self.neurons,
            connections: &mut self.connections,
            order:       &mut self.order,
        }
    }
    // TODO: provide stable operations on `Brain`
}

#[derive(Debug)]
pub struct RawBrainAccess<'b, A, P>
where
    A: Activator,
    P: Propagator,
{
    pub neurons:     &'b mut ThinVec<Neuron<A>>,
    pub connections: &'b mut ThinVec<Connection<P>>,
    pub order:       &'b mut NeuronOrder,
}
impl<A, P> Drop for RawBrainAccess<'_, A, P>
where
    A: Activator,
    P: Propagator,
{
    fn drop(&mut self) {
        todo!("fix invariants before returning data")
        // - order neurons in topological order
        // - create NeuronOrder using remap with neurons
        // - apply remap to all NeuronIDs
        // - order connections by NeuronID
    }
}
