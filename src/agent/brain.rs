use std::{collections::VecDeque, fmt::Debug};

use bit_set::BitSet;
use thin_vec::ThinVec;

use super::*;

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
            inputs:      ThinVec::new(),
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
    pub inputs:      ThinVec<NeuronID>,
}
impl<A, P> Drop for RawBrainAccess<'_, A, P>
where
    A: Activator,
    P: Propagator,
{
    fn drop(&mut self) {
        let mut order = Vec::with_capacity(self.neurons.len());
        let mut open = VecDeque::from_iter(
            self.inputs
                .iter()
                .copied()
                .map(|id| self.order.index(id).expect("all inputs should be in the ordering")),
        );
        let mut seen = BitSet::with_capacity(self.neurons.len());
        seen.extend(open.iter().copied());
        while let Some(current) = open.pop_front() {
            let id = self.neurons[current].id;
            order.push(id);
            for next in self
                .connections
                .iter()
                .enumerate()
                .filter_map(|(i, conn)| (conn.from == id).then_some(i))
            {
                if seen.insert(next) {
                    open.push_back(next);
                }
            }
        }
        let map = self.order.rebuild(order);
        self.neurons.iter_mut().for_each(|neuron| neuron.id = map[&neuron.id]);
        self.connections.iter_mut().for_each(|conn| {
            conn.from = map[&conn.from];
            conn.to = map[&conn.to];
            P::remap_gene(&mut conn.propagator_gene, &map);
        });
        self.neurons.sort_unstable_by_key(|neuron| neuron.id);
        self.connections.sort_unstable_by_key(|conn| conn.from);
    }
}
