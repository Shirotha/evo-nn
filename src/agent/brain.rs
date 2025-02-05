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
            for next in
                self.connections.iter().filter_map(|conn| (conn.from == id).then_some(conn.to))
            {
                let index =
                    self.order.index(next).expect("all connections should be in the ordering");
                if seen.insert(index) {
                    open.push_back(index);
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
        self.connections.sort_by_key(|conn| conn.from);
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[derive(Debug, Default)]
    struct DummyData;
    impl Activator for DummyData {
        type Config = ();
        type Gene = ();
        type Input<'i>
            = f64
        where
            Self: 'i;
        type Output<'o>
            = f64
        where
            Self: 'o;

        fn activate(
            &mut self,
            _input: Self::Input<'_>,
            _gene: &Self::Gene,
            _config: &Self::Config,
        ) {
        }

        fn output(&self) -> Self::Output<'_> {
            0.0
        }
    }
    impl Propagator for DummyData {
        type Config = ();
        type Gene = ();
        type Input<'i>
            = f64
        where
            Self: 'i;
        type Output<'o>
            = f64
        where
            Self: 'o;

        fn modulation(
            &self,
            _gene: &Self::Gene,
            _config: &Self::Config,
        ) -> impl Iterator<Item: std::borrow::Borrow<NeuronID>> {
            std::iter::empty::<NeuronID>()
        }

        fn propagate(
            &mut self,
            input: Self::Input<'_>,
            _modulation: &[Self::Input<'_>],
            _gene: &Self::Gene,
            _config: &Self::Config,
        ) -> Self::Output<'_> {
            input
        }
    }
    #[test]
    fn raw_brain_access_leaves_brain_ordered() {
        let mut brain = Brain::<DummyData, DummyData>::new();
        {
            let mut access = brain.raw();
            let id0 = access.order.next_free(None).unwrap();
            let id1 = access.order.next_free(Some(id0)).unwrap();
            let id2 = access.order.next_free(Some(id1)).unwrap();
            unsafe {
                access.order.set_unchecked(id0, Some(0));
                access.order.set_unchecked(id1, Some(1));
                access.order.set_unchecked(id2, Some(2));
            }
            access.neurons.push(Neuron { id: id0, activator_gene: () });
            access.neurons.push(Neuron { id: id1, activator_gene: () });
            access.neurons.push(Neuron { id: id2, activator_gene: () });
            access.connections.push(Connection { from: id1, to: id0, propagator_gene: () });
            access.connections.push(Connection { from: id0, to: id1, propagator_gene: () });
            access.connections.push(Connection { from: id0, to: id2, propagator_gene: () });
            access.inputs.push(id1);
        }
        let ids = brain.order().iter_used().collect::<Box<_>>();
        let conns = brain.connections();
        assert_eq!(ids[0], conns[0].from);
        assert_eq!(ids[1], conns[0].to);
        assert_eq!(ids[1], conns[1].from);
        assert_eq!(ids[0], conns[1].to);
        assert_eq!(ids[1], conns[2].from);
        assert_eq!(ids[2], conns[2].to);
    }
}
