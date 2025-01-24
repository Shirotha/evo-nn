use std::{borrow::Borrow, mem::transmute};

use super::{NeuronID, NeuronOrder, brain::*, connection::*, neuron::*};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Config<A, P, C>
where
    A: Activator,
    P: Propagator,
    C: Collector,
{
    pub activator:  A::Config,
    pub propagator: P::Config,
    pub collector:  C::Config,
}

#[derive(Debug, Clone)]
pub struct State<'b, A, P, C>
where
    A: Activator,
    P: Propagator,
    C: Collector,
{
    brain: &'b Brain<A, P>,
    neuron_state: Vec<A>,
    connection_state: Vec<P>,
    collector: C,
}
impl<'b, A, P, C> State<'b, A, P, C>
where
    //  P -> C -> A -> P
    //      /      \
    //  in /        \ out
    // NOTE: `'static` bound is required by generic associated types at the moment
    A: 'static + for<'a> Activator<Input<'a> = C::Output<'a>, Output<'a> = P::Input<'a>>,
    P: 'static + for<'p> Propagator<Output<'p> = C::Input<'p>>,
    C: Collector,
{
    pub fn new(brain: &'b Brain<A, P>) -> Self {
        let mut neuron_state = Vec::new();
        neuron_state.resize_with(brain.neurons().len(), A::default);
        let mut connection_state = Vec::new();
        connection_state.resize_with(brain.connections().len(), P::default);
        Self { brain, neuron_state, connection_state, collector: C::default() }
    }

    pub fn new_with(
        brain: &'b Brain<A, P>,
        mut neuron_state: Vec<A>,
        mut connection_state: Vec<P>,
        collector: C,
    ) -> Self {
        let neurons = brain.neurons().len();
        neuron_state.truncate(neurons);
        neuron_state.fill_with(A::default);
        neuron_state.resize_with(neurons, A::default);
        let connections = brain.connections().len();
        connection_state.truncate(connections);
        connection_state.fill_with(P::default);
        connection_state.resize_with(connections, P::default);
        Self { brain, neuron_state, connection_state, collector }
    }

    fn get<'a>(neurons: &'a [A], order: &NeuronOrder, id: NeuronID) -> &'a A {
        &neurons[order.index(id).expect("all neurons should be included in the order")]
    }

    #[inline(always)]
    fn push(
        neuron: &A,
        edge: (&Connection<P>, &mut P),
        collector: &mut C,
        neurons: &[A],
        order: &NeuronOrder,
        buffer: &mut Vec<P::Input<'_>>,
        config: &Config<A, P, C>,
    ) {
        let input = neuron.output();
        // SAFETY: This is the only place where `buffer` is populated so it will always be empty here.
        // Changing the lifetime of the elements when they don't leak outside this function is always safe.
        let buffer = unsafe {
            transmute::<&mut std::vec::Vec<P::Input<'_>>, &mut Vec<P::Input<'_>>>(buffer)
        };
        buffer
            .extend(edge.1.modulation().map(|id| Self::get(neurons, order, *id.borrow()).output()));
        let edge = edge.1.propagate(input, buffer, &edge.0.propagator_gene, &config.propagator);
        collector.push(edge, &config.collector);
        // NOTE: this is required by the assumption above
        buffer.clear();
    }

    #[inline(always)]
    fn activate(collector: &mut C, neuron: (&Neuron<A>, &mut A), config: &Config<A, P, C>) {
        let input = collector.collect(&config.collector);
        neuron.1.activate(input, &neuron.0.activator_gene, &config.activator);
        collector.clear(&config.collector);
    }

    // TODO: maybe store input/output buffer in here?
    pub fn step<I, O>(&mut self, inputs: &[I], outputs: &mut [O], config: &Config<A, P, C>)
    where
        for<'c> &'c I: Into<C::Input<'c>>,
        O: for<'a> From<A::Output<'a>>,
    {
        assert!(inputs.len() == self.brain.inputs().len(), "wrong number of inputs");
        assert!(outputs.len() == self.brain.outputs().len(), "wrong number of outputs");
        let mut modulation_buffer = Vec::new();
        let mut inputs = self.brain.inputs().iter().zip(inputs).peekable();
        let mut outputs = self.brain.outputs().iter().zip(outputs).peekable();
        let mut connections =
            self.brain.connections().iter().zip(&mut self.connection_state).peekable();
        for (index, neuron) in self.brain.neurons().iter().enumerate() {
            if let Some((_, input)) = inputs.next_if(|(i, _)| **i == neuron.id) {
                self.collector.push(input.into(), &config.collector);
            }
            while let Some(edge) = connections.next_if(|(conn, _)| conn.to == neuron.id) {
                let state = Self::get(&self.neuron_state, self.brain.order(), edge.0.from);
                Self::push(
                    state,
                    edge,
                    &mut self.collector,
                    &self.neuron_state,
                    self.brain.order(),
                    &mut modulation_buffer,
                    config,
                );
            }
            // SAFETY: Since `neuron_state` and `brain.neurons()` are always the same length
            // indexing into `neuron_state` with an index received from enumerating `brain.neurons()` is always safe.
            let state = unsafe { self.neuron_state.get_unchecked_mut(index) };
            Self::activate(&mut self.collector, (neuron, state), config);
            if let Some((_, output)) = outputs.next_if(|(o, _)| **o == neuron.id) {
                *output = state.output().into();
            }
        }
    }
}
