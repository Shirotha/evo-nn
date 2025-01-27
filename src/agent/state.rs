use std::{borrow::Borrow, mem::transmute};

use thin_vec::ThinVec;

use super::{NeuronID, NeuronOrder, body::*, brain::*, connection::*, neuron::*};
use crate::arena::*;

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Interface {
    Input(NeuronID),
    Output(NeuronID),
}

#[derive(Debug)]
pub struct State<'b, A, P, C>
where
    A: Activator,
    P: 'static + Propagator,
    C: Collector,
{
    brain: &'b Brain<A, P>,
    neuron_state: Buffer<A>,
    connection_state: Buffer<P>,
    interface_order: Buffer<Interface>,
    modulation_buffer: ThinVec<P::Input<'static>>,
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
    pub fn create_for<'a: 'b>(brain: &'b Brain<A, P>, body: &Body, arena: &'a mut Arena) -> Self {
        // SAFETY: slices allocated by arena will never overlap each other
        let neuron_state = arena.alloc_slice_with(brain.neurons().len(), A::default);
        let connection_state = arena.alloc_slice_with(brain.connections().len(), P::default);
        let sensors = body.sensor_neurons();
        let actions = body.action_neurons();
        let interface_order =
            arena.alloc_slice_from_iter(brain.order().iter_used().filter_map(|neuron| {
                sensors.binary_search(&neuron).ok().map(|_| Interface::Input(neuron)).or_else(
                    || actions.binary_search(&neuron).ok().map(|_| Interface::Output(neuron)),
                )
            }));
        Self {
            brain,
            neuron_state,
            connection_state,
            interface_order,
            collector: C::default(),
            modulation_buffer: ThinVec::new(),
        }
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
        buffer: &mut ThinVec<P::Input<'_>>,
        config: &Config<A, P, C>,
    ) {
        let input = neuron.output();
        // SAFETY: This is the only place where `buffer` is populated so it will always be empty here.
        // Changing the lifetime of the elements when they don't leak outside this function is always safe.
        let buffer =
            unsafe { transmute::<&mut ThinVec<P::Input<'_>>, &mut ThinVec<P::Input<'_>>>(buffer) };
        // TODO: does this run performant with len on the heap?
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
        let mut inputs = inputs.iter();
        let mut outputs = outputs.iter_mut();
        let mut interface = self.interface_order.iter().peekable();
        let mut connections =
            self.brain.connections().iter().zip(self.connection_state.iter_mut()).peekable();
        for (index, neuron) in self.brain.neurons().iter().enumerate() {
            if interface.next_if(|i| **i == Interface::Input(neuron.id)).is_some() {
                self.collector.push(inputs.next().expect("").into(), &config.collector);
            }
            while let Some(edge) = connections.next_if(|(conn, _)| conn.to == neuron.id) {
                let state = Self::get(&self.neuron_state, self.brain.order(), edge.0.from);
                Self::push(
                    state,
                    edge,
                    &mut self.collector,
                    &self.neuron_state,
                    self.brain.order(),
                    &mut self.modulation_buffer,
                    config,
                );
            }
            // SAFETY: Since `neuron_state` and `brain.neurons()` are always the same length
            // indexing into `neuron_state` with an index received from enumerating `brain.neurons()` is always safe.
            let state = unsafe { self.neuron_state.get_unchecked_mut(index) };
            Self::activate(&mut self.collector, (neuron, state), config);
            if interface.next_if(|o| **o == Interface::Output(neuron.id)).is_some() {
                *outputs.next().expect("output buffer is not big enough") = state.output().into();
            }
        }
    }
}
