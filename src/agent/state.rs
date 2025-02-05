use std::{borrow::Borrow, fmt::Debug, mem::transmute};

use thin_vec::ThinVec;

use super::*;
use crate::arena::*;

#[derive(Debug, Default)]
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

impl<A, P, C> Clone for Config<A, P, C>
where
    A: Activator<Config: Clone>,
    P: Propagator<Config: Clone>,
    C: Collector<Config: Clone>,
{
    fn clone(&self) -> Self {
        Self {
            activator:  self.activator.clone(),
            propagator: self.propagator.clone(),
            collector:  self.collector.clone(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Interface {
    Input(NeuronID),
    Output(NeuronID),
}
impl Interface {
    pub fn into_id(self) -> NeuronID {
        match self {
            Self::Input(id) => id,
            Self::Output(id) => id,
        }
    }
}

pub struct State<A, P, C>
where
    A: Activator,
    P: 'static + Propagator,
    C: Collector,
{
    neuron_state:      Buffer<A>,
    connection_state:  Buffer<P>,
    interface_order:   Buffer<Interface>,
    modulation_buffer: ThinVec<P::Input<'static>>,
    collector:         C,
}

impl<A, P, C> Debug for State<A, P, C>
where
    A: Activator,
    P: 'static + Propagator,
    C: Collector,
{
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.debug_struct("State")
            .field("neuron_state", &self.neuron_state)
            .field("connection_state", &self.connection_state)
            .field("interface_order", &self.interface_order)
            .field("collector", &self.collector)
            .finish_non_exhaustive()
    }
}
impl<A, P, C> State<A, P, C>
where
    //  P -> C -> A -> P
    //      /      \
    //  in /        \ out
    // NOTE: `'static` bound is required by generic associated types at the moment
    A: 'static + for<'a> Activator<Input<'a> = C::Output<'a>, Output<'a> = P::Input<'a>>,
    P: 'static + for<'p> Propagator<Output<'p> = C::Input<'p>>,
    C: Collector,
{
    pub fn create_for<X: Phenotype>(
        brain: &Brain<A, P>,
        body: &Body<X>,
        arena: &mut Arena,
    ) -> Self {
        let neuron_state = arena.alloc_slice_with(brain.neurons().len(), A::default);
        let connection_state = arena.alloc_slice_with(brain.connections().len(), P::default);
        // TODO: since both sensors and actions are sorted in itself, this can be replaced by an interleave_by
        let mut interface_order = arena.alloc_slice_from_iter(
            body.iter_sensor_neurons()
                .map(Interface::Input)
                .chain(body.iter_action_neurons().map(Interface::Output)),
        );
        interface_order.sort_by(|a, b| {
            brain
                .order()
                .cmp(a.into_id(), b.into_id())
                .expect("all interface neurons should be included in the order")
        });
        Self {
            neuron_state,
            connection_state,
            interface_order,
            modulation_buffer: ThinVec::new(),
            collector: C::default(),
        }
    }

    pub fn move_buffers(&mut self, arena: &mut Arena) {
        // SAFETY: original buffers get overwritten so old pointers are inaccessible
        unsafe {
            self.neuron_state = arena.move_into(&self.neuron_state);
            self.connection_state = arena.move_into(&self.connection_state);
            self.interface_order = arena.move_into(&self.interface_order);
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
        buffer.extend(
            edge.1
                .modulation(&edge.0.propagator_gene, &config.propagator)
                .map(|id| Self::get(neurons, order, *id.borrow()).output()),
        );
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

    pub fn step<I, O>(
        &mut self,
        brain: &Brain<A, P>,
        inputs: &[I],
        outputs: &mut [O],
        config: &Config<A, P, C>,
    ) where
        for<'c> &'c I: Into<C::Input<'c>>,
        O: for<'a> From<A::Output<'a>>,
    {
        let mut inputs = inputs.iter();
        let mut outputs = outputs.iter_mut();
        let mut interface = self.interface_order.iter().peekable();
        let mut connections =
            brain.connections().iter().zip(self.connection_state.iter_mut()).peekable();
        for (index, neuron) in brain.neurons().iter().enumerate() {
            if interface.next_if(|i| **i == Interface::Input(neuron.id)).is_some() {
                self.collector.push(
                    inputs.next().expect("input buffer is not big enough").into(),
                    &config.collector,
                );
            }
            while let Some(edge) = connections.next_if(|(conn, _)| conn.to == neuron.id) {
                let state = Self::get(&self.neuron_state, brain.order(), edge.0.from);
                Self::push(
                    state,
                    edge,
                    &mut self.collector,
                    &self.neuron_state,
                    brain.order(),
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

#[cfg(test)]
mod test {
    use super::*;

    #[derive(Debug)]
    struct Signal {
        kind:  SignalKind,
        value: f64,
    }
    #[derive(Debug, Default)]
    struct Cumulant {
        pub data:    f64,
        pub control: f64,
    }
    #[derive(Debug, Clone)]
    struct NeuronGene {
        pub speed: f64,
    }
    #[derive(Debug, Default)]
    struct NeuronConfig {
        pub activation_threshold: f64,
    }
    #[derive(Debug, Default)]
    struct TestActivator {
        pub state: f64,
    }
    impl Activator for TestActivator {
        type Config = NeuronConfig;
        type Gene = NeuronGene;
        type Input<'i>
            = &'i Cumulant
        where
            Self: 'i;
        type Output<'o>
            = f64
        where
            Self: 'o;

        fn activate(&mut self, input: Self::Input<'_>, gene: &Self::Gene, config: &Self::Config) {
            if input.control >= config.activation_threshold {
                self.state += (input.data - self.state) * gene.speed;
            } else {
                self.state = 0.0;
            }
        }

        fn output(&self) -> Self::Output<'_> {
            self.state
        }
    }
    #[derive(Debug, Clone, Copy)]
    enum SignalKind {
        Data,
        Control,
    }
    #[derive(Debug, Clone)]
    enum Weight {
        Direct(f64),
        Modulated(NeuronID),
    }
    #[derive(Debug, Clone)]
    struct ConnectionGene {
        kind:   SignalKind,
        weight: Weight,
    }
    #[derive(Debug, Default)]
    struct ConnectionConfig;
    #[derive(Debug, Default)]
    struct TestPropagator;
    impl Propagator for TestPropagator {
        type Config = ConnectionConfig;
        type Gene = ConnectionGene;
        type Input<'i>
            = f64
        where
            Self: 'i;
        type Output<'o>
            = Signal
        where
            Self: 'o;

        fn modulation(
            &self,
            gene: &Self::Gene,
            _config: &Self::Config,
        ) -> impl Iterator<Item: Borrow<NeuronID>> {
            match gene.weight {
                Weight::Direct(_) => None,
                Weight::Modulated(id) => Some(id),
            }
            .into_iter()
        }

        fn propagate(
            &mut self,
            input: Self::Input<'_>,
            modulation: &[Self::Input<'_>],
            gene: &Self::Gene,
            _config: &Self::Config,
        ) -> Self::Output<'_> {
            let value = match gene.weight {
                Weight::Direct(w) => input * w,
                Weight::Modulated(_) => input * modulation[0],
            };
            Signal { kind: gene.kind, value }
        }
    }
    #[derive(Debug, Default)]
    struct TestCollector {
        state: Cumulant,
    }
    impl Collector for TestCollector {
        type Config = ();
        type Input<'i>
            = Signal
        where
            Self: 'i;
        type Output<'o>
            = &'o Cumulant
        where
            Self: 'o;

        fn push(&mut self, input: Self::Input<'_>, _config: &Self::Config) {
            match input.kind {
                SignalKind::Data => self.state.data += input.value,
                SignalKind::Control => self.state.control += input.value,
            }
        }

        fn collect(&mut self, _config: &Self::Config) -> Self::Output<'_> {
            &self.state
        }

        fn clear(&mut self, _config: &Self::Config) {
            self.state.data = 0.0;
            self.state.control = 0.0;
        }
    }
    struct TestPhenotype;
    impl Phenotype for TestPhenotype {
        type ActionGene = ();
        type SensorGene = ();
    }
    type TestBrain = Brain<TestActivator, TestPropagator>;
    type TestBody = Body<TestPhenotype>;
    type TestConfig = Config<TestActivator, TestPropagator, TestCollector>;

    fn run(brain: &TestBrain, body: &TestBody, config: &TestConfig, inputs: &[f64]) -> Vec<f64> {
        let mut arena = Arena::new();
        // FIXME: trait bounds not satisfied
        // apparently Signal != Signal and &'i Cumulant != &'o Cumulant?
        let mut state = State::create_for(brain, body, &mut arena);
        let mut outputs = vec![0.0; body.action_count()];
        state.step(brain, inputs, &mut outputs, config);
        outputs
    }

    fn free_ids(order: &NeuronOrder, count: usize) -> Vec<NeuronID> {
        let mut result = Vec::new();
        if count == 0 {
            return result;
        }
        let Some(mut current) = order.next_free(None) else { return result };
        result.push(current);
        if count == 1 {
            return result;
        }
        while let Some(next) = order.next_free(Some(current)) {
            result.push(next);
            if result.len() == count {
                return result;
            }
            current = next;
        }
        result
    }

    #[test]
    fn xor() {
        // inputs: <0>, <1>
        // <2> = <0> mod <1>
        // <3> = <0> + <1> if -1.0 * <2>
        // outputs: <3>
        let mut brain = TestBrain::new();
        // TODO: construct body
        let mut body = TestBody::new();
        {
            let mut access = brain.raw();
            let ids = free_ids(&access.order, 4);
            access.neurons.extend(
                ids.iter()
                    .copied()
                    .map(|id| Neuron { id, activator_gene: NeuronGene { speed: 1.0 } }),
            );
            access.connections.push(Connection {
                from: ids[0],
                to: ids[2],
                propagator_gene: ConnectionGene {
                    kind:   SignalKind::Data,
                    weight: Weight::Modulated(ids[1]),
                },
            });
            access.connections.push(Connection {
                from: ids[0],
                to: ids[3],
                propagator_gene: ConnectionGene {
                    kind:   SignalKind::Data,
                    weight: Weight::Direct(1.0),
                },
            });
            access.connections.push(Connection {
                from: ids[1],
                to: ids[3],
                propagator_gene: ConnectionGene {
                    kind:   SignalKind::Data,
                    weight: Weight::Direct(1.0),
                },
            });
            access.connections.push(Connection {
                from: ids[2],
                to: ids[3],
                propagator_gene: ConnectionGene {
                    kind:   SignalKind::Control,
                    weight: Weight::Direct(-1.0),
                },
            });
            access.inputs.extend(ids.into_iter().take(2));
        }
        let config = TestConfig::default();
        assert_eq!(run(&brain, &body, &config, &[0.0, 0.0]), vec![0.0]);
        assert_eq!(run(&brain, &body, &config, &[1.0, 0.0]), vec![1.0]);
        assert_eq!(run(&brain, &body, &config, &[0.0, 1.0]), vec![1.0]);
        assert_eq!(run(&brain, &body, &config, &[1.0, 1.0]), vec![0.0]);
    }
}
