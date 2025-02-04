use std::{
    borrow::Borrow,
    collections::HashMap,
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
    type Config: Debug + Default;
    /// Returns iterator over [`NeuronID`] of modulation inputs.
    fn modulation(
        &self,
        gene: &Self::Gene,
        config: &Self::Config,
    ) -> impl Iterator<Item: Borrow<NeuronID>>;
    /// Calculates the value send to the connection target.
    fn propagate(
        &mut self,
        input: Self::Input<'_>,
        modulation: &[Self::Input<'_>],
        gene: &Self::Gene,
        config: &Self::Config,
    ) -> Self::Output<'_>;

    #[expect(unused_variables)]
    fn remap_gene(gene: &mut Self::Gene, map: &HashMap<NeuronID, NeuronID>) {}
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

pub trait Collector: Debug + Default {
    /// Will be received from [`Propagator::Output`].
    type Input<'i>
    where
        Self: 'i;
    /// Will be send to [`Activator::Input`]
    type Output<'o>
    where
        Self: 'o;
    type Config: Debug + Default;
    /// Adds new value into the [`Collector`].
    fn push(&mut self, input: Self::Input<'_>, config: &Self::Config);
    /// Returns the collected output.
    fn collect(&mut self, config: &Self::Config) -> Self::Output<'_>;
    /// Resets the state to be ready to collect a new value.
    fn clear(&mut self, config: &Self::Config);
}

#[cfg(test)]
mod test {
    use super::*;

    #[derive(Debug, Clone)]
    struct BiasGene(f64);
    #[derive(Debug, Default)]
    struct WeightedPropagator(f64);
    impl Propagator for WeightedPropagator {
        type Config = ();
        type Gene = BiasGene;
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
        ) -> impl Iterator<Item: Borrow<NeuronID>> {
            std::iter::empty::<NeuronID>()
        }

        fn propagate(
            &mut self,
            input: Self::Input<'_>,
            _modulation: &[Self::Input<'_>],
            gene: &Self::Gene,
            _config: &Self::Config,
        ) -> Self::Output<'_> {
            input * self.0 + gene.0
        }
    }

    #[test]
    fn weighted_propagation() {
        let inputs = vec![0.3, 1.2, -0.7, 0.0];
        let genes = vec![BiasGene(0.0), BiasGene(1.0)];
        let mut propagator = WeightedPropagator::default();
        for input in inputs {
            for gene in &genes {
                let result = propagator.propagate(input, &[], gene, &());
                assert!(result.is_finite(), "{gene:?} | {input:?}");
            }
        }
    }

    #[derive(Debug, Clone)]
    struct ModulatorGene(Option<NeuronID>);
    #[derive(Debug, Default)]
    struct ModulatorPropagator;
    impl Propagator for ModulatorPropagator {
        type Config = ();
        type Gene = ModulatorGene;
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
            gene: &Self::Gene,
            _config: &Self::Config,
        ) -> impl Iterator<Item: Borrow<NeuronID>> {
            gene.0.iter()
        }

        fn propagate(
            &mut self,
            input: Self::Input<'_>,
            modulation: &[Self::Input<'_>],
            _gene: &Self::Gene,
            _config: &Self::Config,
        ) -> Self::Output<'_> {
            input * modulation.first().copied().unwrap_or(1.0)
        }
    }

    #[test]
    fn modulated_propagation() {
        let inputs = vec![0.4, -1.2, 0.7];
        let genes = vec![
            ModulatorGene(None),
            ModulatorGene(NeuronID::try_from(0)),
            ModulatorGene(NeuronID::try_from(2)),
        ];
        let mut propagator = ModulatorPropagator::default();
        let others = vec![0.7, 0.3, -1.2];
        let mut modulation = vec![0.0];
        for input in inputs {
            for gene in &genes {
                modulation.clear();
                for id in propagator.modulation(gene, &()) {
                    let id = id.borrow().into_inner() as usize;
                    assert!(id < others.len(), "modulation out of bounds");
                    modulation.push(others[id]);
                }
                let result = propagator.propagate(input, &modulation, gene, &());
                assert!(result.is_finite(), "{gene:?} | {input:?} | {modulation:?}");
            }
        }
    }

    #[derive(Debug)]
    enum TypedInput {
        Data(f64),
        Control(f64),
    }
    #[derive(Debug, Default)]
    struct TestCollector(f64, f64);
    impl Collector for TestCollector {
        type Config = ();
        type Input<'i>
            = TypedInput
        where
            Self: 'i;
        type Output<'o>
            = (f64, f64)
        where
            Self: 'o;

        fn push(&mut self, input: Self::Input<'_>, _config: &Self::Config) {
            match input {
                TypedInput::Data(value) => self.0 += value,
                TypedInput::Control(value) => self.1 += value,
            }
        }

        fn collect(&mut self, _config: &Self::Config) -> Self::Output<'_> {
            (self.0, self.1)
        }

        fn clear(&mut self, _config: &Self::Config) {
            self.0 = 0.0;
            self.1 = 0.0;
        }
    }

    #[test]
    fn collect_heterogeneous_signals() {
        let inputs = vec![
            TypedInput::Data(1.2),
            TypedInput::Control(0.4),
            TypedInput::Data(-0.7),
            TypedInput::Data(0.5),
            TypedInput::Control(-0.9),
        ];
        let mut collector = TestCollector::default();
        for input in inputs {
            collector.push(input, &());
        }
        let result = collector.collect(&());
        assert!((result.0 - 1.0) <= 1e-6);
        assert!((result.1 + 0.5) <= 1e-6);
        collector.clear(&());
        assert_eq!(collector.collect(&()), (0.0, 0.0));
    }
}
