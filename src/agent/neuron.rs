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
    type Config: Debug + Default;
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

#[cfg(test)]
mod test {
    use super::*;

    #[derive(Debug, Default)]
    struct TestConfig(f64);
    #[derive(Debug, Clone)]
    enum TestGene {
        Identity,
        Threshold,
        Tanh,
    }
    #[derive(Debug, Default)]
    struct ValueActivator(f64);
    impl Activator for ValueActivator {
        type Config = TestConfig;
        type Gene = TestGene;
        type Input<'i>
            = f64
        where
            Self: 'i;
        type Output<'o>
            = f64
        where
            Self: 'o;

        fn activate(&mut self, input: Self::Input<'_>, gene: &Self::Gene, config: &Self::Config) {
            self.0 = match gene {
                TestGene::Identity => input + config.0,
                TestGene::Threshold =>
                    if input >= config.0 {
                        1.0
                    } else {
                        0.0
                    },
                TestGene::Tanh => input.tanh() + config.0,
            };
        }

        fn output(&self) -> Self::Output<'_> {
            self.0
        }
    }

    #[test]
    fn can_pass_by_value() {
        let configs = vec![TestConfig(0.0), TestConfig(1.0)];
        let genes = vec![TestGene::Identity, TestGene::Threshold, TestGene::Tanh];
        let inputs = vec![-2.0, -0.5, 0.0, 0.5, 2.0];
        let mut activator = ValueActivator::default();
        for input in inputs {
            for config in &configs {
                for gene in &genes {
                    activator.activate(input, gene, config);
                    assert!(activator.output().is_finite(), "{config:?} | {gene:?} | {input:?}")
                }
            }
        }
    }

    #[derive(Debug, Default)]
    struct RefActivator(f64);
    impl Activator for RefActivator {
        type Config = ();
        type Gene = ();
        type Input<'i>
            = &'i [f64]
        where
            Self: 'i;
        type Output<'o>
            = &'o f64
        where
            Self: 'o;

        fn activate(&mut self, input: Self::Input<'_>, _gene: &Self::Gene, _config: &Self::Config) {
            self.0 = input.iter().copied().sum();
        }

        fn output(&self) -> Self::Output<'_> {
            &self.0
        }
    }

    #[test]
    fn can_pass_by_ref() {
        let input = vec![0.2, 0.7, 0.1, -0.5];
        let mut activator = RefActivator::default();
        activator.activate(&input, &(), &());
        assert!((*activator.output() - 0.5) < 1e-6);
    }
}
