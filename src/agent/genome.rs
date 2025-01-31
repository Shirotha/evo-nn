use std::fmt::Debug;

use super::*;

// A: 'static + for<'a> Activator<Input<'a> = C::Output<'a>, Output<'a> = P::Input<'a>>,
// P: 'static + for<'p> Propagator<Output<'p> = C::Input<'p>>,

pub trait Genome: Debug + Clone {
    type Activator: for<'a> Activator<
            Input<'a> = <Self::Collector as Collector>::Output<'a>,
            Output<'a> = <Self::Propagator as Propagator>::Input<'a>,
        >;
    type Propagator: for<'p> Propagator<Output<'p> = <Self::Collector as Collector>::Input<'p>>;
    type Collector: Collector;

    fn populate<P: Phenotype>(
        parents: impl IntoIterator<Item = (Self, Brain<Self::Activator, Self::Propagator>, Body<P>)>,
        parent_count: usize,
        children_count: usize,
    ) -> impl Iterator<Item = (Self, Brain<Self::Activator, Self::Propagator>, Body<P>)>;

    fn spawn<'a, P, I>(
        parents: I,
        count: usize,
    ) -> (Self, Brain<Self::Activator, Self::Propagator>, Body<P>)
    where
        P: 'a + Phenotype,
        Self: 'a,
        Self::Activator: 'a,
        Self::Propagator: 'a,
        I: IntoIterator<
            Item = (&'a Self, &'a Brain<Self::Activator, Self::Propagator>, &'a Body<P>),
        >;
}
