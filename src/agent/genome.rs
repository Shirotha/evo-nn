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
    // TODO: functions to populate agents
}
