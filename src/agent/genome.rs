use super::*;

pub trait Genome {
    type Activator: Activator;
    type Propagator: Propagator;
    // TODO: functions to populate agents
}
