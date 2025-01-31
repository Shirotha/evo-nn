use std::fmt::Debug;

use thin_vec::ThinVec;

use super::NeuronID;

// TODO: add config
pub trait Phenotype {
    type SensorGene: Debug + Clone;
    type ActionGene: Debug + Clone;
    // TODO: functions to use in Genome.populate for dealing with Body mutations
    // make sure sensors and actions are sorted using `NeuronOrder::cmp`
}

#[derive(Debug, Clone)]
pub struct Sensor<S> {
    pub neuron: NeuronID,
    pub gene:   S,
}

#[derive(Debug, Clone)]
pub struct Action<A> {
    pub neuron: NeuronID,
    pub gene:   A,
}

#[derive(Debug, Clone)]
pub struct Body<P: Phenotype> {
    sensors:   ThinVec<Sensor<P::SensorGene>>,
    actions:   ThinVec<Action<P::ActionGene>>,
    phenotype: P,
}
impl<P> Body<P>
where
    P: Phenotype,
{
    pub fn iter_sensor_neurons(&self) -> impl Iterator<Item = NeuronID> {
        self.sensors.iter().map(|sensor| sensor.neuron)
    }

    pub fn iter_action_neurons(&self) -> impl Iterator<Item = NeuronID> {
        self.actions.iter().map(|action| action.neuron)
    }

    pub fn iter_sensors(&self) -> impl Iterator<Item = &P::SensorGene> {
        self.sensors.iter().map(|sensor| &sensor.gene)
    }

    pub fn iter_actions(&self) -> impl Iterator<Item = &P::ActionGene> {
        self.actions.iter().map(|action| &action.gene)
    }

    pub fn sensor_count(&self) -> usize {
        self.sensors.len()
    }

    pub fn action_count(&self) -> usize {
        self.actions.len()
    }

    pub fn phenotype(&self) -> &P {
        &self.phenotype
    }
}
