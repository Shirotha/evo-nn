use std::fmt::Debug;

use thin_vec::ThinVec;

use super::NeuronID;

pub trait Phenotype {
    type SensorGene: Debug + Clone;
    type ActionGene: Debug + Clone;
    // TODO: functions to use in Genome.populate for dealing with Body mutations
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

    pub fn sensors(&self) -> &[Sensor<P::SensorGene>] {
        &self.sensors
    }

    pub fn actions(&self) -> &[Action<P::ActionGene>] {
        &self.actions
    }

    pub fn phenotype(&self) -> &P {
        &self.phenotype
    }
}
