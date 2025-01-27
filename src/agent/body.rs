use super::NeuronID;

pub struct Body {
    // TODO: body definition as defined by the environment
}
impl Body {
    pub fn sensor_neurons(&self) -> &[NeuronID] {
        todo!("somehow return neurons corresponding to sensors")
    }
    pub fn action_neurons(&self) -> &[NeuronID] {
        todo!("somehow return neurons corresponding to actions")
    }
}
