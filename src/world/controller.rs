use crate::agent::Phenotype;

pub trait Controller {
    type Phenotype: Phenotype;
    type SensorOutput;
    type ActionInput;

    fn read_sensors<'s>(
        &self,
        sensors: impl IntoIterator<Item = &'s <Self::Phenotype as Phenotype>::SensorGene>,
        outputs: &mut Vec<Self::SensorOutput>,
    ) -> bool
    where
        Self::Phenotype: 's;

    fn perform_actions<'a>(
        &mut self,
        actions: impl IntoIterator<Item = &'a <Self::Phenotype as Phenotype>::ActionGene>,
        inputs: &[Self::ActionInput],
    ) where
        Self::Phenotype: 'a;
}
