use std::{borrow::Borrow, fmt::Debug};

use super::*;

/// Marker trait for [`Controller`]s that treat each agent seperate without interaction.
/// This enables parallel processing of agents (see [`World::cylce_par`]).
/// # Safety
/// Types implementing this should not implement [`Controller::step`].
pub unsafe trait NoGlobalStep: Controller {}

/// Commands issued by [`Controller`] to [`World`].
///
/// When multiple commands are issued in the same `step` call the order is important:
/// - send `Spawn` before `Kill` commands
/// - send `Kill` commands with higher index first
///
/// Misordering commands can lead to unexpected results due to agents being moved during command execution.
#[derive(Debug)]
pub enum Command<C>
where
    C: ?Sized + Controller,
{
    Spawn { parents: C::ParentIter, init: C::SpawnHelper },
    Kill(usize),
}

pub trait Controller: Debug {
    type Phenotype: Phenotype;
    type State: Debug;
    type SensorOutput;
    type ActionInput;
    type Score;
    type SpawnHelper;
    type ParentIter: ExactSizeIterator<Item: Borrow<usize>>;
    type Config: Debug + Default;

    fn initial_state(&self, phenotype: &Self::Phenotype, config: &Self::Config) -> Self::State;
    fn create_state(
        &self,
        phenotype: &Self::Phenotype,
        init: Self::SpawnHelper,
        config: &Self::Config,
    ) -> Self::State;

    fn read_sensors<'s>(
        &self,
        sensors: impl IntoIterator<Item = &'s <Self::Phenotype as Phenotype>::SensorGene>,
        outputs: &mut [Self::SensorOutput],
        config: &Self::Config,
    ) where
        Self::Phenotype: 's;

    fn perform_actions<'a>(
        &mut self,
        actions: impl IntoIterator<Item = &'a <Self::Phenotype as Phenotype>::ActionGene>,
        inputs: &[Self::ActionInput],
        config: &Self::Config,
    ) -> Option<Self::Score>
    where
        Self::Phenotype: 'a;

    /// Advances the world state.
    /// Returns `None` when the cycle is complete.
    #[allow(unused_variables)]
    fn step<G>(
        &mut self,
        agents: &[Agent<G, Self::Phenotype>],
        issue_command: impl FnMut(Command<Self>),
        config: &Self::Config,
    ) -> Option<()>
    where
        G: Genome,
    {
        Some(())
    }
}
