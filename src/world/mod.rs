use crate::{agent::*, arena::Arena};

mod controller;

pub use controller::*;

pub struct World<G, C>
where
    // NOTE: `'static` bound is required by generic associated types at the moment
    G: 'static + Genome,
    C: Controller,
{
    agents:        Vec<Agent<G, C::Phenotype>>,
    state:         Vec<State<G::Activator, G::Propagator, G::Collector>>,
    sensor_buffer: Vec<C::SensorOutput>,
    action_buffer: Vec<C::ActionInput>,
    controller:    C,
}

impl<G, C> World<G, C>
where
    // NOTE: `'static` bound is required by generic associated types at the moment
    G: 'static + Genome,
    C: Controller,
    for<'c> <G::Collector as Collector>::Input<'c>: From<&'c C::SensorOutput>,
    for<'p> C::ActionInput: From<<G::Propagator as Propagator>::Input<'p>>,
{
    // TODO: create world without agents
    // TODO: should controller be Default?
    pub fn from_population(
        controller: C,
        agents: Vec<Agent<G, C::Phenotype>>,
        arena: &mut Arena,
    ) -> Self {
        let state = agents
            .iter()
            .map(|agent| State::create_for(agent.brain(), agent.body(), arena))
            .collect();
        Self { agents, state, sensor_buffer: Vec::new(), action_buffer: Vec::new(), controller }
    }

    pub fn step(&mut self, config: &Config<G::Activator, G::Propagator, G::Collector>) {
        for (agent, state) in self.agents.iter().zip(self.state.iter_mut()) {
            if self.controller.read_sensors(agent.body().iter_sensors(), &mut self.sensor_buffer) {
                state.step(agent.brain(), &self.sensor_buffer, &mut self.action_buffer, config);
                self.controller.perform_actions(agent.body().iter_actions(), &self.action_buffer);
            }
        }
        // TODO: step world (should be able to create/destroy agents)
    }

    pub fn agents(&self) -> &[Agent<G, C::Phenotype>] {
        &self.agents
    }
    // TODO: function to process complete generation
}
