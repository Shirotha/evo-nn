use std::{borrow::Borrow, fmt::Debug};

use crate::{
    agent::{self, *},
    arena::Arena,
};

mod controller;

pub use controller::*;

#[expect(type_alias_bounds)]
type StoreRef<'s, G, C: Controller, S> = (&'s Agent<G, C::Phenotype>, &'s S);

pub trait AgentStore<G, C>: Debug + Default
where
    // NOTE: `'static` bound is required by generic associated types at the moment
    G: 'static + Genome,
    C: Controller,
{
    type Score: From<C::Score>;
    type Config: Debug;

    fn insert(&mut self, agent: Agent<G, C::Phenotype>, score: Self::Score);
    fn len(&self) -> usize;
    fn best(&self, config: &Self::Config) -> Option<StoreRef<G, C, Self::Score>>;
    fn drain(&mut self) -> impl Iterator<Item = (Agent<G, C::Phenotype>, Self::Score)>;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    #[allow(unused_variables)]
    fn populate(
        &mut self,
        count: usize,
        config: (&Self::Config, &G::Config),
    ) -> impl Iterator<Item = Agent<G, C::Phenotype>> {
        let len = self.len();
        // SAFETY: `len` is always the count of items returned by `drain`
        unsafe {
            Agent::populate_unchecked(self.drain().map(|(agent, _)| agent), len, count, config.1)
        }
    }
}

// FIXME: for some reason Propagator::Input is required to implement Debug?
// #[derive(Debug)]
pub struct State<G, C>
where
    G: 'static + Genome,
    C: Controller,
{
    pub brain: agent::State<G::Activator, G::Propagator, G::Collector>,
    pub body:  C::State,
}

#[derive(Debug)]
pub struct Config<G, C, S>
where
    G: 'static + Genome,
    C: Controller,
    S: AgentStore<G, C>,
{
    pub brain:      agent::Config<G::Activator, G::Propagator, G::Collector>,
    pub body:       C::Config,
    pub genome:     G::Config,
    pub store:      S::Config,
    pub world_size: u32,
}

pub struct World<G, C, S>
where
    // NOTE: `'static` bound is required by generic associated types at the moment
    G: 'static + Genome,
    C: Controller,
    S: AgentStore<G, C>,
{
    arena:          [Arena; 2],
    agents:         Vec<Agent<G, C::Phenotype>>,
    state:          Vec<State<G, C>>,
    sensor_buffer:  Vec<C::SensorOutput>,
    action_buffer:  Vec<C::ActionInput>,
    command_buffer: Vec<Command<C>>,
    controller:     C,
    store:          S,
}

impl<G, C, S> World<G, C, S>
where
    // NOTE: `'static` bound is required by generic associated types at the moment
    G: 'static + Genome,
    C: Controller,
    S: AgentStore<G, C>,
    for<'c> <G::Collector as Collector>::Input<'c>: From<&'c C::SensorOutput>,
    for<'p> C::ActionInput: From<<G::Propagator as Propagator>::Input<'p>>,
{
    pub fn new(controller: C) -> Self {
        Self {
            arena: [Arena::new(), Arena::new()],
            agents: Vec::new(),
            state: Vec::new(),
            sensor_buffer: Vec::new(),
            action_buffer: Vec::new(),
            command_buffer: Vec::new(),
            controller,
            store: S::default(),
        }
    }

    pub fn initialize(&mut self, config: &Config<G, C, S>) {
        let len = self.agents.len();
        self.agents.extend(
            self.store.populate(config.world_size as usize, (&config.store, &config.genome)),
        );
        self.state.extend(self.agents[len..].iter().map(|agent| State {
            brain: agent::State::create_for(agent.brain(), agent.body(), &mut self.arena[0]),
            body:  self.controller.initial_state(agent.body().phenotype(), &config.body),
        }))
    }

    pub fn step(&mut self, config: &Config<G, C, S>) -> Option<()> {
        let mut i = 0;
        while i < self.agents.len() {
            // SAFETY: agent is always inbounds because of the loop condition
            let agent = unsafe { self.agents.get_unchecked(i) };
            // SAFETY: state has always the same length as agents
            let state = unsafe { self.state.get_unchecked_mut(i) };
            self.controller.read_sensors(
                agent.body().iter_sensors(),
                &mut self.sensor_buffer,
                &config.body,
            );
            state.brain.step(
                agent.brain(),
                &self.sensor_buffer,
                &mut self.action_buffer,
                &config.brain,
            );
            if let Some(score) = self.controller.perform_actions(
                agent.body().iter_actions(),
                &self.action_buffer,
                &config.body,
            ) {
                let agent = self.agents.swap_remove(i);
                self.store.insert(agent, score.into());
                self.state.swap_remove(i);
            } else {
                i += 1;
            }
        }
        self.controller.step(&self.agents, |cmd| self.command_buffer.push(cmd), &config.body)?;
        for cmd in self.command_buffer.drain(..) {
            match cmd {
                Command::Spawn { parents, init } => {
                    let agent = Agent::spawn(
                        parents.map(|i| self.agents.get(*i.borrow()).expect("valid agent index")),
                        &config.genome,
                    );
                    let state = State {
                        brain: agent::State::create_for(
                            agent.brain(),
                            agent.body(),
                            &mut self.arena[0],
                        ),
                        body:  self.controller.create_state(
                            agent.body().phenotype(),
                            init,
                            &config.body,
                        ),
                    };
                    self.agents.push(agent);
                    self.state.push(state);
                },
                Command::Kill(index) => {
                    self.agents.swap_remove(index);
                    self.state.swap_remove(index);
                },
            }
        }
        Some(())
    }

    pub fn finalize(&mut self, config: &Config<G, C, S>) -> Option<StoreRef<G, C, S::Score>> {
        // TODO: consider using an allocator that can reuse memory if this is not good enough
        self.state.iter_mut().for_each(|state| state.brain.move_buffers(&mut self.arena[1]));
        // SAFETY: all data was moved and arena is empty here
        unsafe {
            self.arena[0].free_all();
        }
        self.arena.swap(0, 1);
        self.store.best(&config.store)
    }

    pub fn cycle_par(&mut self, config: &Config<G, C, S>)
    where
        C: Sync + NoGlobalStep,
    {
        self.initialize(config);
        // TODO: run each agent in own task returning final score
        //   each agent can have their own fixed sized sensor/action buffer
        // TODO: batch add all agents to store
        self.finalize(config);
    }

    pub fn agents(&self) -> &[Agent<G, C::Phenotype>] {
        &self.agents
    }
}
