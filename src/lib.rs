/*
    # Genetic Data
    Agent
        - has to contain all data needed to create a network simulator and a environemnt stepper
        - should not contain data that was used during runtime only
        - should be able to build its own runtime state
        Brain - direct genome
            Neurons
            Connections
            Inputs - mirrors Body.Sensors
            Ouputs - mirrors Body.Actions
        Body - environemnt interface
            Sensors - extracted from Definition
            Actions - extracted from Definition
            Definition - from specific Environment implementation [generic param in Agent]
                distance(Self, Self, params) - meant to be used by Genome.distance
                populate(?) - meant to be used by Genome.populate
        Genome - custom data and logic [generic param in Agent]
            Data - from specific Genome implementation
            distance(Agent, Agent, Config)
            TODO: how to initialize output when this is the first call
                or have two versions? or use &mut Vec + desired size?
            populate(&[Agent], &mut [Agent], Config)
    Controlled Environment
        Data - from specific Environemnt implementation
        setup(&[Agent], params) -> State - build runtime state for all agents
        step(Self, State) - advance state
        evaluate(Self, State, &mut [Fitness]) - return fitness for all agents in current state
    Natural Environment
        - should be able to run its main loop without outside interference
        - should be able to call Genome.populate during simulation instead of relying on Selector
        Data
        setup(Vec<Agent>, params) -> State
        step(Self, State)
        agents(Self) -> &[Agent] - check current population
    Selector
        - map result of Environment.evaluate to filter of Agent population

    # Additional Runtime Data
    ## Network Evaluation
    Genome
        Brain
            Activator - from specific Genome implementation [associated type]
            Propagator - from specific Genome implementation [associated type]
            Collector - from specific Genome implementation [associated type]
    ## Environment Stepping
    Environment
        State - from specific Environemnt implementation [associated type]
    Genome
        Body
            State - from specific Environment implementation [associated type in Environment]
*/

// NOTE: used in `NeuronPool` implementation.
#![feature(option_zip, iter_map_windows)]
// NOTE: used in `NeuronID` definition.
#![allow(internal_features)]
#![feature(rustc_attrs)]

pub mod agent;
mod arena;
pub mod world;
