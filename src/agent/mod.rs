pub mod body;
pub mod brain;
pub mod connection;
pub mod genome;
mod index;
pub mod neuron;
pub mod state;

pub use index::*;

#[derive(Debug)]
pub struct Agent {
    // TODO: brain data (network with direct genome of Neuron, Connection)
    // TODO: body data (sensors and action mirroring network input/output and local body parameters)
    // TODO: genome (indirect genome data and mutation/crossover operations)
}

// TODO: simulation state that holds data only needed during evaluation of the environment
