pub mod body;
pub mod brain;
pub mod connection;
pub mod genome;
mod index;
pub mod neuron;
pub mod state;

use body::*;
use brain::*;
use connection::*;
use genome::*;
pub use index::*;
use neuron::*;

#[derive(Debug)]
pub struct Agent<G, P>
where
    G: Genome,
    P: Phenotype,
{
    brain:  Brain<G::Activator, G::Propagator>,
    body:   Body<P>,
    genome: G,
}
