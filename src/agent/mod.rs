mod body;
mod brain;
mod connection;
mod genome;
mod index;
mod neuron;
mod state;

pub use body::*;
pub use brain::*;
pub use connection::*;
pub use genome::*;
pub use index::*;
pub use neuron::*;
pub use state::*;

#[derive(Debug, Clone)]
pub struct Agent<G, P>
where
    G: Genome,
    P: Phenotype,
{
    brain:  Brain<G::Activator, G::Propagator>,
    body:   Body<P>,
    genome: G,
}

impl<G, P> Agent<G, P>
where
    G: Genome,
    P: Phenotype,
{
    pub fn brain(&self) -> &Brain<G::Activator, G::Propagator> {
        &self.brain
    }

    pub fn body(&self) -> &Body<P> {
        &self.body
    }
}
