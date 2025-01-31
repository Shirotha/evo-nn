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

    // TODO: add config
    /// # Safety
    /// Assumes that `parents` has at least `parent_count` elements.
    pub unsafe fn populate_unchecked(
        parents: impl IntoIterator<Item = Self>,
        parent_count: usize,
        children_count: usize,
    ) -> impl Iterator<Item = Self> {
        G::populate(
            parents.into_iter().map(|agent| (agent.genome, agent.brain, agent.body)),
            parent_count,
            children_count,
        )
        .map(|(genome, brain, body)| Self { genome, brain, body })
    }

    pub fn populate(
        parents: impl IntoIterator<IntoIter: ExactSizeIterator, Item = Self>,
        count: usize,
    ) -> impl Iterator<Item = Self> {
        let parents = parents.into_iter();
        let parent_count = parents.size_hint().0;
        unsafe { Self::populate_unchecked(parents, parent_count, count) }
    }

    /// # Safety
    /// Assumes that `parents` has at least `parent_count` elements.
    pub unsafe fn spawn_unchecked<'a, I>(parents: I, count: usize) -> Self
    where
        I: IntoIterator<Item = &'a Self>,
        G: 'a,
        P: 'a,
    {
        let (genome, brain, body) = G::spawn(
            parents.into_iter().map(|agent| (&agent.genome, &agent.brain, &agent.body)),
            count,
        );
        Self { genome, brain, body }
    }

    pub fn spawn<'a, I>(parents: I) -> Self
    where
        I: IntoIterator<IntoIter: ExactSizeIterator, Item = &'a Self>,
        G: 'a,
        P: 'a,
    {
        let parents = parents.into_iter();
        let count = parents.size_hint().0;
        unsafe { Self::spawn_unchecked(parents, count) }
    }
}
