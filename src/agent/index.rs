use std::{cmp::Ordering, fmt::Display, hash::Hash};

mod neuron_id {
    #[rustc_layout_scalar_valid_range_end(0xFFFFFF00)]
    #[rustc_nonnull_optimization_guaranteed]
    #[repr(transparent)]
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
    pub struct NeuronID(u32);
    impl NeuronID {
        pub(super) const MAX: u32 = 0xFFFFFF00;

        #[cfg(not(test))]
        pub(super) fn try_from(id: u32) -> Option<Self> {
            // SAFETY: Values not larger than `MAX` are always safe to construct.
            (id <= Self::MAX).then_some(unsafe { Self(id) })
        }

        #[cfg(test)]
        pub fn try_from(id: u32) -> Option<Self> {
            // SAFETY: Values not larger than `MAX` are always safe to construct.
            (id <= Self::MAX).then_some(unsafe { Self(id) })
        }

        #[cfg(not(test))]
        pub(super) fn into_inner(self) -> u32 {
            self.0
        }

        #[cfg(test)]
        pub fn into_inner(self) -> u32 {
            self.0
        }
    }
}
pub use neuron_id::*;

impl Display for NeuronID {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ID:{}", self.into_inner())
    }
}

mod neuron_order {
    use std::collections::HashMap;

    use thin_vec::ThinVec;

    use super::*;

    #[derive(Debug, Clone)]
    pub struct NeuronOrder(ThinVec<Option<NeuronID>>);
    impl NeuronOrder {
        pub fn new() -> Self {
            Self(ThinVec::new())
        }

        pub fn index(&self, neuron: NeuronID) -> Option<usize> {
            self.0
                .get(neuron.into_inner() as usize)
                .copied()
                .flatten()
                .map(|i| i.into_inner() as usize)
        }

        /// # Safety
        /// Assumes that `neuron` is a valid neuron in the ordering.
        pub unsafe fn index_unchecked(&self, neuron: NeuronID) -> usize {
            // SAFETY: Under the assumptions of this method indexing with `neuron` should always yield a valid index.
            unsafe {
                self.0.get_unchecked(neuron.into_inner() as usize).unwrap_unchecked().into_inner()
                    as usize
            }
        }

        /// Set location of a neuron in the ordering.
        /// When `index` is `None` the neuron will be removed from the ordering.
        /// # Safety
        /// Assumes that index is not larger than [`NeuronID::MAX`] and not already listed in the order.
        pub unsafe fn set_unchecked(&mut self, neuron: NeuronID, index: Option<usize>) {
            let id = neuron.into_inner() as usize;
            if id >= self.0.len() {
                self.0.resize(id + 1, None);
            }
            self.0[id] = index.and_then(|index| NeuronID::try_from(index as u32));
        }

        /// # Panics
        /// Panics if `a` or `b` are out of bounds.
        pub fn swap(&mut self, a: NeuronID, b: NeuronID) {
            self.0.swap(a.into_inner() as usize, b.into_inner() as usize);
        }

        /// Remove unused space at the end of the ordering.
        /// Returns largest [`NeuronID`].
        pub fn truncate(&mut self) -> Option<NeuronID> {
            let cutoff = self.0.iter().rev().take_while(|i| i.is_none()).count();
            let len = self.0.len() - cutoff;
            self.0.truncate(len);
            if len == 0 {
                return None;
            }
            // SAFETY: len is positive and not larger than previous length of `index` so this is always valid.
            NeuronID::try_from(len as u32 - 1)
        }

        /// Returns the next free [`NeuronID`] that is bigger than `start`.
        pub fn next_free(&self, start: Option<NeuronID>) -> Option<NeuronID> {
            let offset = start.map(|id| id.into_inner() + 1).unwrap_or_default();
            self.0
                .iter()
                .skip(offset as usize)
                .enumerate()
                .find(|(_, index)| index.is_none())
                .and_then(|(id, _)| NeuronID::try_from(id as u32))
                .or_else(|| NeuronID::try_from(offset.max(self.0.len() as u32)))
        }

        pub fn iter_used(&self) -> impl Iterator<Item = NeuronID> {
            self.0
                .iter()
                .enumerate()
                .filter_map(|(id, index)| index.and_then(|_| NeuronID::try_from(id as u32)))
        }

        pub fn iter_free(&self) -> impl Iterator<Item = NeuronID> {
            self.0.iter().enumerate().filter_map(|(id, index)| {
                index.is_none().then(|| NeuronID::try_from(id as u32)).flatten()
            })
        }

        /// Create a mapping from any index type to a matching packed [`NeuronID`] ordering.
        pub fn build_mapping<T>(order: T) -> HashMap<T::Item, NeuronID>
        where
            T: IntoIterator,
            T::Item: Eq + Hash,
        {
            order
                .into_iter()
                .enumerate()
                .map(|(i, id)| {
                    (
                        id,
                        NeuronID::try_from(i as u32)
                            .expect("length of order cannot exceed NeuronID::MAX"),
                    )
                })
                .collect()
        }

        // TODO: optimize calling build_mapping -> apply map at callsite -> call rebuild
        /// rebuild the `NeuronOrder` with a given ordering.
        /// This will also optimize the `NeuronID`s to minimize storage
        /// returning a mapping that can be used to update the `NeuronID`s in the original collection.
        pub fn rebuild(
            &mut self,
            order: impl IntoIterator<Item = NeuronID>,
        ) -> HashMap<NeuronID, NeuronID> {
            let map = Self::build_mapping(order);
            self.0.clear();
            self.0.reserve(map.len());
            self.0.extend(map.keys().copied().map(Some));
            map
        }
    }
}
pub use neuron_order::*;

impl NeuronOrder {
    pub fn cmp(&self, lhs: NeuronID, rhs: NeuronID) -> Option<Ordering> {
        self.index(lhs).zip_with(self.index(rhs), |a, b| a.cmp(&b))
    }
}
impl Default for NeuronOrder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn order_can_reorder() {
        let mut order = NeuronOrder::new();
        let id0 = order.next_free(None).unwrap();
        unsafe {
            order.set_unchecked(id0, Some(0));
        }
        let id1 = order.next_free(Some(id0)).unwrap();
        unsafe {
            order.set_unchecked(id1, Some(1));
        }
        assert_eq!(Some(0), order.index(id0));
        assert_eq!(Some(1), order.index(id1));
        order.swap(id0, id1);
        assert_eq!(Some(1), order.index(id0));
        assert_eq!(Some(0), order.index(id1));
    }

    #[test]
    fn order_can_truncate() {
        let mut order = NeuronOrder::new();
        let id0 = order.next_free(None).unwrap();
        unsafe {
            order.set_unchecked(id0, Some(0));
        }
        let id1 = order.next_free(Some(id0)).unwrap();
        unsafe {
            order.set_unchecked(id1, Some(1));
        }
        assert_eq!(2, order.iter_used().count());
        unsafe {
            order.set_unchecked(id1, None);
        }
        assert_eq!(1, order.iter_used().count());
        assert_eq!(1, order.iter_free().count());
        assert_eq!(Some(id0), order.truncate());
        assert_eq!(0, order.iter_free().count());
        assert_eq!(Some(id1), order.next_free(None));
    }

    #[test]
    fn order_can_rebuild() {
        let mut order = NeuronOrder::new();
        let id0 = order.next_free(None).unwrap();
        unsafe {
            order.set_unchecked(id0, Some(0));
        }
        let id1 = order.next_free(Some(id0)).unwrap();
        unsafe {
            order.set_unchecked(id1, Some(1));
        }
        let map = order.rebuild([id1, id0]);
        assert_eq!(2, map.len());
        assert_eq!(id1, map[&id0]);
        assert_eq!(id0, map[&id1]);
    }

    #[test]
    fn order_can_create_from_any_index_type() {
        let data = vec![1, 2, 0];
        let map = NeuronOrder::build_mapping(data.iter().copied());
        let mut order = NeuronOrder::new();
        for i in &data {
            unsafe {
                order.set_unchecked(map[i], Some(*i as usize));
            }
        }
        assert_eq!(std::cmp::Ordering::Equal, data.iter().map(|i| map[i]).cmp(order.iter_used()));
    }
}
