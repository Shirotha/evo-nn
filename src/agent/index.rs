mod neuron_id {
    #[rustc_layout_scalar_valid_range_end(0xFFFFFF00)]
    #[rustc_nonnull_optimization_guaranteed]
    #[repr(transparent)]
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
    pub struct NeuronID(u32);
    impl NeuronID {
        pub(super) const MAX: u32 = 0xFFFFFF00;
        pub(super) const ZERO: NeuronID = unsafe { NeuronID(0) };

        pub(super) fn try_from(id: u32) -> Option<Self> {
            // SAFETY: Values not larger than `MAX` are always safe to construct.
            (id <= Self::MAX).then_some(unsafe { Self(id) })
        }

        pub(super) fn next(self) -> Option<Self> {
            // SAFETY: After checking that the value is smaller than `MAX`
            // incrementing it will never exceed `MAX`.
            (self.0 < Self::MAX).then(|| unsafe { Self(self.0 + 1) })
        }

        pub(super) fn prev(self) -> Option<Self> {
            // SAFETY: The lower bound of `NeuronID` is consistent with the lower bound of `u32`
            // so all results of `checked_sub` are valid values for `NeuronID`.
            self.0.checked_sub(1).map(|id| unsafe { Self(id) })
        }

        pub(super) fn into_inner(self) -> u32 {
            self.0
        }
    }
}
use std::{cmp::Ordering, fmt::Display};

pub use neuron_id::*;

impl Display for NeuronID {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ID:{}", self.into_inner())
    }
}

#[derive(Debug, Default)]
pub struct NeuronPool {
    largest_used: Option<NeuronID>,
    free_list:    Vec<NeuronID>,
}
impl NeuronPool {
    /// Create a new [`IndexPool`] with no indices.
    pub fn new() -> Self {
        Self::default()
    }

    /// # Safety
    /// `largest_used` will not be checked.
    pub unsafe fn new_unchecked(
        largest_used: Option<NeuronID>,
        free_list: impl IntoIterator<Item = NeuronID>,
    ) -> Self {
        Self { largest_used, free_list: free_list.into_iter().collect() }
    }

    /// Get a free [`Index`].
    /// Returns `None` is no free indices are availible.
    pub fn take(&mut self) -> Option<NeuronID> {
        self.free_list.pop().or_else(|| {
            let index = Some(self.largest_used.map_or(Some(NeuronID::ZERO), NeuronID::next)?);
            self.largest_used = index;
            index
        })
    }

    /// Gives an [`Index`] back to be re-used.
    /// # Safety
    /// Assumes that `index` was originally taken from the same [`IndexPool`].
    pub unsafe fn give_unchecked(&mut self, index: NeuronID) {
        self.free_list.push(index);
    }

    pub fn used(&mut self) -> InUse {
        self.free_list.sort_unstable();
        // SAFETY: `skip_list` will always be sorted before `InUse` is created.
        // Uniqueness is guarantied by the implementation of `NeuronPool`.
        unsafe { InUse::new_unchecked(self.largest_used, &self.free_list) }
    }
}

mod in_use {
    use super::NeuronID;

    #[derive(Debug, Clone)]
    pub struct InUse<'a> {
        begin:     Option<NeuronID>,
        end:       Option<NeuronID>,
        skip_list: &'a [NeuronID],
    }
    impl<'a> InUse<'a> {
        /// # Safety
        /// Assumes that `skip_list` is sorted in ascending order
        /// and that each entry is a unique ID not larger then `end`.
        pub(super) unsafe fn new_unchecked(
            end: Option<NeuronID>,
            skip_list: &'a [NeuronID],
        ) -> Self {
            #[cfg(debug_assertions)]
            if let Some(end) = end {
                assert!(
                    skip_list.iter().all(|id| *id <= end),
                    "skip_list contains out of bounds elements"
                );
                assert!(skip_list.iter().is_sorted(), "skip_list is not sorted");
                assert!(
                    skip_list.iter().map_windows(|[a, b]| a != b).all(|x| x),
                    "skip_list contains duplicate elements"
                )
            }
            Self { begin: Some(NeuronID::ZERO), end, skip_list }
        }
    }
    impl Iterator for InUse<'_> {
        type Item = NeuronID;

        fn next(&mut self) -> Option<Self::Item> {
            let mut current = self.begin?;
            let end = self.end?;
            while self.skip_list.first().is_some_and(|id| *id == current) {
                current = current.next()?;
                self.skip_list = self.skip_list.split_first()?.1;
            }
            self.begin = current.next();
            (current <= end).then_some(current)
        }

        fn size_hint(&self) -> (usize, Option<usize>) {
            let Some(upper) = self.begin.zip_with(self.end, |begin, end| {
                begin.into_inner().abs_diff(end.into_inner()) as usize + 1
            }) else {
                return (0, Some(0));
            };
            let lower = upper.saturating_sub(self.skip_list.len());
            (lower, Some(lower))
        }

        fn count(self) -> usize
        where
            Self: Sized,
        {
            self.size_hint().0
        }

        fn last(mut self) -> Option<Self::Item>
        where
            Self: Sized,
        {
            self.next_back()
        }

        fn is_sorted(self) -> bool
        where
            Self: Sized,
            Self::Item: PartialOrd,
        {
            true
        }

        fn min(mut self) -> Option<Self::Item>
        where
            Self: Sized,
            Self::Item: Ord,
        {
            self.next()
        }

        fn max(mut self) -> Option<Self::Item>
        where
            Self: Sized,
            Self::Item: Ord,
        {
            self.next_back()
        }
    }
    impl ExactSizeIterator for InUse<'_> {}
    impl DoubleEndedIterator for InUse<'_> {
        fn next_back(&mut self) -> Option<Self::Item> {
            let mut current = self.end?;
            let begin = self.begin?;
            while self.skip_list.last().is_some_and(|id| *id == current) {
                current = current.prev()?;
                self.skip_list = self.skip_list.split_last()?.1;
            }
            self.end = current.prev();
            (current >= begin).then_some(current)
        }
    }
}
pub use in_use::*;

mod neuron_order {
    use std::collections::HashMap;

    use super::*;

    #[derive(Debug, Clone)]
    pub struct NeuronOrder {
        index:        Vec<Option<NeuronID>>,
        count:        u32,
        largest_used: Option<NeuronID>,
    }
    impl NeuronOrder {
        pub fn new() -> Self {
            Self { index: Vec::new(), count: 0, largest_used: None }
        }

        pub fn index(&self, neuron: NeuronID) -> Option<usize> {
            self.index
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
                self.index
                    .get_unchecked(neuron.into_inner() as usize)
                    .unwrap_unchecked()
                    .into_inner() as usize
            }
        }

        /// Returns largest neuron in the order if known.
        /// After removing the largest neuron this becomes unknown.
        /// Knowledge can be restored by truncating.
        pub fn largest_used(&self) -> Option<NeuronID> {
            self.largest_used
        }

        /// Returns number of neurons in the order.
        pub fn count(&self) -> usize {
            self.count as usize
        }

        /// Set location of a neuron in the ordering.
        /// When `index` is `None` the neuron will be removed from the ordering.
        /// # Safety
        /// Assumes that index is not larger than [`NeuronID::MAX`] and not already listed in the order.
        pub unsafe fn set_unchecked(&mut self, neuron: NeuronID, index: Option<usize>) {
            let i = neuron.into_inner() as usize;
            if let Some(index) = index {
                if i >= self.index.len() {
                    self.index.resize(i + 1, None);
                    self.count += 1;
                } else if self.index[i].is_none() {
                    self.count += 1;
                }
                self.index[i] = NeuronID::try_from(index as u32);
            } else if i < self.index.len() && self.index[i].take().is_some() {
                if self.largest_used == Some(neuron) {
                    self.largest_used = None;
                }
                // SAFETY: This can't overflow because `index` was holding at least one neuron when `take` returns `Some`.
                self.count -= 1;
            }
        }

        /// # Panics
        /// Panics if `a` or `b` are out of bounds.
        pub fn swap(&mut self, a: NeuronID, b: NeuronID) {
            self.index.swap(a.into_inner() as usize, b.into_inner() as usize);
        }

        pub fn truncate(&mut self) {
            if let Some(max) = self.largest_used {
                self.index.truncate(max.into_inner() as usize + 1);
            } else if self.count == 0 {
                self.index.clear();
            } else {
                let cutoff = self.index.iter().rev().take_while(|i| i.is_none()).count();
                let len = self.index.len() - cutoff;
                self.index.truncate(len);
                // SAFETY: len is positive and not larger than previous length of `index` so this is always valid.
                self.largest_used = NeuronID::try_from(len as u32 - 1);
            }
        }

        pub fn is_packed(&self) -> bool {
            self.count == 0 || self.largest_used.is_some_and(|id| id.into_inner() + 1 == self.count)
        }

        /// Returns an unused `NeuronID` if availible.
        pub fn find_free(&self) -> Option<NeuronID> {
            if self.is_packed() {
                return self.largest_used.map_or(Some(NeuronID::ZERO), NeuronID::next);
            }
            self.index
                .iter()
                .enumerate()
                .find(|(_, index)| index.is_none())
                .and_then(|(id, _)| NeuronID::try_from(id as u32))
                .or_else(|| NeuronID::try_from(self.index.len() as u32))
        }

        pub fn iter_used(&self) -> impl Iterator<Item = NeuronID> {
            self.index
                .iter()
                .enumerate()
                .filter_map(|(id, index)| index.and_then(|_| NeuronID::try_from(id as u32)))
        }

        pub fn iter_free(&self) -> impl Iterator<Item = NeuronID> {
            self.index.iter().enumerate().filter_map(|(id, index)| {
                index.is_none().then(|| NeuronID::try_from(id as u32)).flatten()
            })
        }

        /// rebuild the `NeuronOrder` with a given ordering.
        /// This will also optimize the `NeuronID`s to minimize storage
        /// returning a mapping that can be used to update the `NeuronID`s in the original collection.
        pub fn rebuild(
            &mut self,
            order: impl Iterator<Item = NeuronID>,
        ) -> HashMap<NeuronID, NeuronID> {
            let map = order
                .enumerate()
                .map(|(i, id)| {
                    (
                        id,
                        NeuronID::try_from(i as u32)
                            .expect("length of order cannot exceed NeuronID::MAX"),
                    )
                })
                .collect::<HashMap<_, _>>();
            self.index.clear();
            self.index.reserve(map.len());
            self.index.extend(map.keys().copied().map(Some));
            self.count = map.len() as u32;
            self.largest_used = self.index.last().copied().flatten();
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
