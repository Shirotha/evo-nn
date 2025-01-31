use std::{
    mem::{self, ManuallyDrop},
    ptr, slice,
};

use thin_slice::ThinBoxedSlice;

pub type Buffer<T> = ManuallyDrop<ThinBoxedSlice<T>>;

/// # Safety
/// Assumes that `src` is a valid aligned pointer to values of type `B`.
unsafe fn cast_slice<A, B>(src: &mut [A]) -> &mut [B] {
    let new_len = if size_of::<A>() == size_of::<B>() {
        src.len()
    } else {
        mem::size_of_val(src) / size_of::<B>()
    };
    // SAFETY: constraits should be validated at callsite
    unsafe { slice::from_raw_parts_mut(src.as_mut_ptr() as *mut B, new_len) }
}

// TODO: validate that casting from slice -> box is ok
#[derive(Debug, Default)]
pub struct Arena(Vec<u8>);
impl Arena {
    pub fn new() -> Self {
        Self::default()
    }

    fn next_aligned<T>(&self) -> usize {
        // FIXME: this will leave a align_of::<T> sized gap when ptr is aligned
        self.0.len() + align_of::<T>() - self.0.as_ptr() as usize % align_of::<T>()
    }

    pub fn alloc_slice_with<T>(&mut self, size: usize, default: impl Fn() -> T) -> Buffer<T> {
        let len = size_of::<T>() * size;
        // TODO: handle zero sized types
        let start = self.next_aligned::<T>();
        self.0.resize(start + len, 0);
        // SAFETY: results from `next_aligned` will always return aligned values
        let result = unsafe { cast_slice(&mut self.0[start..]) };
        result.iter_mut().for_each(|x| unsafe { ptr::write(x, default()) });
        unsafe { ManuallyDrop::new(ThinBoxedSlice::from_raw(result as *mut _)) }
    }

    pub fn alloc_slice_from_iter<T>(&mut self, items: impl IntoIterator<Item = T>) -> Buffer<T> {
        let iter = items.into_iter();
        let stride = size_of::<T>();
        // TODO handle zero sized types
        let start = self.next_aligned::<T>();
        self.0.reserve(start + iter.size_hint().0 - self.0.len());
        self.0.resize(start, 0);
        for mut item in iter {
            // SAFETY: casting to bytes is always valid
            let item = unsafe { cast_slice(slice::from_mut(&mut item)) };
            let start = self.0.len();
            self.0.resize(start + stride, 0);
            self.0[start..].copy_from_slice(item);
        }
        // SAFETY: results from `next_aligned` will always return aligned values
        unsafe {
            ManuallyDrop::new(ThinBoxedSlice::from_raw(cast_slice(&mut self.0[start..]) as *mut _))
        }
    }

    /// # Safety
    /// Assumes that original buffer is not used anymore after moving.
    pub unsafe fn move_into<T>(&mut self, items: &Buffer<T>) -> Buffer<T> {
        let size = items.len() * size_of::<T>();
        let start = self.next_aligned::<T>();
        self.0.resize(start + size, 0);
        // SAFETY: the nessesary amount of data was allocated beforehand
        // also input and output types are the same, so copying their bytes is always valid
        unsafe {
            ptr::copy_nonoverlapping(
                items.as_ptr() as *const u8,
                self.0[start..].as_mut_ptr(),
                size,
            )
        };
        // SAFETY: results from `next_aligned` will always return aligned values
        unsafe {
            ManuallyDrop::new(ThinBoxedSlice::from_raw(cast_slice(&mut self.0[start..]) as *mut _))
        }
    }

    /// # Safety
    /// This does not check for use after free cases.
    pub unsafe fn free_all(&mut self) {
        self.0.clear();
    }
}
