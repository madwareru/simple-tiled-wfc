use bitsetium::{BitSearch, BitSet, BitEmpty, BitIntersection, BitUnion, BitTestNone};
use std::hash::Hash;

pub mod errors;
pub mod grid_generation;
pub mod voxel_generation;
mod grid_drawing;

pub type B256 = [u8; 32];

pub struct BitsIterator<'a, T: BitSearch>  {
    iterated: &'a T,
    idx: usize
}

impl<'a, T: BitSearch> BitsIterator<'a, T> {
    pub fn new(iterated: &'a T) -> Self {
        Self {
            iterated,
            idx: 0
        }
    }
}

impl<'a, T: BitSearch> Iterator for BitsIterator<'a, T> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        match self.iterated.find_first_set(self.idx) {
            None => None,
            Some(id) => {
                self.idx = id + 1;
                Some(id)
            }
        }
    }
}

pub fn get_bits_set_count<'a, T: BitSearch>(bit_set: &T) -> usize {
    BitsIterator::new(bit_set).fold(0, |acc, _| acc + 1)
}

pub fn make_one_bit_entry<TBitSet: BitEmpty+BitSet>(bit: usize) -> TBitSet {
    let mut slot = TBitSet::empty();
    slot.set(bit);
    slot
}

pub fn make_initial_probabilities<TBitSet>(size: usize) -> TBitSet
    where TBitSet:
    BitSearch + BitEmpty + BitSet + BitIntersection + BitUnion +
    BitTestNone + Hash + Eq + Copy + BitIntersection<Output = TBitSet> +
    BitUnion<Output = TBitSet>
{
    (0..size)
        .fold(
            TBitSet::empty(),
            |acc, module_id| {
                let mut acc = acc;
                acc.set(module_id);
                acc
            }
        )
}