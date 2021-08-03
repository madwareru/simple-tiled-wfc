use bitsetium::{BitSearch, BitSet, BitEmpty, BitIntersection, BitUnion, BitTestNone};
use std::hash::Hash;

/// A module which contains an info about errors which may occur during Wave Function Collapse
pub mod errors;

/// A grid variant of algorithm implementation
pub mod grid_generation;

/// A voxel variant of algorithm implementation
pub mod voxel_generation;
mod grid_drawing;

/// An iterator which lets you iterate on bits of an **iterated** bitset which have been set
///
/// ### Example usage:
/// ```
/// use simple_tiled_wfc::BitsIterator;
/// let bitset: [u8;2] = [0b00101001; 2];
/// let mut bits = Vec::new();
/// for bit in BitsIterator::new(&bitset) {
///     bits.push(bit);
/// }
/// assert_eq!(vec![0, 3, 5, 8, 11, 13], bits);
/// ```
pub struct BitsIterator<'a, T: BitSearch>  {
    iterated: &'a T,
    idx: usize
}

impl<'a, T: BitSearch> BitsIterator<'a, T> {
    /// A constructor for **BitsIterator**
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

/// A helper function which lets to know an amount of bits which is are set in a **bit_set**
pub fn get_bits_set_count<'a, T: BitSearch>(bit_set: &T) -> usize {
    BitsIterator::new(bit_set).fold(0, |acc, _| acc + 1)
}

/// A helper function which lets to create a bitset with an exactly one **bit** set
pub fn make_one_bit_entry<TBitSet: BitEmpty+BitSet>(bit: usize) -> TBitSet {
    let mut slot = TBitSet::empty();
    slot.set(bit);
    slot
}

/// A helper function which lets to create a bitset with a starting **size** of bits set
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