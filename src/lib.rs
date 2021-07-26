use bitsetium::BitSearch;

pub mod errors;
pub mod grid_generation;
pub mod voxel_generation;

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