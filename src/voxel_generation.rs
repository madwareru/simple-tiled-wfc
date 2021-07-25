use {
    rand::{thread_rng, distributions::Uniform, prelude::{Distribution, ThreadRng}},
    std::{hash::Hash, collections::{HashMap, VecDeque}},
    bitsetium::{BitSearch, BitEmpty, BitSet, BitIntersection, BitUnion, BitTestNone},
    crate::{get_bits_set_count, errors::WfcError, BitsIterator}
};

struct NeighbourQueryResult {
    north: Option<usize>,
    south: Option<usize>,
    east: Option<usize>,
    west: Option<usize>,
    upper: Option<usize>,
    bottom: Option<usize>,
}

pub struct WfcModule<TBitSet>
    where TBitSet:
    BitSearch + BitEmpty + BitSet + BitIntersection +
    BitUnion + BitTestNone + Hash + Eq + Copy + BitIntersection<Output = TBitSet> +
    BitUnion<Output = TBitSet>
{
    pub north_neighbours: TBitSet,
    pub south_neighbours: TBitSet,
    pub east_neighbours: TBitSet,
    pub west_neighbours: TBitSet,
    pub upper_neighbours: TBitSet,
    pub bottom_neighbours: TBitSet,
}

impl<TBitSet> WfcModule<TBitSet>
    where TBitSet:
    BitSearch + BitEmpty + BitSet + BitIntersection +
    BitUnion + BitTestNone + Hash + Eq + Copy + BitIntersection<Output = TBitSet> +
    BitUnion<Output = TBitSet>
{
    pub fn new() -> Self {
        Self {
            north_neighbours: TBitSet::empty(),
            south_neighbours: TBitSet::empty(),
            east_neighbours: TBitSet::empty(),
            west_neighbours: TBitSet::empty(),
            upper_neighbours: TBitSet::empty(),
            bottom_neighbours: TBitSet::empty()
        }
    }

    pub fn add_north_neighbour(&mut self, idx: usize) { self.north_neighbours.set(idx) }
    pub fn add_south_neighbour(&mut self, idx: usize) { self.south_neighbours.set(idx) }
    pub fn add_east_neighbour(&mut self, idx: usize) { self.east_neighbours.set(idx) }
    pub fn add_west_neighbour(&mut self, idx: usize) { self.west_neighbours.set(idx) }
    pub fn add_upper_neighbour(&mut self, idx: usize) { self.upper_neighbours.set(idx) }
    pub fn add_bottom_neighbour(&mut self, idx: usize) { self.bottom_neighbours.set(idx) }
}

fn make_initial_probabilities<TBitSet>(modules: &[WfcModule<TBitSet>]) -> TBitSet
    where TBitSet:
    BitSearch + BitEmpty + BitSet + BitIntersection + BitUnion +
    BitTestNone + Hash + Eq + Copy + BitIntersection<Output = TBitSet> +
    BitUnion<Output = TBitSet>
{
    (0..modules.len())
        .fold(
            TBitSet::empty(),
            |acc, module_id| {
                let mut acc = acc;
                acc.set(module_id);
                acc
            }
        )
}

pub struct WfcContext<'a, TBitSet>
    where TBitSet:
    BitSearch + BitEmpty + BitSet + BitIntersection + BitUnion +
    BitTestNone + Hash + Eq + Copy + BitIntersection<Output = TBitSet> +
    BitUnion<Output = TBitSet>
{
    modules: &'a [WfcModule<TBitSet>],
    x_size: usize,
    z_size: usize,
    y_size: usize,
    voxels: Vec<TBitSet>,
    north_memoizer: HashMap<TBitSet, TBitSet>,
    south_memoizer: HashMap<TBitSet, TBitSet>,
    east_memoizer: HashMap<TBitSet, TBitSet>,
    west_memoizer: HashMap<TBitSet, TBitSet>,
    upper_memoizer: HashMap<TBitSet, TBitSet>,
    bottom_memoizer: HashMap<TBitSet, TBitSet>,
    buckets: Vec<Vec<usize>>
}

impl<'a, TBitSet> WfcContext<'a, TBitSet>
    where TBitSet:
    BitSearch + BitEmpty + BitSet + BitIntersection + BitUnion +
    BitTestNone + Hash + Eq + Copy + BitIntersection<Output = TBitSet> +
    BitUnion<Output = TBitSet>
{
    pub fn new(
        modules: &'a [WfcModule<TBitSet>],
        x_size: usize,
        z_size: usize,
        y_size: usize,
    ) -> Self {
        let mut voxels: Vec<TBitSet> = Vec::new();
        let mut buckets: Vec<Vec<usize>> = vec![Vec::new(); modules.len()+1];
        let initial_probabilities = make_initial_probabilities(modules);
        for idx in 0..(x_size * z_size * y_size) {
            buckets[modules.len()].push(idx);
            voxels.push(initial_probabilities);
        }
        Self {
            modules,
            x_size,
            z_size,
            y_size,
            voxels,
            north_memoizer: HashMap::new(),
            south_memoizer: HashMap::new(),
            east_memoizer: HashMap::new(),
            west_memoizer: HashMap::new(),
            upper_memoizer: HashMap::new(),
            bottom_memoizer: HashMap::new(),
            buckets
        }
    }

    pub fn reset(&mut self) {
        self.north_memoizer.clear();
        self.south_memoizer.clear();
        self.east_memoizer.clear();
        self.west_memoizer.clear();
        self.upper_memoizer.clear();
        self.bottom_memoizer.clear();

        for bucket in self.buckets.iter_mut() {
            bucket.clear();
        }
        let initial_probabilities = make_initial_probabilities(self.modules);
        for idx in 0..(self.x_size * self.z_size * self.y_size ) {
            self.buckets[self.modules.len()].push(idx);
            self.voxels[idx] = initial_probabilities;
        }
    }

    fn set(&mut self, idx: usize, value: TBitSet) {
        let old_v = self.voxels[idx];
        let old_bits_set = get_bits_set_count(&old_v);
        let new_bits_set = get_bits_set_count(&value);

        let len = self.buckets[old_bits_set].len();

        for i in (0..len).rev() {
            if self.buckets[old_bits_set][i].eq(&idx) {
                self.buckets[old_bits_set][i] = self.buckets[old_bits_set][len-1];
                self.buckets[old_bits_set].remove(len-1);
                break;
            }
        }

        self.buckets[new_bits_set].push(idx);
        self.voxels[idx] = value;
    }

    pub fn set_module(&mut self, height: usize, row: usize, column: usize, module: usize) {
        let idx = self.make_idx(height, row, column);
        let mut value = TBitSet::empty();
        value.set(module);
        self.set(idx, value);
        let mut propagation_queue: VecDeque<usize> = VecDeque::new();
        propagation_queue.push_back(idx);
        self.propagate(&mut propagation_queue);
    }

    pub fn collapse(&mut self, max_contradictions: i32) -> Result<Vec<usize>, WfcError> {
        let mut contradictions_allowed = max_contradictions;
        let old_voxels = self.voxels.clone();
        let old_buckets = self.buckets.clone();
        let mut propagation_queue: VecDeque<usize> = VecDeque::new();
        let mut rng = thread_rng();
        'outer: loop {
            'backtrack: loop {
                // I. Detect quit condition
                if !self.buckets[0].is_empty() {
                    break 'backtrack; // we found contradiction and need to backtrack
                }
                let mut min_bucket_id = 1;
                'bucket_search: for i in 2_..self.buckets.len() {
                    if !self.buckets[i].is_empty() {
                        min_bucket_id = i;
                        break 'bucket_search;
                    }
                }
                if min_bucket_id == 1 {
                    return Ok(self.voxels
                        .iter()
                        .map(|it| it.find_first_set(0).unwrap())
                        .collect()); // We are done!
                }

                // II. Choose random slot with a minimum probability set and collapse it's
                // set to just one module
                //println!("collapse no {}", collapse_no);
                self.collapse_random_slot(&mut propagation_queue, &mut rng, min_bucket_id);

                // III. While propagation queue is not empty, propagate probability set to neighbours
                // If neighbour's probability set is changed, add its index to a propagation queue
                self.propagate(&mut propagation_queue);
            }

            // println!("contradiction!");
            // self.print_probability_counts();

            if contradictions_allowed == 0 {
                break 'outer;
            }

            contradictions_allowed -= 1;

            // In the case of backtrack we need to bring the state of a voxels back to what it was
            // at the beginning. The propagation queue need to be flushed too obviously
            for i in 0..self.voxels.len() {
                self.voxels[i] = old_voxels[i];
            }
            self.buckets = old_buckets.clone();
            propagation_queue.clear();
        }
        Err(WfcError::TooManyContradictions)
    }

    fn collapse_random_slot(
        &mut self,
        propagation_queue: &mut VecDeque<usize>,
        mut rng: &mut ThreadRng,
        min_bucket_id: usize
    ) {
        let random_slot_id_in_bucket = {
            let uniform = Uniform::from(0..self.buckets[min_bucket_id].len());
            uniform.sample(&mut rng)
        };
        let slot_id = self.buckets[min_bucket_id][random_slot_id_in_bucket];
        let random_bit = {
            let slot = &self.voxels[slot_id];
            let uniform = Uniform::from(0..get_bits_set_count(slot));
            let random_bit_id = uniform.sample(&mut rng);
            let mut iterator = BitsIterator::new(slot);
            iterator.nth(random_bit_id).unwrap()
        };
        let new_slot = {
            let mut slot = TBitSet::empty();
            slot.set(random_bit);
            slot
        };
        self.voxels[slot_id] = new_slot;
        self.buckets[min_bucket_id].remove(random_slot_id_in_bucket);
        self.buckets[1].push(slot_id);
        propagation_queue.push_back(slot_id);
    }
    fn east_neighbours(&mut self, module_variants: &TBitSet) -> TBitSet {
        match self.east_memoizer.get(module_variants) {
            Some(v) => v.clone(),
            None => {
                let iterator = BitsIterator::new(module_variants);
                let mut set = TBitSet::empty();
                for module_id in iterator {
                    set = set.union(self.modules[module_id].east_neighbours);
                }
                self.east_memoizer.insert(module_variants.clone(), set);
                set
            }
        }
    }
    fn west_neighbours(&mut self, module_variants: &TBitSet) -> TBitSet {
        match self.west_memoizer.get(module_variants) {
            Some(v) => v.clone(),
            None => {
                let iterator = BitsIterator::new(module_variants);
                let mut set = TBitSet::empty();
                for module_id in iterator {
                    set = set.union(self.modules[module_id].west_neighbours);
                }
                self.west_memoizer.insert(module_variants.clone(), set);
                set
            }
        }
    }
    fn north_neighbours(&mut self, module_variants: &TBitSet) -> TBitSet {
        match self.north_memoizer.get(module_variants) {
            Some(v) => v.clone(),
            None => {
                let iterator = BitsIterator::new(module_variants);
                let mut set = TBitSet::empty();
                for module_id in iterator {
                    set = set.union(self.modules[module_id].north_neighbours);
                }
                self.north_memoizer.insert(module_variants.clone(), set);
                set
            }
        }
    }
    fn south_neighbours(&mut self, module_variants: &TBitSet) -> TBitSet {
        match self.south_memoizer.get(module_variants) {
            Some(v) => v.clone(),
            None => {
                let iterator = BitsIterator::new(module_variants);
                let mut set = TBitSet::empty();
                for module_id in iterator {
                    set = set.union(self.modules[module_id].south_neighbours);
                }
                self.south_memoizer.insert(module_variants.clone(), set);
                set
            }
        }
    }
    fn upper_neighbours(&mut self, module_variants: &TBitSet) -> TBitSet {
        match self.upper_memoizer.get(module_variants) {
            Some(v) => v.clone(),
            None => {
                let iterator = BitsIterator::new(module_variants);
                let mut set = TBitSet::empty();
                for module_id in iterator {
                    set = set.union(self.modules[module_id].upper_neighbours);
                }
                self.upper_memoizer.insert(module_variants.clone(), set);
                set
            }
        }
    }
    fn bottom_neighbours(&mut self, module_variants: &TBitSet) -> TBitSet {
        match self.bottom_memoizer.get(module_variants) {
            Some(v) => v.clone(),
            None => {
                let iterator = BitsIterator::new(module_variants);
                let mut set = TBitSet::empty();
                for module_id in iterator {
                    set = set.union(self.modules[module_id].bottom_neighbours);
                }
                self.bottom_memoizer.insert(module_variants.clone(), set);
                set
            }
        }
    }

    fn propagate(&mut self, mut propagation_queue: &mut VecDeque<usize>) {
        'propagation: while !propagation_queue.is_empty() {
            let next_id = propagation_queue.pop_front().unwrap();
            let nbr_ids = self.get_neighbours(next_id);
            for neighbour in &[
                nbr_ids.north, nbr_ids.south,
                nbr_ids.east, nbr_ids.west,
                nbr_ids.upper, nbr_ids.bottom
            ] {
                if let &Some(neighbour_slot_id) = neighbour {
                    if get_bits_set_count(&self.voxels[neighbour_slot_id]) != 1 {
                        self.propagate_slot(&mut propagation_queue, neighbour_slot_id);
                        if self.voxels[neighbour_slot_id].test_none() {
                            // We hit a contradiction.
                            break 'propagation;
                        }
                    }
                }
            }
        }
    }

    fn get_neighbours(&self, idx: usize) -> NeighbourQueryResult {
        let height = idx / self.slice_size();
        let row = (idx % self.slice_size()) / self.row_size();
        let column = (idx % self.slice_size()) % self.row_size();
        NeighbourQueryResult {
            north: if row == 0 { None } else { Some(idx - self.row_size()) },
            south: if row >= self.z_size-1 { None } else { Some(idx + self.row_size()) },
            east: if column >= self.x_size-1 { None } else { Some(idx + 1) },
            west: if column == 0 { None } else { Some(idx - 1) },
            upper: if height >= self.y_size-1 { None } else { Some(idx + self.slice_size()) },
            bottom: if height == 0 { None } else { Some(idx - self.slice_size()) }
        }
    }

    fn slice_size(&self) -> usize { self.x_size * self.z_size }
    fn row_size(&self) -> usize { self.x_size }

    fn make_idx(&self, height: usize, row: usize, column: usize) -> usize {
        self.slice_size() * height + self.row_size() * row + column
    }
    fn propagate_slot(&mut self, propagation_queue: &mut &mut VecDeque<usize>, neighbour_id: usize) {
        let mut probability_set = make_initial_probabilities(self.modules);
        let nbr_ids = self.get_neighbours(neighbour_id);
        if let Some(west_neighbour) = nbr_ids.west {
            let west_neighbour = self.voxels[west_neighbour];
            probability_set = probability_set.intersection(self.east_neighbours(&west_neighbour));
        }
        if let Some(east_neighbour) = nbr_ids.east {
            let east_neighbour = self.voxels[east_neighbour];
            probability_set = probability_set.intersection(self.west_neighbours(&east_neighbour));
        }
        if let Some(north_neighbour) = nbr_ids.north {
            let north_neighbour = self.voxels[north_neighbour];
            probability_set = probability_set.intersection(self.south_neighbours(&north_neighbour));
        }
        if let Some(south_neighbour) = nbr_ids.south {
            let south_neighbour = self.voxels[south_neighbour];
            probability_set = probability_set.intersection(self.north_neighbours(&south_neighbour));
        }
        if let Some(upper_neighbour) = nbr_ids.upper {
            let upper_neighbour = self.voxels[upper_neighbour];
            probability_set = probability_set.intersection(self.bottom_neighbours(&upper_neighbour));
        }
        if let Some(bottom_neighbour) = nbr_ids.bottom {
            let bottom_neighbour = self.voxels[bottom_neighbour];
            probability_set = probability_set.intersection(self.upper_neighbours(&bottom_neighbour));
        }

        if self.voxels[neighbour_id].eq(&probability_set) { return; }

        self.set(neighbour_id, probability_set);

        if probability_set.test_none() { return; }

        propagation_queue.push_back(neighbour_id);
    }
}
