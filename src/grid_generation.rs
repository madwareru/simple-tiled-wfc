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
        }
    }

    pub fn add_north_neighbour(&mut self, idx: usize) {
        self.north_neighbours.set(idx)
    }

    pub fn add_south_neighbour(&mut self, idx: usize) {
        self.south_neighbours.set(idx)
    }

    pub fn add_east_neighbour(&mut self, idx: usize) {
        self.east_neighbours.set(idx)
    }

    pub fn add_west_neighbour(&mut self, idx: usize) {
        self.west_neighbours.set(idx)
    }
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
    width: usize,
    height: usize,
    grid: Vec<TBitSet>,
    north_memoizer: HashMap<TBitSet, TBitSet>,
    south_memoizer: HashMap<TBitSet, TBitSet>,
    east_memoizer: HashMap<TBitSet, TBitSet>,
    west_memoizer: HashMap<TBitSet, TBitSet>,
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
        width: usize,
        height: usize,
    ) -> Self {
        let mut grid: Vec<TBitSet> = Vec::new();
        let mut buckets: Vec<Vec<usize>> = vec![Vec::new(); modules.len()+1];
        let initial_probabilities = make_initial_probabilities(modules);
        for idx in 0..(width * height) {
            buckets[modules.len()].push(idx);
            grid.push(initial_probabilities);
        }
        Self {
            modules,
            width,
            height,
            grid,
            north_memoizer: HashMap::new(),
            south_memoizer: HashMap::new(),
            east_memoizer: HashMap::new(),
            west_memoizer: HashMap::new(),
            buckets
        }
    }

    pub fn from_existing_collapse(
        modules: &'a [WfcModule<TBitSet>],
        width: usize,
        height: usize,
        collapse: &[usize]
    ) -> Self<> {
        assert_eq!(collapse.len(), width * height);

        let mut grid: Vec<TBitSet> = Vec::new();
        let mut buckets: Vec<Vec<usize>> = vec![Vec::new(); modules.len()+1];
        for idx in 0..(width * height) {
            let mut initial = TBitSet::empty();
            initial.set(collapse[idx]);
            buckets[1].push(idx);
            grid.push(initial);
        }
        Self {
            modules,
            width,
            height,
            grid,
            north_memoizer: HashMap::new(),
            south_memoizer: HashMap::new(),
            east_memoizer: HashMap::new(),
            west_memoizer: HashMap::new(),
            buckets
        }
    }

    pub fn local_collapse(
        &mut self,
        row: usize,
        column: usize,
        module: usize
    ) -> Result<Vec<usize>, WfcError> {
        let idx = row * self.width + column;
        let mut value = TBitSet::empty();
        value.set(module);

        let mut first_neighbour_tier = Vec::new();
        self.propagate_neighbour_tier(idx, &mut first_neighbour_tier);
        for &id in first_neighbour_tier.iter() {
            self.propagate_backward(id);
        }

        let result = self.collapse(10);
        if result.is_ok() { return result }

        let mut second_neighbour_tier = Vec::new();
        for &prev_id in first_neighbour_tier.iter() {
            self.propagate_neighbour_tier(idx, &mut second_neighbour_tier);
        }
        for &id in second_neighbour_tier.iter() {
            self.propagate_backward(id);
        }

        let result = self.collapse(10);
        if result.is_ok() { return result }

        let mut third_neighbour_tier = Vec::new();
        for &prev_id in second_neighbour_tier.iter() {
            self.propagate_neighbour_tier(idx, &mut third_neighbour_tier);
        }
        for &id in third_neighbour_tier.iter() {
            self.propagate_backward(id);
        }

        self.collapse(10)
    }

    fn propagate_backward(&mut self, id: usize) {
        let mut probability_set = make_initial_probabilities(self.modules);
        let nbr_ids = self.get_neighbours(id);
        if let Some(west_neighbour) = nbr_ids.west {
            let west_neighbour = self.grid[west_neighbour];
            probability_set = probability_set.intersection(self.east_neighbours(&west_neighbour));
        }
        if let Some(east_neighbour) = nbr_ids.east {
            let east_neighbour = self.grid[east_neighbour];
            probability_set = probability_set.intersection(self.west_neighbours(&east_neighbour));
        }
        if let Some(north_neighbour) = nbr_ids.north {
            let north_neighbour = self.grid[north_neighbour];
            probability_set = probability_set.intersection(self.south_neighbours(&north_neighbour));
        }
        if let Some(south_neighbour) = nbr_ids.south {
            let south_neighbour = self.grid[south_neighbour];
            probability_set = probability_set.intersection(self.north_neighbours(&south_neighbour));
        }
        self.set(id, probability_set);
    }

    fn propagate_neighbour_tier(&mut self, idx: usize, first_neighbour_tier: &mut Vec<usize>) {
        let neighbours = self.get_neighbours(idx);
        if let Some(west_neighbour) = neighbours.west {
            first_neighbour_tier.push(west_neighbour);
            let nbrs = self.get_neighbours(west_neighbour);
            if let Some(nbr) = nbrs.north {
                if !first_neighbour_tier.iter().any(|it| *it == nbr) {
                    first_neighbour_tier.push(nbr);
                }
            }
            if let Some(nbr) = nbrs.south {
                if !first_neighbour_tier.iter().any(|it| *it == nbr) {
                    first_neighbour_tier.push(nbr);
                }
            }
        }
        if let Some(east_neighbour) = neighbours.east {
            first_neighbour_tier.push(east_neighbour);
            let nbrs = self.get_neighbours(east_neighbour);
            if let Some(nbr) = nbrs.north {
                if !first_neighbour_tier.iter().any(|it| *it == nbr) {
                    first_neighbour_tier.push(nbr);
                }
            }
            if let Some(nbr) = nbrs.south {
                if !first_neighbour_tier.iter().any(|it| *it == nbr) {
                    first_neighbour_tier.push(nbr);
                }
            }
        }
        if let Some(north_neighbour) = neighbours.north {
            first_neighbour_tier.push(north_neighbour);
            let nbrs = self.get_neighbours(north_neighbour);
            if let Some(nbr) = nbrs.east {
                if !first_neighbour_tier.iter().any(|it| *it == nbr) {
                    first_neighbour_tier.push(nbr);
                }
            }
            if let Some(nbr) = nbrs.west {
                if !first_neighbour_tier.iter().any(|it| *it == nbr) {
                    first_neighbour_tier.push(nbr);
                }
            }
        }
        if let Some(south_neighbour) = neighbours.south {
            first_neighbour_tier.push(south_neighbour);
            let nbrs = self.get_neighbours(south_neighbour);
            if let Some(nbr) = nbrs.east {
                if !first_neighbour_tier.iter().any(|it| *it == nbr) {
                    first_neighbour_tier.push(nbr);
                }
            }
            if let Some(nbr) = nbrs.west {
                if !first_neighbour_tier.iter().any(|it| *it == nbr) {
                    first_neighbour_tier.push(nbr);
                }
            }
        }
    }

    pub fn reset(&mut self) {
        self.north_memoizer.clear();
        self.south_memoizer.clear();
        self.east_memoizer.clear();
        self.west_memoizer.clear();

        for bucket in self.buckets.iter_mut() {
            bucket.clear();
        }
        let initial_probabilities = make_initial_probabilities(self.modules);
        for idx in 0..(self.width * self.height) {
            self.buckets[self.modules.len()].push(idx);
            self.grid[idx] = initial_probabilities;
        }
    }

    fn set(&mut self, idx: usize, value: TBitSet) {
        let old_v = self.grid[idx];
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
        self.grid[idx] = value;
    }

    pub fn set_module(&mut self, row: usize, column: usize, module: usize) {
        let idx = row * self.width + column;
        let mut value = TBitSet::empty();
        value.set(module);
        self.set(idx, value);
        let mut propagation_queue: VecDeque<usize> = VecDeque::new();
        propagation_queue.push_back(idx);
        self.propagate(&mut propagation_queue);
    }

    pub fn collapse(&mut self, max_contradictions: i32) -> Result<Vec<usize>, WfcError> {
        let mut contradictions_allowed = max_contradictions;
        let old_grid = self.grid.clone();
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
                    return Ok(self.grid
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

            // In the case of backtrack we need to bring the state of a grid back to what it was
            // at the beginning. The propagation queue need to be flushed too obviously
            for i in 0..self.grid.len() {
                self.grid[i] = old_grid[i];
            }
            self.buckets = old_buckets.clone();
            propagation_queue.clear();
        }
        Err(WfcError::TooManyContradictions)
    }

    fn get_neighbours(&self, idx: usize) -> NeighbourQueryResult {
        let row = idx / self.width;
        let column = idx % self.width;
        NeighbourQueryResult {
            north: if row == 0 { None } else {Some(idx-self.width)},
            south: if row >= self.height-1 { None } else {Some(idx+self.width)},
            east: if column >= self.width-1 { None } else {Some(idx+1)},
            west: if column == 0 { None } else {Some(idx-1)}
        }
    }

    fn propagate(&mut self, mut propagation_queue: &mut VecDeque<usize>) {
        'propagation: while !propagation_queue.is_empty() {
            let next_id = propagation_queue.pop_front().unwrap();
            let nbr_ids = self.get_neighbours(next_id);
            for neighbour in &[nbr_ids.north, nbr_ids.south, nbr_ids.east, nbr_ids.west] {
                if let &Some(neighbour_slot_id) = neighbour {
                    if get_bits_set_count(&self.grid[neighbour_slot_id]) != 1 {
                        self.propagate_slot(&mut propagation_queue, neighbour_slot_id);
                        if self.grid[neighbour_slot_id].test_none() {
                            // We hit a contradiction.
                            break 'propagation;
                        }
                    }
                }
            }
        }
    }

    fn propagate_slot(&mut self, propagation_queue: &mut VecDeque<usize>, neighbour_id: usize) {
        let mut probability_set = make_initial_probabilities(self.modules);
        let nbr_ids = self.get_neighbours(neighbour_id);
        if let Some(west_neighbour) = nbr_ids.west {
            let west_neighbour = self.grid[west_neighbour];
            probability_set = probability_set.intersection(self.east_neighbours(&west_neighbour));
        }
        if let Some(east_neighbour) = nbr_ids.east {
            let east_neighbour = self.grid[east_neighbour];
            probability_set = probability_set.intersection(self.west_neighbours(&east_neighbour));
        }
        if let Some(north_neighbour) = nbr_ids.north {
            let north_neighbour = self.grid[north_neighbour];
            probability_set = probability_set.intersection(self.south_neighbours(&north_neighbour));
        }
        if let Some(south_neighbour) = nbr_ids.south {
            let south_neighbour = self.grid[south_neighbour];
            probability_set = probability_set.intersection(self.north_neighbours(&south_neighbour));
        }

        if self.grid[neighbour_id].eq(&probability_set) { return; }

        self.set(neighbour_id, probability_set);

        if probability_set.test_none() { return; }

        propagation_queue.push_back(neighbour_id);
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
            let slot = &self.grid[slot_id];
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
        self.grid[slot_id] = new_slot;
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
}
