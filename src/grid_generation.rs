use {
    rand::{thread_rng},
    std::{hash::Hash, collections::{HashMap, VecDeque}},
    bitsetium::{BitSearch, BitEmpty, BitSet, BitIntersection, BitUnion, BitTestNone},
    crate::{get_bits_set_count, errors::WfcError, BitsIterator}
};
use rand::Rng;

struct NeighbourQueryResult {
    north: Option<usize>,
    south: Option<usize>,
    east: Option<usize>,
    west: Option<usize>,
}

pub trait WfcEntropyHeuristic<TBitSet>
    where TBitSet:
    BitSearch + BitEmpty + BitSet + BitIntersection +
    BitUnion + BitTestNone + Hash + Eq + Copy + BitIntersection<Output = TBitSet> +
    BitUnion<Output = TBitSet>
{
    fn choose_next_collapsed_slot(
        &self,
        width: usize,
        height: usize,
        modules: &[WfcModule<TBitSet>],
        available_indices: &[usize]
    ) -> usize;

    fn new() -> Self;
}

pub struct DefaultEntropyHeuristic {
    _noop: u8
}
impl<TBitSet> WfcEntropyHeuristic<TBitSet> for DefaultEntropyHeuristic
    where TBitSet:
    BitSearch + BitEmpty + BitSet + BitIntersection +
    BitUnion + BitTestNone + Hash + Eq + Copy + BitIntersection<Output = TBitSet> +
    BitUnion<Output = TBitSet> {
    fn choose_next_collapsed_slot(
        &self,
        _width: usize,
        _height: usize,
        _modules: &[WfcModule<TBitSet>],
        available_indices: &[usize]
    ) -> usize {
        let mut rng = thread_rng();
        rng.gen_range(0, available_indices.len())
    }
    fn new() -> Self { Self { _noop: Default::default() } }
}

pub trait WfcEntropyChoiceHeuristic<TBitSet>
    where TBitSet:
    BitSearch + BitEmpty + BitSet + BitIntersection +
    BitUnion + BitTestNone + Hash + Eq + Copy + BitIntersection<Output = TBitSet> +
    BitUnion<Output = TBitSet>
{
    fn choose_least_entropy_bit(
        &self,
        width: usize,
        height: usize,
        row: usize,
        column: usize,
        modules: &[WfcModule<TBitSet>],
        slot_bits: &TBitSet,
    ) -> usize;

    fn new() -> Self;
}

pub struct DefaultEntropyChoiceHeuristic {
    _noop: u8
}
impl<TBitSet> WfcEntropyChoiceHeuristic<TBitSet> for DefaultEntropyChoiceHeuristic
    where TBitSet:
    BitSearch + BitEmpty + BitSet + BitIntersection +
    BitUnion + BitTestNone + Hash + Eq + Copy + BitIntersection<Output = TBitSet> +
    BitUnion<Output = TBitSet> {
    fn choose_least_entropy_bit(
        &self,
        _width: usize,
        _height: usize,
        _row: usize,
        _column: usize,
        _modules: &[WfcModule<TBitSet>],
        slot_bits: &TBitSet
    ) -> usize
    {
        let mut rng = thread_rng();
        let random_bit_id = rng.gen_range(0, get_bits_set_count(slot_bits));
        let mut iterator = BitsIterator::new(slot_bits);
        iterator.nth(random_bit_id).unwrap()
    }

    fn new() -> Self { Self { _noop: Default::default() } }
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

pub struct WfcContext<'a, TBitSet, TEntropyHeuristic = DefaultEntropyHeuristic, TEntropyChoiceHeuristic = DefaultEntropyChoiceHeuristic>
    where
    TBitSet:
        BitSearch + BitEmpty + BitSet + BitIntersection + BitUnion +
        BitTestNone + Hash + Eq + Copy + BitIntersection<Output = TBitSet> +
        BitUnion<Output = TBitSet>,
    TEntropyHeuristic: WfcEntropyHeuristic<TBitSet>,
    TEntropyChoiceHeuristic: WfcEntropyChoiceHeuristic<TBitSet>
{
    modules: &'a [WfcModule<TBitSet>],
    width: usize,
    height: usize,
    grid: Vec<TBitSet>,
    north_memoizer: HashMap<TBitSet, TBitSet>,
    south_memoizer: HashMap<TBitSet, TBitSet>,
    east_memoizer: HashMap<TBitSet, TBitSet>,
    west_memoizer: HashMap<TBitSet, TBitSet>,
    entropy_heuristic: TEntropyHeuristic,
    entropy_choice_heuristic: TEntropyChoiceHeuristic,
    buckets: Vec<Vec<usize>>
}

impl<'a, TBitSet, TEntropyHeuristic, TEntropyChoiceHeuristic> WfcContext<'a, TBitSet, TEntropyHeuristic, TEntropyChoiceHeuristic>
    where
    TBitSet:
        BitSearch + BitEmpty + BitSet + BitIntersection + BitUnion +
        BitTestNone + Hash + Eq + Copy + BitIntersection<Output = TBitSet> +
        BitUnion<Output = TBitSet>,
    TEntropyHeuristic: WfcEntropyHeuristic<TBitSet>,
    TEntropyChoiceHeuristic: WfcEntropyChoiceHeuristic<TBitSet>
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
            entropy_heuristic: TEntropyHeuristic::new(),
            entropy_choice_heuristic: TEntropyChoiceHeuristic::new(),
            buckets
        }
    }

    pub fn from_existing_collapse(
        modules: &'a [WfcModule<TBitSet>],
        width: usize,
        height: usize,
        collapse: &[usize]
    ) -> Self {
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
            entropy_heuristic: TEntropyHeuristic::new(),
            entropy_choice_heuristic: TEntropyChoiceHeuristic::new(),
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
        self.set(idx, value);

        let mut tier_1 = Vec::new();
        self.propagate_neighbour_tier(idx, &mut tier_1);
        for &id in tier_1.iter() {
            if id != idx {
                self.set(id, make_initial_probabilities(self.modules));
            }
        }
        for &id in tier_1.iter() {
            self.propagate_backward(idx, id);
        }

        let result = self.collapse(10);
        if result.is_ok() { return result }

        let mut tier_2 = Vec::new();
        for &prev_id in tier_1.iter() {
            tier_2.push(prev_id);
            self.propagate_neighbour_tier(prev_id, &mut tier_2);
        }
        for &id in tier_2.iter() {
            if id != idx {
                self.set(id, make_initial_probabilities(self.modules));
            }
        }
        for &id in tier_2.iter() {
            self.propagate_backward(idx, id);
        }

        let result = self.collapse(10);
        if result.is_ok() { return result }

        let mut tier_3 = Vec::new();
        for &prev_id in tier_2.iter() {
            tier_3.push(prev_id);
            self.propagate_neighbour_tier(prev_id, &mut tier_3);
        }
        for &id in tier_3.iter() {
            if id != idx {
                self.set(id, make_initial_probabilities(self.modules));
            }
        }
        for &id in tier_3.iter() {
            self.propagate_backward(idx, id);
        }

        let result = self.collapse(10);
        if result.is_ok() { return result }

        let mut tier_4 = Vec::new();
        for &prev_id in tier_3.iter() {
            tier_4.push(prev_id);
            self.propagate_neighbour_tier(prev_id, &mut tier_4);
        }
        for &id in tier_4.iter() {
            if id != idx {
                self.set(id, make_initial_probabilities(self.modules));
            }
        }
        for &id in tier_4.iter() {
            self.propagate_backward(idx, id);
        }

        let result = self.collapse(10);
        if result.is_ok() { return result }

        let mut tier_5 = Vec::new();
        for &prev_id in tier_4.iter() {
            tier_5.push(prev_id);
            self.propagate_neighbour_tier(prev_id, &mut tier_5);
        }
        for &id in tier_5.iter() {
            if id != idx {
                self.set(id, make_initial_probabilities(self.modules));
            }
        }
        for &id in tier_5.iter() {
            self.propagate_backward(idx, id);
        }

        let result = self.collapse(10);
        if result.is_ok() { return result }

        let mut tier_6 = Vec::new();
        for &prev_id in tier_5.iter() {
            tier_6.push(prev_id);
            self.propagate_neighbour_tier(prev_id, &mut tier_6);
        }
        for &id in tier_6.iter() {
            if id != idx {
                self.set(id, make_initial_probabilities(self.modules));
            }
        }
        for &id in tier_6.iter() {
            self.propagate_backward(idx, id);
        }

        let result = self.collapse(10);
        if result.is_ok() { return result }

        let mut tier_7 = Vec::new();
        for &prev_id in tier_6.iter() {
            tier_7.push(prev_id);
            self.propagate_neighbour_tier(prev_id, &mut tier_7);
        }
        for &id in tier_7.iter() {
            if id != idx {
                self.set(id, make_initial_probabilities(self.modules));
            }
        }
        for &id in tier_7.iter() {
            self.propagate_backward(idx, id);
        }

        let result = self.collapse(10);
        if result.is_ok() { return result }

        let mut tier_8 = Vec::new();
        for &prev_id in tier_7.iter() {
            tier_8.push(prev_id);
            self.propagate_neighbour_tier(prev_id, &mut tier_8);
        }
        for &id in tier_8.iter() {
            if id != idx {
                self.set(id, make_initial_probabilities(self.modules));
            }
        }
        for &id in tier_8.iter() {
            self.propagate_backward(idx, id);
        }

        let result = self.collapse(10);
        if result.is_ok() { return result }

        let mut tier_9 = Vec::new();
        for &prev_id in tier_8.iter() {
            tier_9.push(prev_id);
            self.propagate_neighbour_tier(prev_id, &mut tier_9);
        }
        for &id in tier_9.iter() {
            if id != idx {
                self.set(id, make_initial_probabilities(self.modules));
            }
        }
        for &id in tier_9.iter() {
            self.propagate_backward(idx, id);
        }

        self.collapse(10)
    }

    fn propagate_backward(&mut self, id_to_ignore: usize, id: usize) {
        if id == id_to_ignore {
            return;
        }

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

    fn propagate_neighbour_tier(&mut self, idx: usize, neighbour_tier: &mut Vec<usize>) {
        let neighbours = self.get_neighbours(idx);
        if let Some(west_neighbour) = neighbours.west {
            if !neighbour_tier.iter().any(|it| *it == west_neighbour) {
                neighbour_tier.push(west_neighbour);
            }
        }
        if let Some(east_neighbour) = neighbours.east {
            if !neighbour_tier.iter().any(|it| *it == east_neighbour) {
                neighbour_tier.push(east_neighbour);
            }
        }
        if let Some(north_neighbour) = neighbours.north {
            if !neighbour_tier.iter().any(|it| *it == north_neighbour) {
                neighbour_tier.push(north_neighbour);
            }
        }
        if let Some(south_neighbour) = neighbours.south {
            if !neighbour_tier.iter().any(|it| *it == south_neighbour) {
                neighbour_tier.push(south_neighbour);
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
                self.collapse_next_slot(&mut propagation_queue, min_bucket_id);

                // III. While propagation queue is not empty, propagate probability set to neighbours
                // If neighbour's probability set is changed, add its index to a propagation queue
                self.propagate(&mut propagation_queue);
            }

            // In the case of backtrack we need to bring the state of a grid back to what it was
            // at the beginning. The propagation queue need to be flushed too obviously
            for i in 0..self.grid.len() {
                self.grid[i] = old_grid[i];
            }
            self.buckets = old_buckets.clone();
            propagation_queue.clear();

            if contradictions_allowed == 0 {
                break 'outer;
            }

            contradictions_allowed -= 1;
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

    fn collapse_next_slot(
        &mut self,
        propagation_queue: &mut VecDeque<usize>,
        min_bucket_id: usize
    ) {
        let next_slot_id_in_bucket = self.entropy_heuristic.choose_next_collapsed_slot(
            self.width,
            self.height,
            self.modules,
            &self.buckets[min_bucket_id]
        );
        let slot_id = self.buckets[min_bucket_id][next_slot_id_in_bucket];
        let next_bit = self.entropy_choice_heuristic.choose_least_entropy_bit(
            self.width,
            self.height,
            slot_id / self.width,
            slot_id % self.width,
            self.modules,
            &self.grid[slot_id]
        );
        let new_slot = {
            let mut slot = TBitSet::empty();
            slot.set(next_bit);
            slot
        };
        self.grid[slot_id] = new_slot;
        self.buckets[min_bucket_id].remove(next_slot_id_in_bucket);
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
