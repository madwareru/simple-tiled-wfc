use {
    rand::{thread_rng, Rng},
    std::{
        hash::Hash,
        collections::{HashMap, VecDeque}
    },
    bitsetium::{BitSearch, BitEmpty, BitSet, BitIntersection, BitUnion, BitTestNone},
    crate::{get_bits_set_count, make_one_bit_entry, errors::WfcError, BitsIterator}
};

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
}

#[derive(Default)]
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
}

#[derive(Default)]
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
    buckets: Vec<Vec<usize>>,
    history: VecDeque<(usize, TBitSet)>
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
        entropy_heuristic: TEntropyHeuristic,
        entropy_choice_heuristic: TEntropyChoiceHeuristic
    ) -> Self {
        let mut grid: Vec<TBitSet> = Vec::new();
        let mut buckets: Vec<Vec<usize>> = vec![Vec::new(); modules.len()+1];
        let initial_probabilities = make_initial_probabilities(modules);
        let mut history = VecDeque::new();
        for idx in 0..(width * height) {
            buckets[modules.len()].push(idx);
            history.push_back((idx, initial_probabilities));
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
            entropy_heuristic,
            entropy_choice_heuristic,
            buckets,
            history
        }
    }

    pub fn from_existing_collapse(
        modules: &'a [WfcModule<TBitSet>],
        width: usize,
        height: usize,
        entropy_heuristic: TEntropyHeuristic,
        entropy_choice_heuristic: TEntropyChoiceHeuristic,
        collapse: &[usize]
    ) -> Self {
        assert_eq!(collapse.len(), width * height);

        let mut grid: Vec<TBitSet> = Vec::new();
        let mut buckets: Vec<Vec<usize>> = vec![Vec::new(); modules.len()+1];
        let mut history = VecDeque::new();
        for idx in 0..(width * height) {
            buckets[1].push(idx);
            grid.push(make_one_bit_entry(collapse[idx]));
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
            entropy_heuristic,
            entropy_choice_heuristic,
            buckets,
            history
        }
    }

    pub fn local_collapse(
        &mut self,
        row: usize,
        column: usize,
        module: usize
    ) -> Result<Vec<usize>, WfcError> {
        let idx = row * self.width + column;
        let value = make_one_bit_entry(module);
        self.set(idx, value);

        let mut tier = Vec::new();
        let mut tier_probabilities: Vec<TBitSet> = Vec::new();
        self.propagate_neighbour_tier(idx, &mut tier);
        for &id in tier.iter() {
            if id != idx {
                self.set(id, make_initial_probabilities(self.modules));
            }
        }
        for _ in 0..16 {
            // we are trying multiple awful stuff to make our thing look better :)
            // here we have some kind of convolution and we are trying to make it in clear phases
            tier_probabilities.clear();
            for &id in tier.iter() {
                if id != idx {
                    self.propagate_backward(id, &mut tier_probabilities);
                } else {
                    tier_probabilities.push(value)
                }
            }
            for (&id, &prob) in tier.iter().zip(tier_probabilities.iter()) {
                self.set(id, prob);
            }
        }

        let result = self.collapse(10);
        if result.is_ok() { return result }

        for _ in 0..30 {
            let mut next_tier = Vec::new();
            let mut tier_probabilities: Vec<TBitSet> = Vec::new();
            for &prev_id in tier.iter() {
                next_tier.push(prev_id);
                self.propagate_neighbour_tier(prev_id, &mut next_tier);
            }
            for &id in next_tier.iter() {
                if id != idx {
                    self.set(id, make_initial_probabilities(self.modules));
                }
            }
            for _ in 0..16 {
                // we are trying multiple awful stuff to make our thing look better :)
                // here we have some kind of convolution and we are trying to make it in clear phases
                tier_probabilities.clear();
                for &id in next_tier.iter() {
                    if id != idx {
                        self.propagate_backward(id, &mut tier_probabilities);
                    } else {
                        tier_probabilities.push(value)
                    }
                }
                for (&id, &prob) in next_tier.iter().zip(tier_probabilities.iter()) {
                    self.set(id, prob);
                }
            }
            tier = next_tier;

            let result = self.collapse(10);
            if result.is_ok() { return result }
        }

        Err(WfcError::TooManyContradictions)
    }

    fn propagate_backward(&mut self, id: usize, probs: &mut Vec<TBitSet>) {
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
        probs.push(probability_set);
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
            self.history.push_back((idx, initial_probabilities));
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
        self.history.push_back((idx, value));
    }

    pub fn set_module(&mut self, row: usize, column: usize, module: usize) {
        let idx = row * self.width + column;
        self.set(idx, make_one_bit_entry(module));
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
                self.history.push_back((i, old_grid[i]));
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

    pub fn become_history(self) -> VecDeque<(usize, TBitSet)> {
        self.history
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
        let new_slot = make_one_bit_entry(next_bit);
        self.grid[slot_id] = new_slot;
        self.history.push_back((slot_id, new_slot));
        self.buckets[min_bucket_id].remove(next_slot_id_in_bucket);
        self.buckets[1].push(slot_id);
        propagation_queue.push_back(slot_id);
    }

    fn east_neighbours(&mut self, module_variants: &TBitSet) -> TBitSet {
        self.east_memoizer
            .get(module_variants)
            .map(|it| *it)
            .unwrap_or_else(|| {
                let mut set = TBitSet::empty();
                for module_id in BitsIterator::new(module_variants) {
                    set = set.union(self.modules[module_id].east_neighbours);
                }
                self.east_memoizer.insert(module_variants.clone(), set);
                set
            })
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
