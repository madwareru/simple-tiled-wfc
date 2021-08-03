use {
    rand::{thread_rng, Rng},
    std::{
        hash::Hash,
        collections::{HashMap, VecDeque},
        sync::mpsc::{channel, Sender}
    },
    bitsetium::{BitSearch, BitEmpty, BitSet, BitIntersection, BitUnion, BitTestNone},
    crate::{
        grid_drawing::{get_brush_ranges, DRAW_LOOKUP},
        get_bits_set_count,
        make_one_bit_entry,
        make_initial_probabilities,
        errors::WfcError,
        BitsIterator
    }
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
pub struct DefaultEntropyHeuristic;
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
    ) -> Option<usize>;
}

#[derive(Default)]
pub struct DefaultEntropyChoiceHeuristic;
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
    ) -> Option<usize>
    {
        let mut rng = thread_rng();
        let random_bit_id = rng.gen_range(0, get_bits_set_count(slot_bits));
        let mut iterator = BitsIterator::new(slot_bits);
        Some(iterator.nth(random_bit_id).unwrap())
    }
}

#[derive(Copy, Clone)]
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

enum WfcContextBuilderExtra<'a> {
    General,
    FromExisting { collapse: &'a [usize] }
}

pub struct WfcContextBuilder<'a, TBitSet>
    where TBitSet:
        BitSearch + BitEmpty + BitSet + BitIntersection + BitUnion +
        BitTestNone + Hash + Eq + Copy + BitIntersection<Output = TBitSet> +
        BitUnion<Output = TBitSet>
{
    extra: WfcContextBuilderExtra<'a>,
    modules: &'a [WfcModule<TBitSet>],
    width: usize,
    height: usize,
    entropy_heuristic: Box<dyn WfcEntropyHeuristic<TBitSet>>,
    entropy_choice_heuristic: Box<dyn WfcEntropyChoiceHeuristic<TBitSet>>,
    history_transmitter: Option<Sender<(usize, TBitSet)>>
}

impl<'a, TBitSet> WfcContextBuilder<'a, TBitSet>
    where TBitSet:
        BitSearch + BitEmpty + BitSet + BitIntersection + BitUnion +
        BitTestNone + Hash + Eq + Copy + BitIntersection<Output = TBitSet> +
        BitUnion<Output = TBitSet>
{
    pub fn new(modules: &'a [WfcModule<TBitSet>], width: usize, height: usize) -> Self {
        Self {
            extra: WfcContextBuilderExtra::General,
            modules,
            width,
            height,
            entropy_heuristic: Box::new(DefaultEntropyHeuristic::default()),
            entropy_choice_heuristic: Box::new(DefaultEntropyChoiceHeuristic::default()),
            history_transmitter: None
        }
    }

    pub fn use_existing_collapse(self, collapse: &'a [usize]) -> Self {
        Self {
            extra: WfcContextBuilderExtra::FromExisting { collapse },
            ..self
        }
    }

    pub fn with_entropy_heuristic(
        self,
        heuristic: Box<dyn WfcEntropyHeuristic<TBitSet>>
    ) -> Self {
        Self {
            entropy_heuristic: heuristic,
            ..self
        }
    }

    pub fn with_entropy_choice_heuristic(
        self,
        heuristic: Box<dyn WfcEntropyChoiceHeuristic<TBitSet>>
    ) -> Self {
        Self {
            entropy_choice_heuristic: heuristic,
            ..self
        }
    }

    pub fn with_history_transmitter(
        self,
        history_transmitter: Sender<(usize, TBitSet)>
    ) -> Self {
        Self {
            history_transmitter: Some(history_transmitter),
            ..self
        }
    }

    pub fn build(self) -> WfcContext<'a, TBitSet> {
        match self.extra {
            WfcContextBuilderExtra::General => {
                WfcContext::new(
                    self.modules,
                    self.width,
                    self.height,
                    self.entropy_heuristic,
                    self.entropy_choice_heuristic,
                    self.history_transmitter
                )
            }
            WfcContextBuilderExtra::FromExisting { collapse } => {
                WfcContext::from_existing_collapse(
                    self.modules,
                    self.width,
                    self.height,
                    self.entropy_heuristic,
                    self.entropy_choice_heuristic,
                    collapse,
                    self.history_transmitter
                )
            }
        }
    }
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
    entropy_heuristic: Box<dyn WfcEntropyHeuristic<TBitSet>>,
    entropy_choice_heuristic: Box<dyn WfcEntropyChoiceHeuristic<TBitSet>>,
    buckets: Vec<Vec<usize>>,
    history_transmitter: Option<Sender<(usize, TBitSet)>>
}

macro_rules! neighbour_func_impl {
    ($func_name:ident of $memo_name:ident and $neighbours_member:ident) => {
        fn $func_name(&mut self, module_variants: &TBitSet) -> TBitSet {
            self.$memo_name
                .get(module_variants)
                .map(|it| *it)
                .unwrap_or_else(|| {
                    let mut set = TBitSet::empty();
                    for module_id in BitsIterator::new(module_variants) {
                        set = set.union(self.modules[module_id].$neighbours_member);
                    }
                    self.$memo_name.insert(module_variants.clone(), set);
                    set
                })
        }
    }
}

impl<'a, TBitSet> WfcContext<'a, TBitSet>
    where TBitSet:
        BitSearch + BitEmpty + BitSet + BitIntersection + BitUnion +
        BitTestNone + Hash + Eq + Copy + BitIntersection<Output = TBitSet> +
        BitUnion<Output = TBitSet>
{
    fn new(
        modules: &'a [WfcModule<TBitSet>],
        width: usize,
        height: usize,
        entropy_heuristic: Box<dyn WfcEntropyHeuristic<TBitSet>>,
        entropy_choice_heuristic: Box<dyn WfcEntropyChoiceHeuristic<TBitSet>>,
        history_transmitter: Option<Sender<(usize, TBitSet)>>
    ) -> Self {
        let mut grid: Vec<TBitSet> = Vec::new();
        let mut buckets: Vec<Vec<usize>> = vec![Vec::new(); modules.len()+1];
        let initial_probabilities = make_initial_probabilities(modules.len());
        for idx in 0..(width * height) {
            buckets[modules.len()].push(idx);
            if let Some(sender) = &history_transmitter {
                sender.send((idx, initial_probabilities)).unwrap();
            }
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
            history_transmitter
        }
    }

    fn from_existing_collapse(
        modules: &'a [WfcModule<TBitSet>],
        width: usize,
        height: usize,
        entropy_heuristic: Box<dyn WfcEntropyHeuristic<TBitSet>>,
        entropy_choice_heuristic: Box<dyn WfcEntropyChoiceHeuristic<TBitSet>>,
        collapse: &[usize],
        history_transmitter: Option<Sender<(usize, TBitSet)>>
    ) -> Self {
        assert_eq!(collapse.len(), width * height);

        let mut grid: Vec<TBitSet> = Vec::new();
        let mut buckets: Vec<Vec<usize>> = vec![Vec::new(); modules.len()+1];
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
            history_transmitter
        }
    }

    pub fn local_collapse(
        &mut self,
        row: usize,
        column: usize,
        module: usize,
        result_transmitter: Sender<Result<Vec<usize>, WfcError>>
    ) {
        let old_grid = self.grid.clone();
        let old_buckets = self.buckets.clone();

        let initial_probabilities: TBitSet = make_initial_probabilities(self.modules.len());

        for brush_id in 0..6 {
            // for test just draw a cross and exit
            let (hor_range_dest, vert_range_dest, hor_range_source, vert_range_source) =
                get_brush_ranges(
                    row,
                    column,
                    brush_id,
                    self.width,
                    self.height
                );

            if brush_id > 0 {
                // backtrack
                self.buckets = old_buckets.clone();
                for j in vert_range_dest.clone() {
                    for i in hor_range_dest.clone() {
                        self.grid[j * self.width + i] = old_grid[j * self.width + i];
                    }
                }
            }

            let (v_zip, h_zip) = (
                vert_range_dest.clone().zip(vert_range_source),
                hor_range_dest.clone().zip(hor_range_source)
            );

            let lookup = &(DRAW_LOOKUP[brush_id]);
            for (j_dest, j_source) in v_zip.clone() {
                for (i_dest, i_source) in h_zip.clone() {
                    if lookup[j_source * 15 + i_source] == 1 {
                        continue;
                    }

                    let mut probability_set = initial_probabilities;
                    let idx = j_dest * self.width + i_dest;

                    let neighbours = self.get_neighbours(idx);

                    if neighbours.north.is_some() && lookup[(j_source - 1) * 15 + i_source] == 1 {
                        let north_neighbour_slot = self.grid[neighbours.north.unwrap()];
                        probability_set = probability_set.intersection(
                            self.south_neighbours(&north_neighbour_slot)
                        );
                    }
                    if neighbours.south.is_some() && lookup[(j_source + 1) * 15 + i_source] == 1 {
                        let south_neighbour_slot = self.grid[neighbours.south.unwrap()];
                        probability_set = probability_set.intersection(
                            self.north_neighbours(&south_neighbour_slot)
                        );
                    }
                    if neighbours.east.is_some() && lookup[j_source * 15 + i_source + 1] == 1 {
                        let east_neighbour_slot = self.grid[neighbours.east.unwrap()];
                        probability_set = probability_set.intersection(
                            self.west_neighbours(&east_neighbour_slot)
                        );
                    }
                    if neighbours.west.is_some() && lookup[j_source * 15 + i_source - 1] == 1 {
                        let west_neighbour_slot = self.grid[neighbours.west.unwrap()];
                        probability_set = probability_set.intersection(
                            self.east_neighbours(&west_neighbour_slot)
                        );
                    }
                    self.set(idx, probability_set);
                }
            }
            self.set_module(row, column, module);

            let (tx, rc) = channel();

            self.collapse(10, tx.clone());

            match rc.recv() {
                Ok(res) => {
                    if res.is_ok() {
                        result_transmitter.send(res).unwrap();
                        return;
                    }
                }
                Err(_) => {
                    result_transmitter.send(Err(WfcError::SomeCreepyShit)).unwrap();
                    return;
                }
            }
        }
        result_transmitter.send(Err(WfcError::TooManyContradictions)).unwrap();
    }

    pub fn reset(&mut self) {
        self.north_memoizer.clear();
        self.south_memoizer.clear();
        self.east_memoizer.clear();
        self.west_memoizer.clear();

        for bucket in self.buckets.iter_mut() {
            bucket.clear();
        }
        let initial_probabilities = make_initial_probabilities(self.modules.len());
        for idx in 0..(self.width * self.height) {
            self.buckets[self.modules.len()].push(idx);
            self.grid[idx] = initial_probabilities;
            if let Some(sender) = &self.history_transmitter {
                sender.send((idx, initial_probabilities)).unwrap();
            }
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
        if let Some(sender) = &self.history_transmitter {
            sender.send((idx, value)).unwrap();
        }
    }

    pub fn set_module(&mut self, row: usize, column: usize, module: usize) {
        let idx = row * self.width + column;
        self.set(idx, make_one_bit_entry(module));
        let mut propagation_queue: VecDeque<usize> = VecDeque::new();
        propagation_queue.push_back(idx);
        self.propagate(&mut propagation_queue);
    }

    pub fn collapse(
        &mut self,
        max_contradictions: i32,
        result_transmitter: Sender<Result<Vec<usize>, WfcError>>
    ) {
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
                    result_transmitter.send(Ok(self.grid
                        .iter()
                        .map(|it| it.find_first_set(0).unwrap())
                        .collect()
                    )).unwrap();
                    return; // We are done!
                }

                // II. Choose random slot with a minimum probability set and collapse it's
                // set to just one module
                //println!("collapse no {}", collapse_no);
                if self.collapse_next_slot(&mut propagation_queue, min_bucket_id).is_none() {
                    break 'backtrack; // couldn't find next slot to collapse, need to backtrack
                }

                // III. While propagation queue is not empty, propagate probability set to neighbours
                // If neighbour's probability set is changed, add its index to a propagation queue
                self.propagate(&mut propagation_queue);
            }

            // In the case of backtrack we need to bring the state of a grid back to what it was
            // at the beginning. The propagation queue need to be flushed too obviously
            for i in 0..self.grid.len() {
                self.grid[i] = old_grid[i];
                if let Some(sender) = &self.history_transmitter {
                    sender.send((i, old_grid[i])).unwrap();
                }
            }
            self.buckets = old_buckets.clone();
            propagation_queue.clear();

            if contradictions_allowed == 0 {
                break 'outer;
            }

            contradictions_allowed -= 1;
        }
        result_transmitter.send(Err(WfcError::TooManyContradictions)).unwrap();
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
        let mut probability_set: TBitSet = make_initial_probabilities(self.modules.len());
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
    ) -> Option<TBitSet> {
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
        )?;
        let new_slot = make_one_bit_entry(next_bit);
        self.grid[slot_id] = new_slot;
        if let Some(sender) = &self.history_transmitter {
            sender.send((slot_id, new_slot)).unwrap();
        }
        self.buckets[min_bucket_id].remove(next_slot_id_in_bucket);
        self.buckets[1].push(slot_id);
        propagation_queue.push_back(slot_id);
        Some(new_slot)
    }

    neighbour_func_impl!{ east_neighbours of east_memoizer and east_neighbours }
    neighbour_func_impl!{ west_neighbours of west_memoizer and west_neighbours }
    neighbour_func_impl!{ north_neighbours of north_memoizer and north_neighbours }
    neighbour_func_impl!{ south_neighbours of south_memoizer and south_neighbours }
}
