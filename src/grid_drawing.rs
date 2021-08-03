pub(crate) const CENTRAL_ID: usize = 7;
pub(crate) const DRAW_RADIUS_OFFSETS: [usize; 7] = [1, 1, 2, 3, 4, 5, 6];

pub(crate) const DRAW_LOOKUP: [[u8; 225]; 6] = [
    [
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,
        1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,
        1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1
    ],
    [
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,
        1,1,1,1,1,0,0,0,0,0,1,1,1,1,1,
        1,1,1,1,1,0,0,0,0,0,1,1,1,1,1,
        1,1,1,1,1,0,0,0,0,0,1,1,1,1,1,
        1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1
    ],
    [
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,
        1,1,1,1,1,0,0,0,0,0,1,1,1,1,1,
        1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,
        1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,
        1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,
        1,1,1,1,1,0,0,0,0,0,1,1,1,1,1,
        1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1
    ],
    [
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,
        1,1,1,1,1,0,0,0,0,0,1,1,1,1,1,
        1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,
        1,1,1,0,0,0,0,0,0,0,0,0,1,1,1,
        1,1,1,0,0,0,0,0,0,0,0,0,1,1,1,
        1,1,1,0,0,0,0,0,0,0,0,0,1,1,1,
        1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,
        1,1,1,1,1,0,0,0,0,0,1,1,1,1,1,
        1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1
    ],
    [
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,
        1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,
        1,1,1,0,0,0,0,0,0,0,0,0,1,1,1,
        1,1,1,0,0,0,0,0,0,0,0,0,1,1,1,
        1,1,0,0,0,0,0,0,0,0,0,0,0,1,1,
        1,1,0,0,0,0,0,0,0,0,0,0,0,1,1,
        1,1,0,0,0,0,0,0,0,0,0,0,0,1,1,
        1,1,1,0,0,0,0,0,0,0,0,0,1,1,1,
        1,1,1,0,0,0,0,0,0,0,0,0,1,1,1,
        1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,
        1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1
    ],
    [
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,
        1,1,1,0,0,0,0,0,0,0,0,0,1,1,1,
        1,1,0,0,0,0,0,0,0,0,0,0,0,1,1,
        1,1,0,0,0,0,0,0,0,0,0,0,0,1,1,
        1,1,0,0,0,0,0,0,0,0,0,0,0,1,1,
        1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,
        1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,
        1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,
        1,1,0,0,0,0,0,0,0,0,0,0,0,1,1,
        1,1,0,0,0,0,0,0,0,0,0,0,0,1,1,
        1,1,0,0,0,0,0,0,0,0,0,0,0,1,1,
        1,1,1,0,0,0,0,0,0,0,0,0,1,1,1,
        1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1
    ]
];

pub(crate) fn get_brush_ranges(
    row_center: usize,
    column_center: usize,
    brush_id:usize,
    width: usize,
    height: usize
) -> (
    std::ops::RangeInclusive<usize>,
    std::ops::RangeInclusive<usize>,
    std::ops::RangeInclusive<usize>,
    std::ops::RangeInclusive<usize>
) {
    assert!(width > 0 && height > 0);
    let left = (column_center as i32 - DRAW_RADIUS_OFFSETS[brush_id] as i32).max(0) as usize;
    let right = (column_center + DRAW_RADIUS_OFFSETS[brush_id]).min(width-1) as usize;
    let top = (row_center as i32 - DRAW_RADIUS_OFFSETS[brush_id] as i32).max(0) as usize;
    let bottom = (row_center + DRAW_RADIUS_OFFSETS[brush_id]).min(height-1) as usize;

    let (horizontal_range_max, vertical_range_max) = (
        left..=right,
        top..=bottom
    );

    let local_left = CENTRAL_ID as i32 - (column_center as i32 - *horizontal_range_max.start() as i32);
    let local_right = CENTRAL_ID as i32 + *horizontal_range_max.end() as i32 - column_center as i32;
    let local_top = CENTRAL_ID as i32 - (row_center as i32 - *vertical_range_max.start() as i32);
    let local_bottom = CENTRAL_ID as i32 + *vertical_range_max.end() as i32 - row_center as i32;

    (
        horizontal_range_max,
        vertical_range_max,
        local_left as usize..=local_right as usize,
        local_top as usize..=local_bottom as usize
    )
}