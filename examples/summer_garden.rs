use {
    simple_tiled_wfc::grid_generation::{WfcModule, WfcContext, WfcContextBuilder},
    macroquad::prelude::{scene::{Node, RefMut}, *},
    macroquad::miniquad::{TextureParams, TextureFormat, TextureWrap},
    ron::de::from_reader,
    std::{ sync::mpsc::channel }
};

mod serialization {
    use serde::Deserialize;
    use std::collections::HashMap;

    #[derive(Copy, Clone, PartialEq, Debug, Deserialize)]
    pub enum TerrainType {
        Land,
        GrassSharp,
        GrassRound,
        Water
    }

    #[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, Deserialize)]
    pub enum TreeType {
        None,
        Pine,
        Oak,
        Bush
    }

    #[derive(Copy, Clone, PartialEq, Debug, Deserialize)]
    pub enum TileKind {
        Inner,
        Outer
    }

    #[derive(Copy, Clone, PartialEq, Debug, Deserialize)]
    pub enum NeighbourKind {
        WangCorners,
        RelOffset(i32)
    }

    #[derive(Copy, Clone, Debug, Deserialize)]
    pub struct NeighbourChooseStrategy {
        pub north: NeighbourKind,
        pub west: NeighbourKind,
        pub east: NeighbourKind,
        pub south: NeighbourKind,
    }

    #[derive(Copy, Clone, Debug, Deserialize)]
    pub struct TileSidesPattern {
        pub north_west: TileKind,
        pub north_east: TileKind,
        pub south_west: TileKind,
        pub south_east: TileKind,
    }

    #[derive(Copy, Clone, Debug, Deserialize)]
    pub struct TileSides {
        pub north_west: TerrainType,
        pub north_east: TerrainType,
        pub south_west: TerrainType,
        pub south_east: TerrainType,
    }

    #[derive(Copy, Clone, Deserialize)]
    pub struct TerrainTilesConfig {
        pub x_offset: i32,
        pub y_offset: i32,
        pub outer_type: TerrainType,
        pub inner_type: TerrainType,
    }

    #[derive(Copy, Clone, Debug, Deserialize)]
    pub struct SubRect {
        pub x: i32,
        pub y: i32,
        pub width: i32,
        pub height: i32
    }

    #[derive(Clone, Deserialize)]
    pub struct SummerGardenAtlas {
        pub tree_sub_rects: HashMap<TreeType, SubRect>,

        pub reduced_wang_patterns: Vec<TileSidesPattern>,
        pub extended_set_1_patterns_north_west: Vec<TileSidesPattern>,
        pub extended_set_1_patterns_north_east: Vec<TileSidesPattern>,
        pub extended_set_1_patterns_south_west: Vec<TileSidesPattern>,
        pub extended_set_1_patterns_south_east: Vec<TileSidesPattern>,
        pub extended_set_2_patterns_north_west: Vec<TileSidesPattern>,
        pub extended_set_2_patterns_north_east: Vec<TileSidesPattern>,
        pub extended_set_2_patterns_south_west: Vec<TileSidesPattern>,
        pub extended_set_2_patterns_south_east: Vec<TileSidesPattern>,

        pub vertical_bridge_sides: Vec<TileSides>,
        pub horizontal_bridge_sides: Vec<TileSides>,

        pub reduced_wang_neighbour_strategy: Vec<NeighbourChooseStrategy>,
        pub neighbour_strategy_2_x_2: Vec<NeighbourChooseStrategy>,
        pub neighbour_strategy_3_x_3: Vec<NeighbourChooseStrategy>,

        pub terrain_tile_configs: Vec<TerrainTilesConfig>
    }
}
use serialization::*;

//noinspection DuplicatedCode
fn draw_subrect(tex: Texture2D, subrect: &SubRect, x: f32, y: f32, scale: f32) {
    let InternalGlContext {
        quad_context: ctx, ..
    } = unsafe { get_internal_gl() };
    draw_texture_ex(
        tex,
        x * ctx.dpi_scale(), y * ctx.dpi_scale(),
        WHITE,
        DrawTextureParams {
            source: Some(Rect::new(
                subrect.x as f32, subrect.y as f32,
                subrect.width as f32, subrect.height as f32
            )),
            dest_size: Some([
                subrect.width as f32 * ctx.dpi_scale() * scale,
                subrect.height as f32 * ctx.dpi_scale() * scale
            ].into()),
            ..Default::default()
        },
    );
}

fn draw_subrect_pivoted(tex: Texture2D, subrect: &SubRect, x: f32, y: f32, scale: f32) {
    let InternalGlContext {
        quad_context: ctx, ..
    } = unsafe { get_internal_gl() };
    draw_texture_ex(
        tex,
        (x - subrect.width as f32 / (2.0 / scale)) * ctx.dpi_scale(),
        (y - subrect.height as f32 * scale) * ctx.dpi_scale(),
        WHITE,
        DrawTextureParams {
            source: Some(Rect::new(
                subrect.x as f32, subrect.y as f32,
                subrect.width as f32, subrect.height as f32
            )),
            dest_size: Some([
                subrect.width as f32 * ctx.dpi_scale() * scale,
                subrect.height as f32 * ctx.dpi_scale() * scale
            ].into()),
            ..Default::default()
        },
    );
}

fn window_conf() -> Conf {
    Conf {
        window_title: "Summer garden".to_owned(),
        fullscreen: false,
        window_width: 1280,
        window_height: 800,
        high_dpi: true,
        ..Default::default()
    }
}

type CustomBitSet = [u8; 32];

struct WangTilemap {
    w: usize,
    h: usize,
    atlas: SummerGardenAtlas,
    draw_scale: f32,
    tile_width: usize,
    tile_height: usize,
    tiles: Vec<SubRect>,
    modules: Vec<WfcModule<CustomBitSet>>,
    texture: Texture2D,
    map_data: Vec<usize>,
    corner_tree_data: Vec<TreeType>,
    cell_tree_data: Vec<TreeType>
}

impl WangTilemap {
    pub async fn new(w: usize, h: usize, draw_scale: f32) -> Self {
        let InternalGlContext {
            quad_context: ctx, ..
        } = unsafe { get_internal_gl() };

        let texture_bytes = load_file("assets/reduced_wang_scheme.png").await.unwrap();

        let img = image::load_from_memory(&texture_bytes[..])
            .unwrap_or_else(|e| panic!("{}", e))
            .to_rgba8();

        let img_w = img.width();
        let img_h = img.height();

        let texture = Texture2D::from_miniquad_texture(
            miniquad::Texture::from_data_and_format(
                ctx,
                &img.into_raw(),
                TextureParams {
                    format: TextureFormat::RGBA8,
                    wrap: TextureWrap::Clamp,
                    filter: FilterMode::Nearest,
                    width: img_w,
                    height: img_h
                }
            )
        );

        let atlas_bytes = load_file("assets/reduced_wang_scheme.ron").await.unwrap();
        let atlas: SummerGardenAtlas = from_reader(&atlas_bytes[..]).unwrap();

        let mut tile_sides = Vec::new();
        let mut neighbour_strategies = Vec::new();
        let mut tiles = Vec::new();

        {
            for tile_cfg in atlas.terrain_tile_configs.iter() {
                for pattern in &[
                    &atlas.reduced_wang_patterns[..],
                    &atlas.extended_set_1_patterns_north_west[..],
                    &atlas.extended_set_1_patterns_north_east[..],
                    &atlas.extended_set_1_patterns_south_west[..],
                    &atlas.extended_set_1_patterns_south_east[..],
                    &atlas.extended_set_2_patterns_north_west[..],
                    &atlas.extended_set_2_patterns_north_east[..],
                    &atlas.extended_set_2_patterns_south_west[..],
                    &atlas.extended_set_2_patterns_south_east[..],
                ] {
                    tile_sides.extend(
                        pattern.iter().map(
                            |pattern| {
                                TileSides {
                                    north_west: match pattern.north_west {
                                        TileKind::Inner => { tile_cfg.inner_type }
                                        TileKind::Outer => { tile_cfg.outer_type }
                                    },
                                    north_east: match pattern.north_east {
                                        TileKind::Inner => { tile_cfg.inner_type }
                                        TileKind::Outer => { tile_cfg.outer_type }
                                    },
                                    south_west: match pattern.south_west {
                                        TileKind::Inner => { tile_cfg.inner_type }
                                        TileKind::Outer => { tile_cfg.outer_type }
                                    },
                                    south_east: match pattern.south_east {
                                        TileKind::Inner => { tile_cfg.inner_type }
                                        TileKind::Outer => { tile_cfg.outer_type }
                                    }
                                }
                            }
                        )
                    );
                }
                for j in 0..4 {
                    for i in 0..4 {
                        tiles.push(SubRect{
                            x: i * 32 + tile_cfg.x_offset,
                            y: j * 32 + tile_cfg.y_offset,
                            width: 32,
                            height: 32
                        })
                    }
                }
                for jj in 0..2 {
                    for ii in 0..2 {
                        for j in 0..2 {
                            for i in 0..2 {
                                tiles.push(SubRect{
                                    x: (i + ii * 2) * 32 + tile_cfg.x_offset + 256,
                                    y: (j + jj * 2) * 32 + tile_cfg.y_offset,
                                    width: 32,
                                    height: 32
                                })
                            }
                        }
                    }
                }
                for jj in 0..2 {
                    for ii in 0..2 {
                        for j in 0..2 {
                            for i in 0..2 {
                                tiles.push(SubRect{
                                    x: (i + ii * 2) * 32 + tile_cfg.x_offset,
                                    y: (j + jj * 2) * 32 + tile_cfg.y_offset + 256,
                                    width: 32,
                                    height: 32
                                })
                            }
                        }
                    }
                }
                neighbour_strategies.extend_from_slice(&atlas.reduced_wang_neighbour_strategy[..]);
                for _ in 0..8 {
                    neighbour_strategies.extend_from_slice(&atlas.neighbour_strategy_2_x_2[..]);
                }
            }
            tile_sides.extend_from_slice(&atlas.vertical_bridge_sides[..]);
            tile_sides.extend_from_slice(&atlas.horizontal_bridge_sides[..]);
            for j in 0..3 {
                for i in 0..3 {
                    tiles.push(SubRect{
                        x: i * 32 + 256,
                        y: j * 32 + 256,
                        width: 32,
                        height: 32
                    })
                }
            }
            for j in 0..3 {
                for i in 0..3 {
                    tiles.push(SubRect{
                        x: i * 32 + 256 + 96,
                        y: j * 32 + 256,
                        width: 32,
                        height: 32
                    })
                }
            }
            neighbour_strategies.extend_from_slice(&atlas.neighbour_strategy_3_x_3[..]);
            neighbour_strategies.extend_from_slice(&atlas.neighbour_strategy_3_x_3[..]);

            assert_eq!(tile_sides.len(), neighbour_strategies.len());
        }

        let mut modules = Vec::new();
        for i in 0..tile_sides.len() {
            let current_sides = tile_sides[i];
            let mut module: WfcModule<CustomBitSet> = WfcModule::new();
            match neighbour_strategies[i].north {
                NeighbourKind::WangCorners => {
                    for j in 0..tile_sides.len() {
                        if neighbour_strategies[j].south != NeighbourKind::WangCorners {
                            continue;
                        }
                        if tile_sides[j].south_west == current_sides.north_west &&
                            tile_sides[j].south_east == current_sides.north_east {
                            module.add_north_neighbour(j);
                        }
                    }
                }
                NeighbourKind::RelOffset(offset) => {
                    let new_offset = (i as i32 + offset) as usize;
                    module.add_north_neighbour(new_offset);
                }
            }
            match neighbour_strategies[i].south {
                NeighbourKind::WangCorners => {
                    for j in 0..tile_sides.len() {
                        if neighbour_strategies[j].north != NeighbourKind::WangCorners {
                            continue;
                        }
                        if tile_sides[j].north_west == current_sides.south_west &&
                            tile_sides[j].north_east == current_sides.south_east {
                            module.add_south_neighbour(j);
                        }
                    }
                }
                NeighbourKind::RelOffset(offset) => {
                    let new_offset = (i as i32 + offset) as usize;
                    module.add_south_neighbour(new_offset);
                }
            }
            match neighbour_strategies[i].east {
                NeighbourKind::WangCorners => {
                    for j in 0..tile_sides.len() {
                        if neighbour_strategies[j].west != NeighbourKind::WangCorners {
                            continue;
                        }
                        if tile_sides[j].north_west == current_sides.north_east &&
                            tile_sides[j].south_west == current_sides.south_east {
                            module.add_east_neighbour(j);
                        }
                    }
                }
                NeighbourKind::RelOffset(offset) => {
                    let new_offset = (i as i32 + offset) as usize;
                    module.add_east_neighbour(new_offset);
                }
            }
            match neighbour_strategies[i].west {
                NeighbourKind::WangCorners => {
                    for j in 0..tile_sides.len() {
                        if neighbour_strategies[j].east != NeighbourKind::WangCorners {
                            continue;
                        }
                        if tile_sides[j].north_east == current_sides.north_west &&
                            tile_sides[j].south_east == current_sides.south_west {
                            module.add_west_neighbour(j);
                        }
                    }
                }
                NeighbourKind::RelOffset(offset) => {
                    let new_offset = (i as i32 + offset) as usize;
                    module.add_west_neighbour(new_offset);
                }
            }
            modules.push(module);
        }

        Self {
            w,
            h,
            draw_scale,
            atlas,
            tile_width: 32,
            tile_height: 32,
            tiles,
            modules,
            texture,
            map_data: vec![0; w*h],
            cell_tree_data: vec![TreeType::None; w*h],
            corner_tree_data: vec![TreeType::None; (w+1)*(h+1)]
        }
    }

    pub fn generate_new_map(&mut self) {
        let mut wfc_context: WfcContext<CustomBitSet> = WfcContextBuilder
        ::new(&self.modules, self.w, self.h)
            .build();

        let (tx, rc) = channel();

        wfc_context.collapse(100, tx.clone());

        let results = rc.recv()
            .unwrap()
            .unwrap_or_else(|_| vec![0; self.w * self.h]);

        self.map_data.clear();
        self.map_data.extend_from_slice(&results[..]);

        self.plant_trees()
    }

    fn plant_trees(&mut self) {
        self.cell_tree_data.fill(TreeType::None);
        self.corner_tree_data.fill(TreeType::None);

        for j in 1..self.h - 1 {
            for i in 1..self.w - 1 {
                let idx = j * self.w + i;
                if !(self.map_data[idx] >= 48 && self.map_data[idx] < 96) {
                    continue;
                }

                let lop_left_corner_idx = j * (self.w + 1) + i;

                self.try_plant_cell_tree(idx);

                self.try_plant_corner_tree(lop_left_corner_idx);

                if j == self.h - 2 {
                    self.try_plant_corner_tree(lop_left_corner_idx + self.w + 1);
                }
                if i == self.w - 2 && j == self.h - 2 {
                    self.try_plant_corner_tree(lop_left_corner_idx + self.w + 2);
                }
                if i == self.w - 2 {
                    self.try_plant_corner_tree(lop_left_corner_idx + 1);
                }
            }
        }
    }

    fn try_plant_corner_tree(&mut self, idx: usize) {
        if rand::gen_range(0, 100) < 4 {
            self.corner_tree_data[idx] = TreeType::Bush;
        } else if rand::gen_range(0, 100) < 12 {
            self.corner_tree_data[idx] = TreeType::Oak;
        } else if rand::gen_range(0, 100) < 18 {
            self.corner_tree_data[idx] = TreeType::Pine;
        }
    }

    fn try_plant_cell_tree(&mut self, idx: usize) {
        if rand::gen_range(0, 100) < 4 {
            self.cell_tree_data[idx] = TreeType::Bush;
        } else if rand::gen_range(0, 100) < 12 {
            self.cell_tree_data[idx] = TreeType::Oak;
        } else if rand::gen_range(0, 100) < 18 {
            self.cell_tree_data[idx] = TreeType::Pine;
        }
    }
}

impl Node for WangTilemap {
    fn update(mut node: RefMut<Self>) {
        if is_key_pressed(KeyCode::Space) {
            node.generate_new_map();
        }
    }

    fn draw(node: RefMut<Self>) {
        let InternalGlContext {
            quad_context: ctx, ..
        } = unsafe { get_internal_gl() };

        let start_x = screen_width() / ctx.dpi_scale();
        let start_x = (start_x - (node.w * node.tile_width) as f32 * node.draw_scale) / 2.0;

        let start_y = screen_height() / ctx.dpi_scale();
        let start_y = (start_y - (node.h * node.tile_height) as f32 * node.draw_scale) / 2.0;

        for j in 0..node.h {
            for i in 0..node.w {
                let idx = node.w * j + i;
                let x = start_x + (node.tile_width * i) as f32 * node.draw_scale;
                let y = start_y + (node.tile_height * j) as f32 * node.draw_scale;
                let idx = node.map_data[idx];
                draw_subrect(node.texture, &(node.tiles[idx]), x, y, node.draw_scale);
            }
        }
        for j in 0..node.h+1 {
            for i in 0..node.w+1 {
                let idx = (node.w + 1) * j + i;
                let x = start_x + (node.tile_width * i - 16) as f32 * node.draw_scale;
                let y = start_y + (node.tile_height * j - 12) as f32 * node.draw_scale;
                let cell_tree = node.corner_tree_data[idx];
                if let Some(subrect) = node.atlas.tree_sub_rects.get(&cell_tree) {
                    draw_subrect_pivoted(node.texture, subrect, x, y, node.draw_scale)
                }
            }

            if j < node.h {
                for i in 0..node.w {
                    let idx = node.w * j + i;
                    let x = start_x + (node.tile_width * i) as f32 * node.draw_scale;
                    let y = start_y + (node.tile_height * j + 4) as f32 * node.draw_scale;
                    let cell_tree = node.cell_tree_data[idx];
                    if let Some(subrect) = node.atlas.tree_sub_rects.get(&cell_tree) {
                        draw_subrect_pivoted(node.texture, subrect, x, y, node.draw_scale)
                    }
                }
            }
        }
    }
}

#[macroquad::main(window_conf)]
async fn main() {
    scene::add_node({
        let mut tilemap = WangTilemap::new(80, 50, 0.5).await;
        tilemap.generate_new_map();
        tilemap
    });
    loop {
        if is_key_pressed(KeyCode::Escape) {
            break;
        }
        clear_background(Color::new(0.12, 0.1, 0.15, 1.00));
        next_frame().await;
    }
}