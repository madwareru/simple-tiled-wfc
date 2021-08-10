use {
    simple_tiled_wfc::grid_generation::{WfcModule, WfcContext, WfcContextBuilder},
    macroquad::prelude::{
        scene::{Node, RefMut},
        *
    },
    macroquad::miniquad::{TextureParams, TextureFormat, TextureWrap},
    ron::de::from_reader,
    std::{
        collections::{VecDeque},
        cmp::Ordering,
        sync::mpsc::channel
    }
};
mod serialization {
    use serde::Deserialize;
    use std::collections::HashMap;

    #[derive(Debug, Deserialize, Clone, Copy)]
    pub struct SubRect {
        pub x: i32,
        pub y: i32,
        pub width: i32,
        pub height: i32
    }

    #[derive(Debug, Deserialize, PartialEq, Eq, Clone, Copy)]
    pub enum TileKind {
        Wang4Corner(u8), // It is named as Wang4Corner,
                         // but you don't have to provide all
                         // four types of corners as well as
                         // exhaustive sets are not required
                         // (while still possible)
        VerticalBridgeGroundVoid0x0,
        VerticalBridgeGroundVoid0x1,
        VerticalBridgeGroundVoid0x2,
        VerticalBridgeGroundVoid1x0,
        VerticalBridgeGroundVoid1x1,
        VerticalBridgeGroundVoid1x2,
        VerticalBridgeGroundVoid2x0,
        VerticalBridgeGroundVoid2x1,
        VerticalBridgeGroundVoid2x2,
        VerticalBridgeWaterGround0x0,
        VerticalBridgeWaterGround0x1,
        VerticalBridgeWaterGround0x2,
        VerticalBridgeWaterGround1x0,
        VerticalBridgeWaterGround1x1,
        VerticalBridgeWaterGround1x2,
        VerticalBridgeWaterGround2x0,
        VerticalBridgeWaterGround2x1,
        VerticalBridgeWaterGround2x2,
        HorizontalBridgeGroundVoid0x0,
        HorizontalBridgeGroundVoid0x1,
        HorizontalBridgeGroundVoid0x2,
        HorizontalBridgeGroundVoid1x0,
        HorizontalBridgeGroundVoid1x1,
        HorizontalBridgeGroundVoid1x2,
        HorizontalBridgeGroundVoid2x0,
        HorizontalBridgeGroundVoid2x1,
        HorizontalBridgeGroundVoid2x2,
        HorizontalBridgeWaterGround0x0,
        HorizontalBridgeWaterGround0x1,
        HorizontalBridgeWaterGround0x2,
        HorizontalBridgeWaterGround1x0,
        HorizontalBridgeWaterGround1x1,
        HorizontalBridgeWaterGround1x2,
        HorizontalBridgeWaterGround2x0,
        HorizontalBridgeWaterGround2x1,
        HorizontalBridgeWaterGround2x2,
    }

    #[derive(Debug, Deserialize, Clone)]
    pub struct Tile {
        pub kind: TileKind,
        pub subrects: Vec<SubRect>,
        pub neighbours_east: Vec<TileKind>,
        pub neighbours_west: Vec<TileKind>,
        pub neighbours_north: Vec<TileKind>,
        pub neighbours_south: Vec<TileKind>
    }

    #[derive(Debug, Deserialize)]
    pub struct DungeonTiles {
        pub tile_width: usize,
        pub tile_height: usize,
        pub wang_4_corner_tiles: HashMap<u8, Vec<SubRect>>,
        pub extra_tiles: Vec<Tile>
    }
}
use serialization::*;

pub const fn get_north_east(kind_code: u8) -> u8 { kind_code & 0b11 }
pub const fn get_north_west(kind_code: u8) -> u8 { (kind_code / 0b100) & 0b11 }
pub const fn get_south_east(kind_code: u8) -> u8 { (kind_code / 0b10000) & 0b11 }
pub const fn get_south_west(kind_code: u8) -> u8 { (kind_code / 0b1000000) & 0b11 }

fn draw_subrect(tex: Texture2D, subrect: &SubRect, x: f32, y: f32, scale: usize) {
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
                subrect.width as f32 * ctx.dpi_scale() * scale as f32,
                subrect.height as f32 * ctx.dpi_scale() * scale as f32
            ].into()),
            ..Default::default()
        },
    );
}

fn window_conf() -> Conf {
    Conf {
        window_title: "Dungeon".to_owned(),
        fullscreen: false,
        window_width: 1280,
        window_height: 800,
        high_dpi: true,
        ..Default::default()
    }
}

type CustomBitSet = [u8; 8];

struct DungeonTilemap {
    w: usize,
    h: usize,
    tile_width: usize,
    tile_height: usize,
    tiles: Vec<Tile>,
    modules: Vec<WfcModule<CustomBitSet>>,
    texture: Texture2D,
    map_data: Vec<(usize, usize)>
}

impl DungeonTilemap {
    pub async fn new(w: usize, h: usize) -> Self {
        let InternalGlContext {
            quad_context: ctx, ..
        } = unsafe { get_internal_gl() };

        let dungeon_bytes = load_file("assets/dungeon_tiles.png").await.unwrap();

        let img = image::load_from_memory(&dungeon_bytes[..])
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

        let tiles_bytes = load_file("assets/dungeon_tiles.ron").await.unwrap();
        let dungeon_tiles: DungeonTiles = from_reader(&tiles_bytes[..]).unwrap();

        let mut tiles = Vec::new();
        for (kind_code, subrects) in dungeon_tiles.wang_4_corner_tiles.iter() {
            tiles.push(Tile {
                kind: TileKind::Wang4Corner(*kind_code),
                subrects: subrects.clone(),
                neighbours_east: Vec::new(),
                neighbours_west: Vec::new(),
                neighbours_north: Vec::new(),
                neighbours_south: Vec::new()
            });
        }
        // We need to be sure that zero tile is always a "void" tile, so we doing an extra sort step
        tiles.sort_by(|lhs, rhs| {
            match (lhs.kind, rhs.kind) {
                (TileKind::Wang4Corner(kind_lhs), TileKind::Wang4Corner(kind_rhs)) => kind_lhs.cmp(&kind_rhs),
                _ => Ordering::Equal
            }
        });

        for i in 0..tiles.len() {
            let current_kind = tiles[i].kind;
            for j in 0..tiles.len() {
                let candidate_kind = tiles[j].kind;
                let (matches_north, matches_south, matches_east, matches_west) = {
                    match (current_kind, candidate_kind) {
                        (TileKind::Wang4Corner(current), TileKind::Wang4Corner(candidate)) => {
                            (
                                get_north_east(current) == get_south_east(candidate) &&
                                    get_north_west(current) == get_south_west(candidate),

                                get_south_east(current) == get_north_east(candidate) &&
                                    get_south_west(current) == get_north_west(candidate),

                                get_north_east(current) == get_north_west(candidate) &&
                                    get_south_east(current) == get_south_west(candidate),

                                get_north_west(current) == get_north_east(candidate) &&
                                    get_south_west(current) == get_south_east(candidate),
                            )
                        },
                        _ => continue
                    }
                };
                if matches_east {
                    tiles[i].neighbours_east.push(candidate_kind);
                }
                if matches_west {
                    tiles[i].neighbours_west.push(candidate_kind);
                }
                if matches_north {
                    tiles[i].neighbours_north.push(candidate_kind);
                }
                if matches_south {
                    tiles[i].neighbours_south.push(candidate_kind);
                }
            }
        }

        //Add bridges:
        {
            let bridge_tiles_offset = tiles.len();
            tiles.extend_from_slice(&dungeon_tiles.extra_tiles[..]);

            let mut bridge_match_queue_south = VecDeque::new();
            let mut bridge_match_queue_north = VecDeque::new();
            let mut bridge_match_queue_east = VecDeque::new();
            let mut bridge_match_queue_west = VecDeque::new();

            for bridge_tile in &tiles[bridge_tiles_offset..] {
                for south_neighbour in bridge_tile.neighbours_south.iter() {
                    if let TileKind::Wang4Corner(kind) = south_neighbour {
                        for i in 0..bridge_tiles_offset {
                            match tiles[i].kind {
                                TileKind::Wang4Corner(kind_inner) if kind_inner == *kind => {
                                    bridge_match_queue_south.push_back((bridge_tile.kind, i))
                                },
                                _ => ()
                            };
                        }
                    }
                }
                for north_neighbour in bridge_tile.neighbours_north.iter() {
                    if let TileKind::Wang4Corner(kind) = north_neighbour {
                        for i in 0..bridge_tiles_offset {
                            match tiles[i].kind {
                                TileKind::Wang4Corner(kind_inner) if kind_inner == *kind => {
                                    bridge_match_queue_north.push_back((bridge_tile.kind, i))
                                },
                                _ => ()
                            };
                        }
                    }
                }
                for east_neighbour in bridge_tile.neighbours_east.iter() {
                    if let TileKind::Wang4Corner(kind) = east_neighbour {
                        for i in 0..bridge_tiles_offset {
                            match tiles[i].kind {
                                TileKind::Wang4Corner(kind_inner) if kind_inner == *kind => {
                                    bridge_match_queue_east.push_back((bridge_tile.kind, i))
                                },
                                _ => ()
                            };
                        }
                    }
                }
                for west_neighbour in bridge_tile.neighbours_west.iter() {
                    if let TileKind::Wang4Corner(kind) = west_neighbour {
                        for i in 0..bridge_tiles_offset {
                            match tiles[i].kind {
                                TileKind::Wang4Corner(kind_inner) if kind_inner == *kind => {
                                    bridge_match_queue_west.push_back((bridge_tile.kind, i))
                                },
                                _ => ()
                            };
                        }
                    }
                }
            }

            while !bridge_match_queue_south.is_empty() {
                let next_match = bridge_match_queue_south.pop_front().unwrap();
                tiles[next_match.1].neighbours_north.push(next_match.0);
            }

            while !bridge_match_queue_north.is_empty() {
                let next_match = bridge_match_queue_north.pop_front().unwrap();
                tiles[next_match.1].neighbours_south.push(next_match.0);
            }

            while !bridge_match_queue_east.is_empty() {
                let next_match = bridge_match_queue_east.pop_front().unwrap();
                tiles[next_match.1].neighbours_west.push(next_match.0);
            }

            while !bridge_match_queue_west.is_empty() {
                let next_match = bridge_match_queue_west.pop_front().unwrap();
                tiles[next_match.1].neighbours_east.push(next_match.0);
            }
        }

        let modules = tiles
            .iter()
            .map(|tile| {
                let mut module: WfcModule<CustomBitSet> = WfcModule::new();
                for i in 0..tiles.len() {
                    if tile.neighbours_north.contains(&tiles[i].kind) {
                        module.add_north_neighbour(i);
                    }
                    if tile.neighbours_south.contains(&tiles[i].kind) {
                        module.add_south_neighbour(i);
                    }
                    if tile.neighbours_west.contains(&tiles[i].kind) {
                        module.add_west_neighbour(i);
                    }
                    if tile.neighbours_east.contains(&tiles[i].kind) {
                        module.add_east_neighbour(i);
                    }
                }
                module
            })
            .collect::<Vec<_>>();
        
        Self {
            w,
            h,
            tile_width: dungeon_tiles.tile_width,
            tile_height: dungeon_tiles.tile_height,
            tiles,
            modules,
            texture,
            map_data: vec![(0, 0); w*h]
        }
    }

    pub fn generate_new_map(&mut self) {
        let mut wfc_context: WfcContext<CustomBitSet> = WfcContextBuilder
            ::new(&self.modules, self.w, self.h)
            .build();

        let (tx, rc) = channel();

        // Preset some fields in a map with a void to enforce more island-ish look
        {
            wfc_context.set_module(0, 0, 0);
            wfc_context.set_module(self.h - 1, 0, 0);
            wfc_context.set_module(0, self.w - 1, 0);
            wfc_context.set_module(self.h - 1, self.w - 1, 0);
            wfc_context.set_module(self.h / 2, self.w / 2, 0);
        }

        wfc_context.collapse(100, tx.clone());

        let results = rc.recv()
            .unwrap()
            .unwrap_or_else(|_| vec![0; self.w * self.h]);

        self.map_data.clear();
        for &idx in results.iter() {
            let random_id = rand::gen_range(0, self.tiles[idx].subrects.len());
            self.map_data.push((idx, random_id));
        }
    }
}

impl Node for DungeonTilemap {
    fn update(mut node: RefMut<Self>) {
        if is_key_pressed(KeyCode::Space) {
            node.generate_new_map();
        }
    }

    fn draw(node: RefMut<Self>) {
        let InternalGlContext {
            quad_context: ctx, ..
        } = unsafe { get_internal_gl() };

        const SCALE_UP: usize = 2;
        let start_x = screen_width() / ctx.dpi_scale();
        let start_x = (start_x - (node.w * node.tile_width * SCALE_UP) as f32) / 2.0;

        let start_y = screen_height() / ctx.dpi_scale();
        let start_y = (start_y - (node.h * node.tile_height * SCALE_UP) as f32) / 2.0;

        for j in 0..node.h {
            for i in 0..node.w {
                let idx = node.w * j + i;
                let x = start_x + (node.tile_width * i * SCALE_UP) as f32;
                let y = start_y + (node.tile_height * j * SCALE_UP) as f32;
                let (idx, random_id) = node.map_data[idx];
                draw_subrect(node.texture, &node.tiles[idx].subrects[random_id], x, y, SCALE_UP);
            }
        }
    }
}

#[macroquad::main(window_conf)]
async fn main() {
    scene::add_node({
        let mut tilemap = DungeonTilemap::new(48, 48).await;
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