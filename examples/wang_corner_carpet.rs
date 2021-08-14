// An example which shows that exhaustive tile sets aren't so much desired by WFC
use {
    simple_tiled_wfc::grid_generation::{WfcModule, WfcContext, WfcContextBuilder},
    macroquad::prelude::{scene::{Node, RefMut}, *},
    macroquad::miniquad::{TextureParams, TextureFormat, TextureWrap},
    std::{ sync::mpsc::channel }
};

pub const fn get_north_west(kind_code: u8) -> u8 { kind_code & 0b11 }
pub const fn get_north_east(kind_code: u8) -> u8 { (kind_code / 0b100) & 0b11 }
pub const fn get_south_east(kind_code: u8) -> u8 { (kind_code / 0b10000) & 0b11 }
pub const fn get_south_west(kind_code: u8) -> u8 { (kind_code / 0b1000000) & 0b11 }

pub struct SubRect {
    pub x: i32,
    pub y: i32,
    pub width: i32,
    pub height: i32
}

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

fn window_conf() -> Conf {
    Conf {
        window_title: "Wang corner carpet".to_owned(),
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
    tile_width: usize,
    tile_height: usize,
    tiles: Vec<(u8, SubRect)>,
    modules: Vec<WfcModule<CustomBitSet>>,
    texture: Texture2D,
    map_data: Vec<usize>
}

impl WangTilemap {
    pub async fn new(w: usize, h: usize) -> Self {
        let InternalGlContext {
            quad_context: ctx, ..
        } = unsafe { get_internal_gl() };

        let texture_bytes = load_file("assets/wang_4_corner.png").await.unwrap();

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

        let tiles = (0..=255)
            .map(|idx| {
                let width = 32;
                let height = 32;
                (
                    idx as u8,
                    SubRect {
                        x: (idx % 16) * width,
                        y: (idx / 16) * height,
                        width,
                        height
                    }
                )
            })
            .collect::<Vec<_>>();

        let modules = tiles
            .iter()
            .map(|tile| {
                let mut module: WfcModule<CustomBitSet> = WfcModule::new();
                for i in 0..tiles.len() {
                    let other_tile = &tiles[i];
                    if get_north_west(tile.0) == get_south_west(other_tile.0) &&
                        get_north_east(tile.0) == get_south_east(other_tile.0) {
                        module.add_north_neighbour(i);
                    }
                    if get_south_west(tile.0) == get_north_west(other_tile.0) &&
                        get_south_east(tile.0) == get_north_east(other_tile.0) {
                        module.add_south_neighbour(i);
                    }
                    if get_north_west(tile.0) == get_north_east(other_tile.0) &&
                        get_south_west(tile.0) == get_south_east(other_tile.0) {
                        module.add_west_neighbour(i);
                    }
                    if get_north_east(tile.0) == get_north_west(other_tile.0) &&
                        get_south_east(tile.0) == get_south_west(other_tile.0) {
                        module.add_east_neighbour(i);
                    }
                }
                module
            })
            .collect::<Vec<_>>();

        Self {
            w,
            h,
            tile_width: 32,
            tile_height: 32,
            tiles,
            modules,
            texture,
            map_data: vec![0; w*h]
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

        const SCALE_UP: f32 = 0.5;
        let start_x = screen_width() / ctx.dpi_scale();
        let start_x = (start_x - (node.w * node.tile_width) as f32 * SCALE_UP) / 2.0;

        let start_y = screen_height() / ctx.dpi_scale();
        let start_y = (start_y - (node.h * node.tile_height) as f32 * SCALE_UP) / 2.0;

        for j in 0..node.h {
            for i in 0..node.w {
                let idx = node.w * j + i;
                let x = start_x + (node.tile_width * i) as f32 * SCALE_UP;
                let y = start_y + (node.tile_height * j) as f32 * SCALE_UP;
                let idx = node.map_data[idx];
                draw_subrect(node.texture, &(node.tiles[idx].1), x, y, SCALE_UP);
            }
        }
    }
}

#[macroquad::main(window_conf)]
async fn main() {
    scene::add_node({
        let mut tilemap = WangTilemap::new(48, 48).await;
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