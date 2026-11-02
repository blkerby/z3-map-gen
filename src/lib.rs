use serde::Deserialize;
use std::sync::Arc;

mod python;

pub const BOARD_SIZE: i8 = 8;
pub const EPISODE_LENGTH: usize = 40;

pub const CELL_EMPTY: i8 = -2;
pub const CELL_FAILED: i8 = -1;

#[derive(Debug)]
pub struct Catalog {
    pub rooms: Vec<Room>,
    segment_matches: Vec<Vec<bool>>,
    segment_closed: Vec<bool>,
}

#[derive(Debug)]
pub struct Room {
    pub id: usize,
    pub size: [i8; 2],
    pub edges: Edges,
}

#[derive(Debug, Default)]
pub struct Edges {
    pub north: Vec<usize>,
    pub south: Vec<usize>,
    pub east: Vec<usize>,
    pub west: Vec<usize>,
}

impl Catalog {
    /// Index room segments and precompute compatibility once at startup.
    pub fn load_json(json: &str) -> Result<Self, serde_json::Error> {
        #[derive(Deserialize)]
        struct SourceCatalog {
            rooms: Vec<SourceRoom>,
        }
        #[derive(Deserialize)]
        struct SourceRoom {
            id: usize,
            size: [usize; 2],
            edges: SourceEdges,
        }
        #[derive(Deserialize)]
        struct SourceEdges {
            north: Vec<Vec<String>>,
            south: Vec<Vec<String>>,
            east: Vec<Vec<String>>,
            west: Vec<Vec<String>>,
        }

        let source: SourceCatalog = serde_json::from_str(json)?;
        let mut rooms = Vec::new();
        let mut segments = Vec::new();
        for room in source.rooms {
            let mut edges = Edges::default();
            for (source_segments, indices) in [
                (room.edges.north, &mut edges.north),
                (room.edges.south, &mut edges.south),
                (room.edges.east, &mut edges.east),
                (room.edges.west, &mut edges.west),
            ] {
                for variants in source_segments {
                    indices.push(segments.len());
                    segments.push(variants);
                }
            }
            rooms.push(Room {
                id: room.id,
                size: [room.size[0] as i8, room.size[1] as i8],
                edges,
            });
        }
        let mut segment_matches = vec![vec![false; segments.len()]; segments.len()];
        let mut segment_closed = Vec::new();
        for (i, variants) in segments.iter().enumerate() {
            segment_closed.push(variants.iter().any(|variant| variant == "closed"));
            for (j, other) in segments.iter().enumerate() {
                segment_matches[i][j] = variants.iter().any(|variant| other.contains(variant));
            }
        }
        Ok(Self {
            rooms,
            segment_matches,
            segment_closed,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Action {
    Place { room_index: i8, x: i8, y: i8 },
    Dummy,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Features {
    pub room_cells: Vec<i8>,
    pub room_positions: Vec<i8>,
}

#[derive(Clone, Debug)]
pub struct Environment {
    catalog: Arc<Catalog>,
    cells: [i8; (BOARD_SIZE * BOARD_SIZE) as usize],
    cell_order: [usize; (BOARD_SIZE * BOARD_SIZE) as usize],
    positions: Vec<Option<[i8; 2]>>,
}

enum Direction {
    North,
    South,
    East,
    West,
}

impl Environment {
    pub fn new(
        catalog: Arc<Catalog>,
        cell_order: [usize; (BOARD_SIZE * BOARD_SIZE) as usize],
    ) -> Self {
        let positions = vec![None; catalog.rooms.len()];
        Self {
            catalog,
            cells: [CELL_EMPTY; (BOARD_SIZE * BOARD_SIZE) as usize],
            cell_order,
            positions,
        }
    }

    /// Occupying room indices in catalog order, with cells in row-major order.
    pub fn get_cells(&self) -> &[i8; (BOARD_SIZE * BOARD_SIZE) as usize] {
        &self.cells
    }

    /// Room anchors in catalog order.
    pub fn get_positions(&self) -> &[Option<[i8; 2]>] {
        &self.positions
    }

    pub fn extract_features(&self) -> Features {
        let mut features = Features {
            // Inherit CELL_EMPTY and CELL_FAILED values. Others will be overwritten.
            room_cells: self.cells.to_vec(),
            room_positions: vec![-1; self.catalog.rooms.len()],
        };
        let mut cell_offset: i8 = 0;
        for (room_index, room) in self.catalog.rooms.iter().enumerate() {
            let [width, height] = room.size;
            if let Some([x, y]) = self.positions[room_index] {
                features.room_positions[room_index] = y * BOARD_SIZE + x;
                for dy in 0..height {
                    for dx in 0..width {
                        let i = (y + dy) * BOARD_SIZE + x + dx;
                        features.room_cells[i as usize] = cell_offset + dy * width + dx;
                    }
                }
            }
            cell_offset += width * height;
        }
        features
    }

    pub fn extract_candidate_features(&mut self) -> Vec<(Action, Features)> {
        let mut candidates = Vec::new();
        for action in self.list_candidates() {
            let mut speculative = self.clone();
            speculative.apply_action(action);
            candidates.push((action, speculative.extract_features()));
        }
        candidates
    }

    /// Check geometric validity at any anchor, independently of scan order.
    pub fn can_place(&self, room_index: usize, x: i8, y: i8) -> bool {
        let room = &self.catalog.rooms[room_index];
        let [width, height] = room.size;
        if self.positions[room_index].is_some()
            || x >= BOARD_SIZE
            || y >= BOARD_SIZE
            || width > BOARD_SIZE - x
            || height > BOARD_SIZE - y
        {
            return false;
        }
        for dy in 0..height {
            for dx in 0..width {
                let i = ((y + dy) * BOARD_SIZE + x + dx) as usize;
                if self.cells[i] >= 0 || self.cells[i] == CELL_FAILED {
                    return false;
                }
            }
        }
        for dx in 0..width {
            if !self.check_contact(
                room.edges.north[dx as usize],
                x + dx,
                y - 1,
                Direction::South,
            ) || !self.check_contact(
                room.edges.south[dx as usize],
                x + dx,
                y + height,
                Direction::North,
            ) {
                return false;
            }
        }
        for dy in 0..height {
            if !self.check_contact(room.edges.west[dy as usize], x - 1, y + dy, Direction::East)
                || !self.check_contact(
                    room.edges.east[dy as usize],
                    x + width,
                    y + dy,
                    Direction::West,
                )
            {
                return false;
            }
        }
        true
    }

    fn check_contact(&self, segment: usize, x: i8, y: i8, side: Direction) -> bool {
        if !(0..BOARD_SIZE).contains(&x) || !(0..BOARD_SIZE).contains(&y) {
            return self.catalog.segment_closed[segment];
        }
        let i = y * BOARD_SIZE + x;
        let room_index = self.cells[i as usize];
        if room_index < 0 {
            return true;
        };
        let [anchor_x, anchor_y] = self.positions[room_index as usize].unwrap();
        let edges = &self.catalog.rooms[room_index as usize].edges;
        let neighbor_segment = match side {
            Direction::North => edges.north[(x - anchor_x) as usize],
            Direction::South => edges.south[(x - anchor_x) as usize],
            Direction::East => edges.east[(y - anchor_y) as usize],
            Direction::West => edges.west[(y - anchor_y) as usize],
        };
        self.catalog.segment_matches[segment][neighbor_segment]
    }

    /// Legal placements covering the next empty cell; an empty result means exhaustion.
    pub fn list_candidates(&mut self) -> Vec<Action> {
        for &i in &self.cell_order {
            if self.cells[i] != CELL_EMPTY {
                continue;
            }
            let x = (i % BOARD_SIZE as usize) as i8;
            let y = (i / BOARD_SIZE as usize) as i8;
            let mut candidates = Vec::new();
            for room_index in 0..self.catalog.rooms.len() {
                let [width, height] = self.catalog.rooms[room_index].size;
                for anchor_y in (y - height + 1).max(0)..=y {
                    for anchor_x in (x - width + 1).max(0)..=x {
                        if self.can_place(room_index, anchor_x, anchor_y) {
                            candidates.push(Action::Place {
                                room_index: room_index as i8,
                                x: anchor_x,
                                y: anchor_y,
                            });
                        }
                    }
                }
            }
            if candidates.is_empty() {
                self.cells[i] = CELL_FAILED;
            } else {
                return candidates;
            }
        }
        Vec::new()
    }

    /// Apply an action trusted to be valid for this state; dummy padding is a no-op.
    pub fn apply_action(&mut self, action: Action) {
        let Action::Place { room_index, x, y } = action else {
            return;
        };
        let [width, height] = self.catalog.rooms[room_index as usize].size;
        self.positions[room_index as usize] = Some([x, y]);
        for dy in 0..height {
            for dx in 0..width {
                let i = (y + dy) * BOARD_SIZE + x + dx;
                self.cells[i as usize] = room_index;
            }
        }
    }

    /// Replay trusted real actions or an episode including its dummy tail.
    pub fn replay(catalog: Arc<Catalog>, actions: &[Action]) -> Result<Self, &'static str> {
        if actions.len() > EPISODE_LENGTH {
            return Err("episode exceeds 40 action slots");
        }
        let mut environment = Self::new(catalog, std::array::from_fn(|i| i));
        for &action in actions {
            environment.apply_action(action);
        }
        Ok(environment)
    }
}
