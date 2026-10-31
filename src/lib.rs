use serde::Deserialize;
use std::sync::Arc;

mod python;

pub const BOARD_SIZE: usize = 8;
pub const EPISODE_LENGTH: usize = 40;

#[derive(Debug)]
pub struct Catalog {
    pub rooms: Vec<Room>,
    segment_matches: Vec<Vec<bool>>,
    segment_closed: Vec<bool>,
}

#[derive(Debug)]
pub struct Room {
    pub id: usize,
    pub size: [usize; 2],
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
                size: room.size,
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
    Place { room_id: usize, x: usize, y: usize },
    Dummy,
}

#[pyo3::pyclass(frozen, get_all, skip_from_py_object, module = "z3_map_gen")]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Features {
    pub room_cells: Vec<i64>,
    pub room_positions: Vec<i64>,
}

#[derive(Clone, Debug)]
pub struct Environment {
    catalog: Arc<Catalog>,
    cells: [Option<usize>; BOARD_SIZE * BOARD_SIZE],
    positions: Vec<Option<[usize; 2]>>,
}

enum Direction {
    North,
    South,
    East,
    West,
}

impl Environment {
    pub fn new(catalog: Arc<Catalog>) -> Self {
        let positions = vec![None; catalog.rooms.len()];
        Self {
            catalog,
            cells: [None; BOARD_SIZE * BOARD_SIZE],
            positions,
        }
    }

    /// Occupying room indices in catalog order, with cells in row-major order.
    pub fn get_cells(&self) -> &[Option<usize>; BOARD_SIZE * BOARD_SIZE] {
        &self.cells
    }

    /// Room anchors in catalog order.
    pub fn get_positions(&self) -> &[Option<[usize; 2]>] {
        &self.positions
    }

    pub fn extract_features(&self) -> Features {
        let mut features = Features {
            room_cells: vec![-1; BOARD_SIZE * BOARD_SIZE],
            room_positions: vec![-1; self.catalog.rooms.len()],
        };
        let mut cell_offset = 0;
        for (room_index, room) in self.catalog.rooms.iter().enumerate() {
            let [width, height] = room.size;
            if let Some([x, y]) = self.positions[room_index] {
                features.room_positions[room_index] = (y * BOARD_SIZE + x) as i64;
                for dy in 0..height {
                    for dx in 0..width {
                        features.room_cells[(y + dy) * BOARD_SIZE + x + dx] =
                            (cell_offset + dy * width + dx) as i64;
                    }
                }
            }
            cell_offset += width * height;
        }
        features
    }

    pub fn extract_candidate_features(&self) -> Vec<(Action, Features)> {
        let mut candidates = Vec::new();
        for action in self.list_candidates() {
            if let Action::Place { room_id, x, y } = action {
                let mut speculative = self.clone();
                speculative.place_room(room_id, x, y);
                candidates.push((action, speculative.extract_features()));
            }
        }
        candidates
    }

    /// Check geometric validity at any anchor, independently of scan order.
    pub fn can_place(&self, room_index: usize, x: usize, y: usize) -> bool {
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
                if self.cells[(y + dy) * BOARD_SIZE + x + dx].is_some() {
                    return false;
                }
            }
        }
        for dx in 0..width {
            if !self.check_contact(
                room.edges.north[dx],
                (x + dx) as isize,
                y as isize - 1,
                Direction::South,
            ) || !self.check_contact(
                room.edges.south[dx],
                (x + dx) as isize,
                (y + height) as isize,
                Direction::North,
            ) {
                return false;
            }
        }
        for dy in 0..height {
            if !self.check_contact(
                room.edges.west[dy],
                x as isize - 1,
                (y + dy) as isize,
                Direction::East,
            ) || !self.check_contact(
                room.edges.east[dy],
                (x + width) as isize,
                (y + dy) as isize,
                Direction::West,
            ) {
                return false;
            }
        }
        true
    }

    fn check_contact(&self, segment: usize, x: isize, y: isize, side: Direction) -> bool {
        if !(0..BOARD_SIZE as isize).contains(&x) || !(0..BOARD_SIZE as isize).contains(&y) {
            return self.catalog.segment_closed[segment];
        }
        let x = x as usize;
        let y = y as usize;
        let Some(room_index) = self.cells[y * BOARD_SIZE + x] else {
            return true;
        };
        let [anchor_x, anchor_y] = self.positions[room_index].unwrap();
        let edges = &self.catalog.rooms[room_index].edges;
        let neighbor_segment = match side {
            Direction::North => edges.north[x - anchor_x],
            Direction::South => edges.south[x - anchor_x],
            Direction::East => edges.east[y - anchor_y],
            Direction::West => edges.west[y - anchor_y],
        };
        self.catalog.segment_matches[segment][neighbor_segment]
    }

    /// An empty result means exhaustion; dummy padding is never a candidate.
    pub fn list_candidates(&self) -> Vec<Action> {
        for y in 0..BOARD_SIZE {
            for x in 0..BOARD_SIZE {
                if self.cells[y * BOARD_SIZE + x].is_some() {
                    continue;
                }
                let mut candidates = Vec::new();
                for (room_index, room) in self.catalog.rooms.iter().enumerate() {
                    if self.can_place(room_index, x, y) {
                        candidates.push(Action::Place {
                            room_id: room.id,
                            x,
                            y,
                        });
                    }
                }
                if !candidates.is_empty() {
                    return candidates;
                }
            }
        }
        Vec::new()
    }

    /// Apply a generation action, rejecting invalid replay data before mutation.
    pub fn apply_action(&mut self, action: Action) -> Result<(), &'static str> {
        let candidates = self.list_candidates();
        let Action::Place { room_id, x, y } = action else {
            return if candidates.is_empty() {
                Ok(())
            } else {
                Err("dummy action before exhaustion")
            };
        };
        if !candidates.contains(&action) {
            return Err("placement is not a current candidate");
        }
        self.place_room(room_id, x, y);
        Ok(())
    }

    fn place_room(&mut self, room_id: usize, x: usize, y: usize) {
        let room_index = self
            .catalog
            .rooms
            .iter()
            .position(|room| room.id == room_id)
            .unwrap();
        let [width, height] = self.catalog.rooms[room_index].size;
        self.positions[room_index] = Some([x, y]);
        for dy in 0..height {
            for dx in 0..width {
                self.cells[(y + dy) * BOARD_SIZE + x + dx] = Some(room_index);
            }
        }
    }

    /// Replay a real-action prefix or an episode including its dummy tail.
    pub fn replay(catalog: Arc<Catalog>, actions: &[Action]) -> Result<Self, &'static str> {
        if actions.len() > EPISODE_LENGTH {
            return Err("episode exceeds 40 action slots");
        }
        let mut environment = Self::new(catalog);
        for &action in actions {
            environment.apply_action(action)?;
        }
        Ok(environment)
    }
}
