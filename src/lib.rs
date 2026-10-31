use serde::Deserialize;
use std::sync::Arc;

pub const BOARD_SIZE: usize = 8;
pub const EPISODE_LENGTH: usize = 40;

#[derive(Debug, Deserialize)]
pub struct Catalog {
    pub rooms: Vec<Room>,
}

#[derive(Debug, Deserialize)]
pub struct Room {
    pub id: usize,
    pub size: [usize; 2],
    pub edges: Edges,
}

#[derive(Debug, Deserialize)]
pub struct Edges {
    pub north: Vec<Vec<String>>,
    pub south: Vec<Vec<String>>,
    pub east: Vec<Vec<String>>,
    pub west: Vec<Vec<String>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Action {
    Place { room_id: usize, x: usize, y: usize },
    Dummy,
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
                &room.edges.north[dx],
                (x + dx) as isize,
                y as isize - 1,
                Direction::South,
            ) || !self.check_contact(
                &room.edges.south[dx],
                (x + dx) as isize,
                (y + height) as isize,
                Direction::North,
            ) {
                return false;
            }
        }
        for dy in 0..height {
            if !self.check_contact(
                &room.edges.west[dy],
                x as isize - 1,
                (y + dy) as isize,
                Direction::East,
            ) || !self.check_contact(
                &room.edges.east[dy],
                (x + width) as isize,
                (y + dy) as isize,
                Direction::West,
            ) {
                return false;
            }
        }
        true
    }

    fn check_contact(&self, variants: &[String], x: isize, y: isize, side: Direction) -> bool {
        if !(0..BOARD_SIZE as isize).contains(&x) || !(0..BOARD_SIZE as isize).contains(&y) {
            return variants.iter().any(|variant| variant == "closed");
        }
        let x = x as usize;
        let y = y as usize;
        let Some(room_index) = self.cells[y * BOARD_SIZE + x] else {
            return true;
        };
        let [anchor_x, anchor_y] = self.positions[room_index].unwrap();
        let edges = &self.catalog.rooms[room_index].edges;
        let neighbor_variants = match side {
            Direction::North => &edges.north[x - anchor_x],
            Direction::South => &edges.south[x - anchor_x],
            Direction::East => &edges.east[y - anchor_y],
            Direction::West => &edges.west[y - anchor_y],
        };
        variants
            .iter()
            .any(|variant| neighbor_variants.contains(variant))
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
        Ok(())
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
