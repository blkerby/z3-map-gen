use crate::{Action, BOARD_SIZE, Catalog, EPISODE_LENGTH, Environment};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::path::PathBuf;
use std::sync::Arc;

#[pyclass(name = "Catalog", frozen, module = "z3_map_gen")]
struct PyCatalog {
    inner: Arc<Catalog>,
}

#[pymethods]
impl PyCatalog {
    #[new]
    fn new(path: PathBuf) -> PyResult<Self> {
        let json = std::fs::read_to_string(path)?;
        let catalog =
            Catalog::load_json(&json).map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok(Self {
            inner: Arc::new(catalog),
        })
    }

    #[getter(room_ids)]
    fn get_room_ids(&self) -> Vec<usize> {
        let mut ids = Vec::new();
        for room in &self.inner.rooms {
            ids.push(room.id);
        }
        ids
    }

    #[getter(room_cell_count)]
    fn get_room_cell_count(&self) -> i8 {
        let mut count = 0;
        for room in &self.inner.rooms {
            count += room.size[0] * room.size[1];
        }
        count
    }
}

#[pyclass(name = "CandidateFeatures", frozen, get_all, module = "z3_map_gen")]
struct PyCandidateFeatures {
    actions: Vec<(i8, i8, i8)>,
    environment_offsets: Vec<usize>,
    features: PyFeatures,
}

#[pyclass(
    name = "Features",
    frozen,
    get_all,
    skip_from_py_object,
    module = "z3_map_gen"
)]
#[derive(Clone)]
struct PyFeatures {
    room_cells: Vec<Vec<i8>>,
    room_positions: Vec<Vec<i8>>,
}

#[pyclass(name = "EnvironmentGroup", module = "z3_map_gen")]
struct PyEnvironmentGroup {
    environments: Vec<Environment>,
}

fn decode_action(action: Option<(i8, i8, i8)>) -> Action {
    match action {
        Some((room_index, x, y)) => Action::Place { room_index, x, y },
        None => Action::Dummy,
    }
}

#[pymethods]
impl PyEnvironmentGroup {
    #[new]
    fn new(catalog: &PyCatalog, environment_count: usize) -> Self {
        Self {
            environments: vec![Environment::new(catalog.inner.clone()); environment_count],
        }
    }

    fn extract_features(&self) -> PyFeatures {
        let mut room_cells = Vec::new();
        let mut room_positions = Vec::new();
        for environment in &self.environments {
            let features = environment.extract_features();
            room_cells.push(features.room_cells);
            room_positions.push(features.room_positions);
        }
        PyFeatures {
            room_cells,
            room_positions,
        }
    }

    fn extract_candidate_features(&mut self) -> PyCandidateFeatures {
        let mut actions = Vec::new();
        let mut environment_offsets = vec![0];
        let mut room_cells = Vec::new();
        let mut room_positions = Vec::new();
        for environment in &mut self.environments {
            for (action, features) in environment.extract_candidate_features() {
                if let Action::Place { room_index, x, y } = action {
                    actions.push((room_index, x, y));
                    room_cells.push(features.room_cells);
                    room_positions.push(features.room_positions);
                }
            }
            environment_offsets.push(actions.len());
        }
        PyCandidateFeatures {
            actions,
            environment_offsets,
            features: PyFeatures {
                room_cells,
                room_positions,
            },
        }
    }

    fn apply_actions(&mut self, actions: Vec<Option<(i8, i8, i8)>>) -> PyResult<()> {
        if actions.len() != self.environments.len() {
            return Err(PyValueError::new_err("expected one action per environment"));
        }
        for (environment, action) in self.environments.iter_mut().zip(actions) {
            environment.apply_action(decode_action(action));
        }
        Ok(())
    }

    #[staticmethod]
    fn replay(catalog: &PyCatalog, episodes: Vec<Vec<Option<(i8, i8, i8)>>>) -> PyResult<Self> {
        let mut environments = Vec::new();
        for actions in episodes {
            let mut decoded = Vec::new();
            for action in actions {
                decoded.push(decode_action(action));
            }
            environments.push(
                Environment::replay(catalog.inner.clone(), &decoded)
                    .map_err(PyValueError::new_err)?,
            );
        }
        Ok(Self { environments })
    }
}

#[pymodule]
fn z3_map_gen(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyCatalog>()?;
    module.add_class::<PyEnvironmentGroup>()?;
    module.add_class::<PyFeatures>()?;
    module.add_class::<PyCandidateFeatures>()?;
    module.add("BOARD_SIZE", BOARD_SIZE)?;
    module.add("EPISODE_LENGTH", EPISODE_LENGTH)?;
    Ok(())
}
