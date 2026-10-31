use crate::{Action, BOARD_SIZE, Catalog, EPISODE_LENGTH, Environment, Features};
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
    fn get_room_cell_count(&self) -> usize {
        let mut count = 0;
        for room in &self.inner.rooms {
            count += room.size[0] * room.size[1];
        }
        count
    }
}

#[pyclass(name = "CandidateFeatures", frozen, get_all, module = "z3_map_gen")]
struct PyCandidateFeatures {
    actions: Vec<(usize, usize, usize)>,
    features: Vec<Features>,
}

#[pyclass(name = "Environment", module = "z3_map_gen")]
struct PyEnvironment {
    inner: Environment,
}

fn decode_action(action: Option<(usize, usize, usize)>) -> Action {
    match action {
        Some((room_id, x, y)) => Action::Place { room_id, x, y },
        None => Action::Dummy,
    }
}

#[pymethods]
impl PyEnvironment {
    #[new]
    fn new(catalog: &PyCatalog) -> Self {
        Self {
            inner: Environment::new(catalog.inner.clone()),
        }
    }

    fn list_candidates(&self) -> Vec<(usize, usize, usize)> {
        let mut actions = Vec::new();
        for action in self.inner.list_candidates() {
            if let Action::Place { room_id, x, y } = action {
                actions.push((room_id, x, y));
            }
        }
        actions
    }

    fn extract_features(&self) -> Features {
        self.inner.extract_features()
    }

    fn extract_candidate_features(&self) -> PyCandidateFeatures {
        let mut actions = Vec::new();
        let mut features = Vec::new();
        for (action, state_features) in self.inner.extract_candidate_features() {
            if let Action::Place { room_id, x, y } = action {
                actions.push((room_id, x, y));
                features.push(state_features);
            }
        }
        PyCandidateFeatures { actions, features }
    }

    #[pyo3(signature = (action))]
    fn apply_action(&mut self, action: Option<(usize, usize, usize)>) {
        self.inner.apply_action(decode_action(action));
    }

    #[staticmethod]
    fn replay(catalog: &PyCatalog, actions: Vec<Option<(usize, usize, usize)>>) -> PyResult<Self> {
        let mut decoded = Vec::new();
        for action in actions {
            decoded.push(decode_action(action));
        }
        let inner =
            Environment::replay(catalog.inner.clone(), &decoded).map_err(PyValueError::new_err)?;
        Ok(Self { inner })
    }
}

#[pymodule]
fn z3_map_gen(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyCatalog>()?;
    module.add_class::<PyEnvironment>()?;
    module.add_class::<Features>()?;
    module.add_class::<PyCandidateFeatures>()?;
    module.add("BOARD_SIZE", BOARD_SIZE)?;
    module.add("EPISODE_LENGTH", EPISODE_LENGTH)?;
    Ok(())
}
