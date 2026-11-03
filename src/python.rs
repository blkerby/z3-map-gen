use crate::{Action, BOARD_SIZE, Catalog, EPISODE_LENGTH, Environment};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyByteArray, PyDict, PyList};
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

    #[getter(room_sizes)]
    fn get_room_sizes(&self) -> Vec<[i8; 2]> {
        self.inner.rooms.iter().map(|room| room.size).collect()
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
    features: Py<PyFeatures>,
}

#[pyclass(
    name = "Features",
    frozen,
    get_all,
    skip_from_py_object,
    module = "z3_map_gen"
)]
struct PyFeatures {
    room_cells: Py<PyAny>,
    room_positions: Py<PyAny>,
}

fn feature_tensor(py: Python<'_>, values: &[i8], shape: (usize, usize)) -> PyResult<Py<PyAny>> {
    let torch = py.import("torch")?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("dtype", torch.getattr("long")?)?;
    if values.is_empty() {
        kwargs.set_item("device", "cpu")?;
        return Ok(torch
            .call_method("empty", (shape,), Some(&kwargs))?
            .unbind());
    }
    // The tensor retains this writable Python buffer after Rust returns.
    let buffer = PyByteArray::new_with(py, values.len() * size_of::<i64>(), |bytes| {
        for (chunk, &value) in bytes.chunks_exact_mut(size_of::<i64>()).zip(values) {
            chunk.copy_from_slice(&i64::from(value).to_ne_bytes());
        }
        Ok(())
    })?;
    Ok(torch
        .call_method("frombuffer", (buffer,), Some(&kwargs))?
        .call_method1("reshape", (shape,))?
        .unbind())
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
    fn new(py: Python<'_>, catalog: &PyCatalog, environment_count: usize) -> PyResult<Self> {
        let sample = py.import("random")?.getattr("sample")?;
        let cells = PyList::new(py, 0..(BOARD_SIZE * BOARD_SIZE) as usize)?;
        let mut environments = Vec::with_capacity(environment_count);
        for _ in 0..environment_count {
            let cell_order = sample.call1((&cells, cells.len()))?.extract()?;
            environments.push(Environment::new(catalog.inner.clone(), cell_order));
        }
        Ok(Self { environments })
    }

    fn extract_features(&self, py: Python<'_>) -> PyResult<PyFeatures> {
        let mut room_cells = Vec::new();
        let mut room_positions = Vec::new();
        let room_count = self
            .environments
            .first()
            .map_or(0, |e| e.get_positions().len());
        for environment in &self.environments {
            let features = environment.extract_features();
            room_cells.extend(features.room_cells);
            room_positions.extend(features.room_positions);
        }
        let rows = self.environments.len();
        Ok(PyFeatures {
            room_cells: feature_tensor(
                py,
                &room_cells,
                (rows, (BOARD_SIZE * BOARD_SIZE) as usize),
            )?,
            room_positions: feature_tensor(py, &room_positions, (rows, room_count))?,
        })
    }

    fn extract_candidate_features(&mut self, py: Python<'_>) -> PyResult<PyCandidateFeatures> {
        let mut actions = Vec::new();
        let mut environment_offsets = vec![0];
        let mut room_cells = Vec::new();
        let mut room_positions = Vec::new();
        let room_count = self
            .environments
            .first()
            .map_or(0, |e| e.get_positions().len());
        let candidates = py.detach(|| {
            let workers = std::thread::available_parallelism().map_or(1, |n| n.get());
            let chunk_size = self.environments.len().div_ceil(workers).max(1);
            std::thread::scope(|scope| {
                let handles: Vec<_> = self
                    .environments
                    .chunks_mut(chunk_size)
                    .map(|chunk| {
                        scope.spawn(|| {
                            chunk
                                .iter_mut()
                                .map(Environment::extract_candidate_features)
                                .collect::<Vec<_>>()
                        })
                    })
                    .collect();
                handles
                    .into_iter()
                    .flat_map(|handle| handle.join().unwrap())
                    .collect::<Vec<_>>()
            })
        });
        for candidates in candidates {
            for (action, features) in candidates {
                if let Action::Place { room_index, x, y } = action {
                    actions.push((room_index, x, y));
                    room_cells.extend(features.room_cells);
                    room_positions.extend(features.room_positions);
                }
            }
            environment_offsets.push(actions.len());
        }
        let rows = actions.len();
        Ok(PyCandidateFeatures {
            actions,
            environment_offsets,
            features: Py::new(
                py,
                PyFeatures {
                    room_cells: feature_tensor(
                        py,
                        &room_cells,
                        (rows, (BOARD_SIZE * BOARD_SIZE) as usize),
                    )?,
                    room_positions: feature_tensor(py, &room_positions, (rows, room_count))?,
                },
            )?,
        })
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
