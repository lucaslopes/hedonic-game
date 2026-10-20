//! The CPython module `hedonic_core_rs`: the functions of hedonic._core
//! (native/binding/) with the same arguments, results, errors and messages,
//! over the Rust engines. `HEDONIC_CORE=rust` makes the package load it as
//! hedonic._core.
//!
//! The GIL is held throughout, as in the C binding: interruption checks and,
//! for a generator that is not an exact random.Random, every random number
//! are Python calls. Such a generator can run any Python code during a call,
//! so its calls copy their input buffers instead of borrowing them.

mod args;

use std::borrow::Cow;

use pyo3::exceptions::{PyOverflowError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::audit::{AuditOptions, AuditState as CoreAudit, Selection};
use crate::cover::{cover as run_cover_core, Cover};
use crate::graph::{Edges, Graph, Options};
use crate::partition::partition as run_partition_core;
use crate::queue::Callbacks;
use crate::trace::Trace;
use crate::verify::{verify_cover as check_cover, verify_partition as check_partition, VerifyReport};
use args::*;

type Any<'py> = Bound<'py, PyAny>;

/// The buffers of partition()/cover(): `make_graph` of arguments.c.
struct PlainGraph {
    n: i64,
    edges: Buffer,
    edge_weights: Option<Buffer>,
    node_weights: Option<Buffer>,
}

impl PlainGraph {
    fn new(n: isize, edges: &Any, edge_weights: Option<&Any>, node_weights: Option<&Any>) -> PyResult<PlainGraph> {
        if n < 0 {
            return Err(PyValueError::new_err("n must be non-negative"));
        }
        let buffer = Buffer::get(edges, Kind8::Int, None, "edges")?;
        let m = buffer.len() / 2;
        if buffer.len() != 2 * m {
            return Err(PyValueError::new_err(format!("edges has {} entries, expected {}", buffer.len(), 2 * m)));
        }
        Ok(PlainGraph {
            n: n as i64,
            edges: buffer,
            edge_weights: optional_doubles(edge_weights, m, "edge_weights")?,
            node_weights: optional_doubles(node_weights, n as usize, "node_weights")?,
        })
    }
}

/// The slices of a call: borrowed, or copied when `own`.
struct Slices<'a> {
    edges: Option<Cow<'a, [i64]>>,
    edge_weights: Option<Cow<'a, [f64]>>,
    node_weights: Option<Cow<'a, [f64]>>,
}

impl<'a> Slices<'a> {
    fn plain(g: &'a PlainGraph, own: bool) -> Slices<'a> {
        Slices {
            edges: Some(g.edges.ints_owned(own)),
            edge_weights: g.edge_weights.as_ref().map(|b| b.doubles_owned(own)),
            node_weights: g.node_weights.as_ref().map(|b| b.doubles_owned(own)),
        }
    }

    fn weights(edge_weights: &'a Option<Buffer>, node_weights: &'a Option<Buffer>, own: bool) -> Slices<'a> {
        Slices {
            edges: None,
            edge_weights: edge_weights.as_ref().map(|b| b.doubles_owned(own)),
            node_weights: node_weights.as_ref().map(|b| b.doubles_owned(own)),
        }
    }

    fn graph<'b>(&'b self, py: Python<'_>, n: i64, edges: Edges<'b>) -> PyResult<Graph<'b>> {
        Graph::new(n, edges, self.edge_weights.as_deref(), self.node_weights.as_deref()).map_err(|e| to_py(py, e, None))
    }

    fn plain_graph(&self, py: Python<'_>, n: i64) -> PyResult<Graph<'_>> {
        self.graph(py, n, Edges::Flat(self.edges.as_deref().unwrap_or(&[])))
    }
}

/// Runs `body` with the core's callbacks over the caller's generator; a
/// failure becomes the Python exception (the generator's own first).
fn with_rng<'py, T>(
    py: Python<'py>,
    rng: &mut PyRng<'py>,
    body: impl FnOnce(&mut Callbacks) -> crate::error::Result<T>,
) -> PyResult<T> {
    let mut check = check_signals;
    let result = {
        let mut cb = Callbacks::new(rng);
        cb.interrupt = Some(&mut check);
        body(&mut cb)
    };
    result.map_err(|e| {
        let rng_error = rng.take_error();
        to_py(py, e, rng_error)
    })
}

fn trace_record<'py>(py: Python<'py>, trace: &Trace, raw_slack: f64) -> PyResult<Bound<'py, PyDict>> {
    let start = PyList::empty(py);
    for row in &trace.start {
        start.append(PyList::new(py, row)?)?;
    }
    let moves = PyList::empty(py);
    for m in &trace.moves {
        moves.append((m.v, PyList::new(py, &m.row)?, m.improvement, m.bound))?;
    }
    let record = PyDict::new(py);
    record.set_item("start", start)?;
    record.set_item("moves", moves)?;
    record.set_item("raw_slack", raw_slack)?;
    Ok(record)
}

// ---- partition() and cover() --------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn plain_partition<'py>(
    py: Python<'py>,
    n: isize,
    edges: &Any<'py>,
    rng: &Any<'py>,
    edge_weights: Option<&Any<'py>>,
    node_weights: Option<&Any<'py>>,
    membership: Option<&Any<'py>>,
    resolution: f64,
    max_total_communities: Option<&Any<'py>>,
    n_communities: Option<&Any<'py>>,
    allow_isolation: Option<&Any<'py>>,
    traced: bool,
) -> PyResult<Py<PyAny>> {
    let allow_isolation = flag(allow_isolation, true)?;
    let options = Options {
        resolution,
        max_memberships: 1,
        max_total_communities: optional_limit(max_total_communities, "max_total_communities")?,
        n_communities: optional_limit(n_communities, "n_communities")?,
        allow_isolation,
    };
    let plain = PlainGraph::new(n, edges, edge_weights, node_weights)?;
    let mut rng = PyRng::new(rng)?;
    let start: Option<Vec<i64>> = match membership {
        Some(m) if !m.is_none() => {
            let read = fast(m, "membership must be a sequence").and_then(|items| {
                if items.len() as i64 != plain.n {
                    return Err(PyValueError::new_err(
                        "Membership vector length does not equal the number of vertices.",
                    ));
                }
                items.iter().map(|x| x.extract::<i64>()).collect::<PyResult<Vec<i64>>>()
            });
            match read {
                Ok(v) => Some(v),
                Err(e) => return rng.release(Err(e)),
            }
        }
        _ => None,
    };
    let slices = Slices::plain(&plain, rng.calls_python());
    let graph = slices.plain_graph(py, plain.n)?;
    let mut trace = Trace::default();
    let result = with_rng(py, &mut rng, |cb| {
        run_partition_core(&graph, &options, start.as_deref(), cb, traced.then_some(&mut trace))
    });
    let result = rng.release(result)?;
    let list = PyList::new(py, &result.membership)?;
    if traced {
        let record = trace_record(py, &trace, result.raw_slack)?;
        Ok((list, result.count, result.quality, record).into_pyobject(py)?.into_any().unbind())
    } else {
        Ok((list, result.count, result.quality).into_pyobject(py)?.into_any().unbind())
    }
}

/// partition(n, edges, rng, edge_weights=None, node_weights=None, membership=None, resolution=1.0, max_total_communities=None, n_communities=None, allow_isolation=True) -> (membership, nb_clusters, quality)
///
/// Local moving of a partition: the CPM hedonic game. `edges` is a flat buffer of
/// 64-bit integers (u0, v0, u1, v1, ...); weights are buffers of doubles or None;
/// `rng` is an object with getrandbits(). A given membership is the start.
#[pyfunction]
#[pyo3(signature = (n, edges, rng, edge_weights=None, node_weights=None, membership=None, resolution=1.0, max_total_communities=None, n_communities=None, allow_isolation=None))]
#[allow(clippy::too_many_arguments)]
fn partition<'py>(
    py: Python<'py>,
    n: isize,
    edges: &Any<'py>,
    rng: &Any<'py>,
    edge_weights: Option<&Any<'py>>,
    node_weights: Option<&Any<'py>>,
    membership: Option<&Any<'py>>,
    resolution: f64,
    max_total_communities: Option<&Any<'py>>,
    n_communities: Option<&Any<'py>>,
    allow_isolation: Option<&Any<'py>>,
) -> PyResult<Py<PyAny>> {
    plain_partition(py, n, edges, rng, edge_weights, node_weights, membership, resolution, max_total_communities, n_communities, allow_isolation, false)
}

/// partition() that also returns {'start', 'moves', 'raw_slack'}: the start rows in
/// the core's internal labels, (v, new row, improvement, bound) for every move, and
/// the slack before division by |W|. For tests.
#[pyfunction]
#[pyo3(name = "_trace_partition", signature = (n, edges, rng, edge_weights=None, node_weights=None, membership=None, resolution=1.0, max_total_communities=None, n_communities=None, allow_isolation=None))]
#[allow(clippy::too_many_arguments)]
fn trace_partition<'py>(
    py: Python<'py>,
    n: isize,
    edges: &Any<'py>,
    rng: &Any<'py>,
    edge_weights: Option<&Any<'py>>,
    node_weights: Option<&Any<'py>>,
    membership: Option<&Any<'py>>,
    resolution: f64,
    max_total_communities: Option<&Any<'py>>,
    n_communities: Option<&Any<'py>>,
    allow_isolation: Option<&Any<'py>>,
) -> PyResult<Py<PyAny>> {
    plain_partition(py, n, edges, rng, edge_weights, node_weights, membership, resolution, max_total_communities, n_communities, allow_isolation, true)
}

/// The rows of plain cover() data as CSR: one sequence per vertex (the core
/// checks them).
fn read_rows(obj: &Any, message: &str) -> PyResult<Cover> {
    let seq = fast(obj, message)?;
    let mut offset = Vec::with_capacity(seq.len() + 1);
    let mut labels = Vec::new();
    offset.push(0i64);
    for row in &seq {
        let items = fast(row, "every membership row must be a sequence")
            .map_err(|_| PyTypeError::new_err("every membership row must be a sequence"))?;
        for item in &items {
            labels.push(item.extract::<i64>()?);
        }
        offset.push(labels.len() as i64);
    }
    Ok(Cover { offset, labels })
}

#[allow(clippy::too_many_arguments)]
fn plain_cover<'py>(
    py: Python<'py>,
    n: isize,
    edges: &Any<'py>,
    rng: &Any<'py>,
    edge_weights: Option<&Any<'py>>,
    node_weights: Option<&Any<'py>>,
    memberships: Option<&Any<'py>>,
    max_memberships: i64,
    resolution: f64,
    max_total_communities: Option<&Any<'py>>,
    n_communities: Option<&Any<'py>>,
    allow_isolation: Option<&Any<'py>>,
    traced: bool,
) -> PyResult<Py<PyAny>> {
    let allow_isolation = flag(allow_isolation, true)?;
    let options = Options {
        resolution,
        max_memberships,
        max_total_communities: optional_limit(max_total_communities, "max_total_communities")?,
        n_communities: optional_limit(n_communities, "n_communities")?,
        allow_isolation,
    };
    let plain = PlainGraph::new(n, edges, edge_weights, node_weights)?;
    let mut rng = PyRng::new(rng)?;
    let start = match memberships {
        Some(m) if !m.is_none() => match read_rows(m, "memberships must be a sequence") {
            Ok(c) => Some(c),
            Err(e) => return rng.release(Err(e)),
        },
        _ => None,
    };
    let slices = Slices::plain(&plain, rng.calls_python());
    let graph = slices.plain_graph(py, plain.n)?;
    let mut trace = Trace::default();
    let result = with_rng(py, &mut rng, |cb| {
        run_cover_core(&graph, &options, start.as_ref(), cb, traced.then_some(&mut trace))
    });
    let result = rng.release(result)?;
    let rows = PyList::empty(py);
    for v in 0..result.cover.vertex_count() as usize {
        rows.append(PyList::new(py, result.cover.row(v))?)?;
    }
    if traced {
        let record = trace_record(py, &trace, result.raw_slack)?;
        Ok((rows, result.count, result.quality, record).into_pyobject(py)?.into_any().unbind())
    } else {
        Ok((rows, result.count, result.quality).into_pyobject(py)?.into_any().unbind())
    }
}

/// cover(n, edges, rng, edge_weights=None, node_weights=None, memberships=None, max_memberships=2, resolution=1.0, max_total_communities=None, n_communities=None, allow_isolation=True) -> (memberships, nb_clusters, quality)
///
/// Local moving of a cover: the unit-l2 CPM hedonic game, at most max_memberships
/// labels per vertex. Given memberships (one row per vertex) are the start.
#[pyfunction]
#[pyo3(signature = (n, edges, rng, edge_weights=None, node_weights=None, memberships=None, max_memberships=2, resolution=1.0, max_total_communities=None, n_communities=None, allow_isolation=None))]
#[allow(clippy::too_many_arguments)]
fn cover<'py>(
    py: Python<'py>,
    n: isize,
    edges: &Any<'py>,
    rng: &Any<'py>,
    edge_weights: Option<&Any<'py>>,
    node_weights: Option<&Any<'py>>,
    memberships: Option<&Any<'py>>,
    max_memberships: i64,
    resolution: f64,
    max_total_communities: Option<&Any<'py>>,
    n_communities: Option<&Any<'py>>,
    allow_isolation: Option<&Any<'py>>,
) -> PyResult<Py<PyAny>> {
    plain_cover(py, n, edges, rng, edge_weights, node_weights, memberships, max_memberships, resolution, max_total_communities, n_communities, allow_isolation, false)
}

/// cover() with the record of _trace_partition(). For tests.
#[pyfunction]
#[pyo3(name = "_trace_cover", signature = (n, edges, rng, edge_weights=None, node_weights=None, memberships=None, max_memberships=2, resolution=1.0, max_total_communities=None, n_communities=None, allow_isolation=None))]
#[allow(clippy::too_many_arguments)]
fn trace_cover<'py>(
    py: Python<'py>,
    n: isize,
    edges: &Any<'py>,
    rng: &Any<'py>,
    edge_weights: Option<&Any<'py>>,
    node_weights: Option<&Any<'py>>,
    memberships: Option<&Any<'py>>,
    max_memberships: i64,
    resolution: f64,
    max_total_communities: Option<&Any<'py>>,
    n_communities: Option<&Any<'py>>,
    allow_isolation: Option<&Any<'py>>,
) -> PyResult<Py<PyAny>> {
    plain_cover(py, n, edges, rng, edge_weights, node_weights, memberships, max_memberships, resolution, max_total_communities, n_communities, allow_isolation, true)
}

// ---- detect() ----------------------------------------------------------------------

/// The rows of a cover and its clusters (one list of vertices per label,
/// vertices increasing); each label and vertex one int object shared by both.
fn cover_lists<'py>(py: Python<'py>, cover: &Cover, count: i64) -> PyResult<(Bound<'py, PyList>, Bound<'py, PyList>)> {
    let labels: Vec<Any<'py>> = (0..count).map(|c| c.into_pyobject(py).unwrap().into_any()).collect();
    let mut members: Vec<Vec<Any<'py>>> = (0..count).map(|_| Vec::new()).collect();
    let rows = PyList::empty(py);
    for v in 0..cover.vertex_count() as usize {
        let vertex = (v as i64).into_pyobject(py)?.into_any();
        let row = cover.row(v);
        rows.append(PyList::new(py, row.iter().map(|&c| &labels[c as usize]))?)?;
        for &c in row {
            members[c as usize].push(vertex.clone());
        }
    }
    let clusters = PyList::empty(py);
    for m in &members {
        clusters.append(PyList::new(py, m)?)?;
    }
    Ok((rows, clusters))
}

/// detect(graph, rng, resolution, *, max_memberships=1, start=None, allow_isolation=False, edge_weights=None, node_weights=None, max_total_communities=None, n_communities=None)
///
/// Best-response local moving on an undirected igraph.Graph, run until no vertex
/// can improve. Returns (membership, nb_clusters, quality, equilibrium_slack)
/// for max_memberships=1, else (rows, clusters, nb_clusters, quality,
/// equilibrium_slack).
/// `start` is flat labels or per-vertex rows, checked as hedonic._native documents;
/// weights are buffers of doubles or None.
#[pyfunction]
#[pyo3(signature = (graph, rng, resolution, *, max_memberships=1, start=None, allow_isolation=None, edge_weights=None, node_weights=None, max_total_communities=None, n_communities=None))]
#[allow(clippy::too_many_arguments)]
fn detect<'py>(
    py: Python<'py>,
    graph: &Any<'py>,
    rng: &Any<'py>,
    resolution: f64,
    max_memberships: i64,
    start: Option<&Any<'py>>,
    allow_isolation: Option<&Any<'py>>,
    edge_weights: Option<&Any<'py>>,
    node_weights: Option<&Any<'py>>,
    max_total_communities: Option<&Any<'py>>,
    n_communities: Option<&Any<'py>>,
) -> PyResult<Py<PyAny>> {
    let allow_isolation = flag(allow_isolation, false)?;
    if max_memberships < 1 {
        return Err(PyValueError::new_err("max_memberships must be an integer >= 1, or -1"));
    }
    let options = Options {
        resolution,
        max_memberships,
        max_total_communities: optional_limit(max_total_communities, "max_total_communities")?,
        n_communities: optional_limit(n_communities, "n_communities")?,
        allow_isolation,
    };
    let mut edges = graph_edges(graph)?;
    let (n, m) = (edges.n, edges.m);
    let ew = optional_doubles(edge_weights, m as usize, "edge_weights")?;
    let nw = optional_doubles(node_weights, n as usize, "node_weights")?;
    let start = match start {
        Some(s) if !s.is_none() => read_start_rows(s, n, max_memberships)?,
        _ => None,
    };
    let mut rng = PyRng::new(rng)?;
    let own = rng.calls_python();
    if own {
        edges.detach();
    }
    let slices = Slices::weights(&ew, &nw, own);
    let g = slices.graph(py, n, edges.edges())?;
    if max_memberships == 1 {
        let membership: Option<Vec<i64>> =
            start.map(|(offset, labels)| offset[..n as usize].iter().map(|&o| labels[o as usize]).collect());
        let result = with_rng(py, &mut rng, |cb| run_partition_core(&g, &options, membership.as_deref(), cb, None));
        let result = rng.release(result)?;
        let list = PyList::new(py, &result.membership)?;
        Ok((list, result.count, result.quality, result.slack).into_pyobject(py)?.into_any().unbind())
    } else {
        let start = start.map(|(offset, labels)| Cover { offset, labels });
        let result = with_rng(py, &mut rng, |cb| run_cover_core(&g, &options, start.as_ref(), cb, None));
        let result = rng.release(result)?;
        let (rows, clusters) = cover_lists(py, &result.cover, result.count)?;
        Ok((rows, clusters, result.count, result.quality, result.slack).into_pyobject(py)?.into_any().unbind())
    }
}

// ---- The verifier (tests) ----------------------------------------------------------

fn report_dict<'py>(py: Python<'py>, r: &VerifyReport) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("vertices", r.vertices)?;
    d.set_item("improvable", r.improvable)?;
    d.set_item("max_certified", r.max_certified)?;
    d.set_item("max_bound", r.max_bound)?;
    Ok(d)
}

/// `read_ints` of checking.c: `count` integers; `what` names the sequence.
fn read_ints(obj: &Any, count: usize, what: &str) -> PyResult<Vec<i64>> {
    let items = fast(obj, what)?;
    if items.len() != count {
        return Err(PyValueError::new_err(format!("{what} has the wrong length")));
    }
    items.iter().map(|x| x.extract::<i64>()).collect()
}

/// _verify_partition(n, edges, membership, edge_weights=None, node_weights=None,
/// resolution=1.0, max_total_communities=None, n_communities=None,
/// allow_isolation=True) -> {'vertices', 'improvable', 'max_certified', 'max_bound'}
///
/// The independent verifier (native/core/check/check.c): every deviation of every vertex
/// scored afresh with certified bounds. For tests.
#[pyfunction]
#[pyo3(name = "_verify_partition", signature = (n, edges, membership, edge_weights=None, node_weights=None, resolution=1.0, max_total_communities=None, n_communities=None, allow_isolation=None))]
#[allow(clippy::too_many_arguments)]
fn verify_partition<'py>(
    py: Python<'py>,
    n: isize,
    edges: &Any<'py>,
    membership: &Any<'py>,
    edge_weights: Option<&Any<'py>>,
    node_weights: Option<&Any<'py>>,
    resolution: f64,
    max_total_communities: Option<&Any<'py>>,
    n_communities: Option<&Any<'py>>,
    allow_isolation: Option<&Any<'py>>,
) -> PyResult<Bound<'py, PyDict>> {
    let options = Options {
        resolution,
        max_memberships: 1,
        max_total_communities: optional_limit(max_total_communities, "max_total_communities")?,
        n_communities: optional_limit(n_communities, "n_communities")?,
        allow_isolation: flag(allow_isolation, true)?,
    };
    let plain = PlainGraph::new(n, edges, edge_weights, node_weights)?;
    let membership = read_ints(membership, plain.n as usize, "membership")?;
    let slices = Slices::plain(&plain, false);
    let graph = slices.plain_graph(py, plain.n)?;
    let report = check_partition(&graph, &options, &membership).map_err(|e| to_py(py, e, None))?;
    report_dict(py, &report)
}

/// _verify_cover(n, edges, memberships, max_memberships, ...): the same for a cover
/// (sorted rows of distinct labels). For tests.
#[pyfunction]
#[pyo3(name = "_verify_cover", signature = (n, edges, memberships, max_memberships, edge_weights=None, node_weights=None, resolution=1.0, max_total_communities=None, n_communities=None, allow_isolation=None))]
#[allow(clippy::too_many_arguments)]
fn verify_cover<'py>(
    py: Python<'py>,
    n: isize,
    edges: &Any<'py>,
    memberships: &Any<'py>,
    max_memberships: i64,
    edge_weights: Option<&Any<'py>>,
    node_weights: Option<&Any<'py>>,
    resolution: f64,
    max_total_communities: Option<&Any<'py>>,
    n_communities: Option<&Any<'py>>,
    allow_isolation: Option<&Any<'py>>,
) -> PyResult<Bound<'py, PyDict>> {
    let options = Options {
        resolution,
        max_memberships,
        max_total_communities: optional_limit(max_total_communities, "max_total_communities")?,
        n_communities: optional_limit(n_communities, "n_communities")?,
        allow_isolation: flag(allow_isolation, true)?,
    };
    let plain = PlainGraph::new(n, edges, edge_weights, node_weights)?;
    let seq = fast(memberships, "memberships must be a sequence")?;
    if seq.len() as i64 != plain.n {
        return Err(PyValueError::new_err("memberships needs one row per vertex"));
    }
    let mut offset = vec![0i64];
    let mut labels = Vec::new();
    for row in &seq {
        let size = row.len()?;
        labels.extend(read_ints(row, size, "a membership row")?);
        offset.push(labels.len() as i64);
    }
    let slices = Slices::plain(&plain, false);
    let graph = slices.plain_graph(py, plain.n)?;
    let report = check_cover(&graph, &options, &Cover { offset, labels }).map_err(|e| to_py(py, e, None))?;
    report_dict(py, &report)
}

// ---- The fixed-cover stability audit ----------------------------------------------

/// The prepared state of a fixed cover (immutable; shared by the queries).
#[cfg_attr(not(feature = "core-module"), pyclass(frozen, name = "AuditState", module = "hedonic_core_rs"))]
#[cfg_attr(feature = "core-module", pyclass(frozen, name = "AuditState", module = "hedonic._core"))]
struct AuditState {
    core: CoreAudit,
}

/// A variable-length buffer (`audit_buffer` of audit.c).
fn audit_buffer(obj: &Any, kind: Kind8, name: &str) -> PyResult<Buffer> {
    Buffer::get(obj, kind, None, name)
}

/// Fixed-cover audit prepare (private binding).
#[pyfunction]
#[pyo3(name = "_audit_prepare", signature = (graph, offsets, labels, edge_weights=None, node_weights=None))]
fn audit_prepare<'py>(
    py: Python<'py>,
    graph: &Any<'py>,
    offsets: &Any<'py>,
    labels: &Any<'py>,
    edge_weights: Option<&Any<'py>>,
    node_weights: Option<&Any<'py>>,
) -> PyResult<AuditState> {
    let edges = graph_edges(graph)?;
    let (n, m) = (edges.n, edges.m);
    if n == isize::MAX as i64 {
        return Err(PyOverflowError::new_err("Too many audit vertices"));
    }
    let offsets = Buffer::get(offsets, Kind8::Int, Some(n as usize + 1), "offsets")?;
    let labels = audit_buffer(labels, Kind8::Int, "labels")?;
    let (offset, label) = (offsets.ints(), labels.ints());
    if offset.iter().any(|&x| x < 0 || x as usize > label.len()) {
        return Err(PyValueError::new_err("Audit row offset outside labels buffer"));
    }
    let ew = optional_doubles(edge_weights, m as usize, "edge_weights")?;
    let nw = optional_doubles(node_weights, n as usize, "node_weights")?;
    let slices = Slices::weights(&ew, &nw, false);
    let g = slices.graph(py, n, edges.edges())?;
    let mut check = check_signals;
    let core = CoreAudit::prepare(&g, &offset, &label, Some(&mut check)).map_err(|e| to_py(py, e, None))?;
    Ok(AuditState { core })
}

fn audit_options(cap: i64, allow_isolation: Option<&Any>, atol: f64, rtol: f64, prune: Option<&Any>) -> PyResult<AuditOptions> {
    Ok(AuditOptions {
        max_memberships: cap,
        allow_isolation: flag(allow_isolation, false)?,
        prune: flag(prune, true)?,
        atol,
        rtol,
    })
}

/// Fixed-cover audit intervals (private binding).
#[pyfunction]
#[pyo3(name = "_audit_intervals", signature = (state, vertices, max_memberships=0, allow_isolation=None, atol=1e-10, rtol=1e-9, prune=None))]
#[allow(clippy::too_many_arguments)]
fn audit_intervals<'py>(
    py: Python<'py>,
    state: &Bound<'py, AuditState>,
    vertices: &Any<'py>,
    max_memberships: i64,
    allow_isolation: Option<&Any<'py>>,
    atol: f64,
    rtol: f64,
    prune: Option<&Any<'py>>,
) -> PyResult<Bound<'py, PyList>> {
    let o = audit_options(max_memberships, allow_isolation, atol, rtol, prune)?;
    let selected = audit_buffer(vertices, Kind8::Int, "vertices")?;
    let list = selected.ints();
    let mut check = check_signals;
    let intervals = state
        .get()
        .core
        .intervals(&o, Selection::List(&list), Some(&mut check))
        .map_err(|e| to_py(py, e, None))?;
    let out = PyList::empty(py);
    for (v, row) in intervals.into_iter().enumerate() {
        out.append(PyList::new(py, row)?)?;
        if v & 4095 == 0 {
            py.check_signals()?;
        }
    }
    Ok(out)
}

/// Fixed-cover audit regrets (private binding).
#[pyfunction]
#[pyo3(name = "_audit_regrets", signature = (state, vertices, resolutions, regret, tolerance, max_memberships=0, allow_isolation=None, atol=1e-10, rtol=1e-9, prune=None))]
#[allow(clippy::too_many_arguments)]
fn audit_regrets<'py>(
    py: Python<'py>,
    state: &Bound<'py, AuditState>,
    vertices: &Any<'py>,
    resolutions: &Any<'py>,
    regret: &Any<'py>,
    tolerance: &Any<'py>,
    max_memberships: i64,
    allow_isolation: Option<&Any<'py>>,
    atol: f64,
    rtol: f64,
    prune: Option<&Any<'py>>,
) -> PyResult<()> {
    let o = audit_options(max_memberships, allow_isolation, atol, rtol, prune)?;
    let selected = audit_buffer(vertices, Kind8::Int, "vertices")?;
    let gammas = audit_buffer(resolutions, Kind8::Double, "resolutions")?;
    let (nv, ng) = (selected.len(), gammas.len());
    if ng > 0 && nv > isize::MAX as usize / 8 / ng {
        return Err(PyOverflowError::new_err("Audit regret output size overflow"));
    }
    let count = nv * ng;
    let r = Buffer::get(regret, Kind8::Double, Some(count), "regret")?;
    let t = Buffer::get(tolerance, Kind8::Double, Some(count), "tolerance")?;
    if r.readonly() || t.readonly() {
        return Err(PyValueError::new_err("Audit outputs must be writable"));
    }
    // Computed apart and copied in: the outputs may alias each other or the inputs.
    let (mut rv, mut tv) = (vec![0.0f64; count], vec![0.0f64; count]);
    let list = selected.ints();
    let grid = gammas.doubles();
    let mut check = check_signals;
    state
        .get()
        .core
        .regrets(&o, Selection::List(&list), &grid, &mut rv, &mut tv, Some(&mut check))
        .map_err(|e| to_py(py, e, None))?;
    r.write_doubles(&rv);
    t.write_doubles(&tv);
    Ok(())
}

/// Which fused multiply-add this build runs: "instruction" or "soft".
#[pyfunction]
fn fma_path() -> &'static str {
    crate::fma::fma_path()
}

/// The Rust core of hedonic: best-response local moving for the CPM hedonic game.
#[cfg(not(feature = "core-module"))]
#[pymodule]
fn hedonic_core_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    module(m)
}

/// The same module as hedonic's own extension, hedonic._core (the package
/// built with the Rust core alone: tools/rust_pyproject.py).
#[cfg(feature = "core-module")]
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    module(m)
}

fn module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("core_version", crate::CORE_VERSION)?;
    m.add("source_sha256", crate::SOURCE_SHA256)?;
    m.add("backend", "rust")?;
    m.add_class::<AuditState>()?;
    m.add_function(wrap_pyfunction!(detect, m)?)?;
    m.add_function(wrap_pyfunction!(partition, m)?)?;
    m.add_function(wrap_pyfunction!(cover, m)?)?;
    m.add_function(wrap_pyfunction!(trace_partition, m)?)?;
    m.add_function(wrap_pyfunction!(trace_cover, m)?)?;
    m.add_function(wrap_pyfunction!(verify_partition, m)?)?;
    m.add_function(wrap_pyfunction!(verify_cover, m)?)?;
    m.add_function(wrap_pyfunction!(audit_prepare, m)?)?;
    m.add_function(wrap_pyfunction!(audit_intervals, m)?)?;
    m.add_function(wrap_pyfunction!(audit_regrets, m)?)?;
    m.add_function(wrap_pyfunction!(fma_path, m)?)?;
    Ok(())
}
