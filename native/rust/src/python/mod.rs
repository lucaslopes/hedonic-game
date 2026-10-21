//! The CPython module hedonic._core (feature `core-module`; `hedonic_core_rs`
//! without it): the functions of hedonic 2.0's C extension (its binding/) with
//! the same arguments, results, errors and messages, over the Rust engines.
//!
//! A call runs in three steps (`PartitionCall`, `CoverCall`): `prepare`
//! reads the arguments' buffers and the graph's edges, borrowed, with the
//! GIL held; the visits run on what the call owns; `finish` borrows the
//! inputs again for the quality. With an exact random.Random (run natively)
//! nothing in the visits needs Python, so they run without the GIL; another
//! generator is called for every random number, so they run attached, as in
//! the C binding. Either way no slice of Python memory lives while Python
//! code can run (the generator, a signal handler, another thread), which may
//! write or reallocate it. Interruption checks re-attach (`Signals`). The
//! audit's queries run without the GIL on its owned state.

mod args;

use std::borrow::Cow;

use pyo3::exceptions::{PyOverflowError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::audit::{AuditOptions, AuditState as CoreAudit, Selection};
use crate::cover::{Cover, CoverCall};
use crate::graph::{Edges, Graph, Options};
use crate::partition::PartitionCall;
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

/// The slices of a call's buffers, borrowed (copied only when unaligned).
/// Formed in `prepare` and in `finish`, never across the visits.
struct Slices<'a> {
    edges: Option<Cow<'a, [i64]>>,
    edge_weights: Option<Cow<'a, [f64]>>,
    node_weights: Option<Cow<'a, [f64]>>,
}

impl<'a> Slices<'a> {
    fn plain(g: &'a PlainGraph) -> Slices<'a> {
        Slices {
            edges: Some(g.edges.ints()),
            edge_weights: g.edge_weights.as_ref().map(|b| b.doubles()),
            node_weights: g.node_weights.as_ref().map(|b| b.doubles()),
        }
    }

    fn weights(edge_weights: &'a Option<Buffer>, node_weights: &'a Option<Buffer>) -> Slices<'a> {
        Slices {
            edges: None,
            edge_weights: edge_weights.as_ref().map(|b| b.doubles()),
            node_weights: node_weights.as_ref().map(|b| b.doubles()),
        }
    }

    fn graph<'b>(&'b self, py: Python<'_>, n: i64, edges: Edges<'b>) -> PyResult<Graph<'b>> {
        Graph::new(n, edges, self.edge_weights.as_deref(), self.node_weights.as_deref()).map_err(|e| to_py(py, e, None))
    }

    fn plain_graph(&self, py: Python<'_>, n: i64) -> PyResult<Graph<'_>> {
        self.graph(py, n, Edges::Flat(self.edges.as_deref().unwrap_or(&[])))
    }
}

/// Runs the visits of a prepared call with the core's callbacks over the
/// caller's generator: without the GIL for an exact random.Random (its state
/// was copied out, so the draws need no Python), attached for any other (each
/// draw calls getrandbits). A failure becomes the Python exception: the
/// generator's own, or a signal handler's, first.
fn run_visits<'py>(
    py: Python<'py>,
    rng: &mut PyRng<'py>,
    visits: impl FnOnce(&mut Callbacks) -> crate::error::Result<()> + Send,
) -> PyResult<()> {
    let mut signals = Signals::new(py)?;
    let result = if let PyRng::Native { mt, .. } = rng {
        py.detach(|| {
            let mut check = || signals.detached();
            let mut cb = Callbacks::new(mt);
            cb.interrupt = Some(&mut check);
            visits(&mut cb)
        })
    } else {
        let mut check = || signals.attached(py);
        let mut cb = Callbacks::new(rng);
        cb.interrupt = Some(&mut check);
        visits(&mut cb)
    };
    result.map_err(|e| {
        let first = rng.take_error().or_else(|| signals.take());
        to_py(py, e, first)
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
    allow_isolation: bool,
    traced: bool,
) -> PyResult<Py<PyAny>> {
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
                items.iter().map(long_long).collect::<PyResult<Vec<i64>>>()
            });
            match read {
                Ok(v) => Some(v),
                Err(e) => return rng.release(Err(e)),
            }
        }
        _ => None,
    };
    let mut trace = Trace::default();
    let prepared = {
        let slices = Slices::plain(&plain);
        let graph = slices.plain_graph(py, plain.n)?;
        PartitionCall::prepare(&graph, &options, start.as_deref(), traced.then_some(&mut trace))
    };
    let mut call = match prepared {
        Ok(call) => call,
        Err(e) => return rng.release(Err(to_py(py, e, None))),
    };
    let ran = run_visits(py, &mut rng, |cb| call.run(cb));
    rng.release(ran)?;
    let result = {
        let slices = Slices::plain(&plain);
        let graph = slices.plain_graph(py, plain.n)?;
        call.finish(&graph).map_err(|e| to_py(py, e, None))?
    };
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
#[pyo3(signature = (n, edges, rng, edge_weights=None, node_weights=None, membership=None, resolution=CArg::ok(1.0), max_total_communities=None, n_communities=None, allow_isolation=CArg::ok(true)))]
#[allow(clippy::too_many_arguments)]
fn partition<'py>(
    py: Python<'py>,
    n: CArg<isize>,
    edges: &Any<'py>,
    rng: &Any<'py>,
    edge_weights: Option<&Any<'py>>,
    node_weights: Option<&Any<'py>>,
    membership: Option<&Any<'py>>,
    resolution: CArg<f64>,
    max_total_communities: Option<&Any<'py>>,
    n_communities: Option<&Any<'py>>,
    allow_isolation: CArg<bool>,
) -> PyResult<Py<PyAny>> {
    plain_partition(py, n.get()?, edges, rng, edge_weights, node_weights, membership, resolution.get()?, max_total_communities, n_communities, allow_isolation.get()?, false)
}

/// partition() that also returns {'start', 'moves', 'raw_slack'}: the start rows in
/// the core's internal labels, (v, new row, improvement, bound) for every move, and
/// the slack before division by |W|. For tests.
#[pyfunction]
#[pyo3(name = "_trace_partition", signature = (n, edges, rng, edge_weights=None, node_weights=None, membership=None, resolution=CArg::ok(1.0), max_total_communities=None, n_communities=None, allow_isolation=CArg::ok(true)))]
#[allow(clippy::too_many_arguments)]
fn trace_partition<'py>(
    py: Python<'py>,
    n: CArg<isize>,
    edges: &Any<'py>,
    rng: &Any<'py>,
    edge_weights: Option<&Any<'py>>,
    node_weights: Option<&Any<'py>>,
    membership: Option<&Any<'py>>,
    resolution: CArg<f64>,
    max_total_communities: Option<&Any<'py>>,
    n_communities: Option<&Any<'py>>,
    allow_isolation: CArg<bool>,
) -> PyResult<Py<PyAny>> {
    plain_partition(py, n.get()?, edges, rng, edge_weights, node_weights, membership, resolution.get()?, max_total_communities, n_communities, allow_isolation.get()?, true)
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
            labels.push(long_long(item)?);
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
    allow_isolation: bool,
    traced: bool,
) -> PyResult<Py<PyAny>> {
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
    let mut trace = Trace::default();
    let prepared = {
        let slices = Slices::plain(&plain);
        let graph = slices.plain_graph(py, plain.n)?;
        CoverCall::prepare(&graph, &options, start.as_ref(), traced.then_some(&mut trace))
    };
    let mut call = match prepared {
        Ok(call) => call,
        Err(e) => return rng.release(Err(to_py(py, e, None))),
    };
    let ran = run_visits(py, &mut rng, |cb| call.run(cb));
    rng.release(ran)?;
    let result = {
        let slices = Slices::plain(&plain);
        let graph = slices.plain_graph(py, plain.n)?;
        call.finish(&graph).map_err(|e| to_py(py, e, None))?
    };
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
#[pyo3(signature = (n, edges, rng, edge_weights=None, node_weights=None, memberships=None, max_memberships=CArg::ok(2), resolution=CArg::ok(1.0), max_total_communities=None, n_communities=None, allow_isolation=CArg::ok(true)))]
#[allow(clippy::too_many_arguments)]
fn cover<'py>(
    py: Python<'py>,
    n: CArg<isize>,
    edges: &Any<'py>,
    rng: &Any<'py>,
    edge_weights: Option<&Any<'py>>,
    node_weights: Option<&Any<'py>>,
    memberships: Option<&Any<'py>>,
    max_memberships: CArg<i64>,
    resolution: CArg<f64>,
    max_total_communities: Option<&Any<'py>>,
    n_communities: Option<&Any<'py>>,
    allow_isolation: CArg<bool>,
) -> PyResult<Py<PyAny>> {
    plain_cover(py, n.get()?, edges, rng, edge_weights, node_weights, memberships, max_memberships.get()?, resolution.get()?, max_total_communities, n_communities, allow_isolation.get()?, false)
}

/// cover() with the record of _trace_partition(). For tests.
#[pyfunction]
#[pyo3(name = "_trace_cover", signature = (n, edges, rng, edge_weights=None, node_weights=None, memberships=None, max_memberships=CArg::ok(2), resolution=CArg::ok(1.0), max_total_communities=None, n_communities=None, allow_isolation=CArg::ok(true)))]
#[allow(clippy::too_many_arguments)]
fn trace_cover<'py>(
    py: Python<'py>,
    n: CArg<isize>,
    edges: &Any<'py>,
    rng: &Any<'py>,
    edge_weights: Option<&Any<'py>>,
    node_weights: Option<&Any<'py>>,
    memberships: Option<&Any<'py>>,
    max_memberships: CArg<i64>,
    resolution: CArg<f64>,
    max_total_communities: Option<&Any<'py>>,
    n_communities: Option<&Any<'py>>,
    allow_isolation: CArg<bool>,
) -> PyResult<Py<PyAny>> {
    plain_cover(py, n.get()?, edges, rng, edge_weights, node_weights, memberships, max_memberships.get()?, resolution.get()?, max_total_communities, n_communities, allow_isolation.get()?, true)
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
#[pyo3(signature = (graph, rng, resolution, *, max_memberships=CArg::ok(1), start=None, allow_isolation=CArg::ok(false), edge_weights=None, node_weights=None, max_total_communities=None, n_communities=None))]
#[allow(clippy::too_many_arguments)]
fn detect<'py>(
    py: Python<'py>,
    graph: &Any<'py>,
    rng: &Any<'py>,
    resolution: CArg<f64>,
    max_memberships: CArg<i64>,
    start: Option<&Any<'py>>,
    allow_isolation: CArg<bool>,
    edge_weights: Option<&Any<'py>>,
    node_weights: Option<&Any<'py>>,
    max_total_communities: Option<&Any<'py>>,
    n_communities: Option<&Any<'py>>,
) -> PyResult<Py<PyAny>> {
    // As C parses them: in parameter order, before anything else.
    let (resolution, max_memberships, allow_isolation) = (resolution.get()?, max_memberships.get()?, allow_isolation.get()?);
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
    // The counts first; the edges last, once every argument that can run
    // Python code is read (see GraphEdges).
    let counts = graph_counts(graph)?;
    let (n, m) = counts;
    let ew = optional_doubles(edge_weights, m as usize, "edge_weights")?;
    let nw = optional_doubles(node_weights, n as usize, "node_weights")?;
    let start = match start {
        Some(s) if !s.is_none() => read_start_rows(s, n, max_memberships)?,
        _ => None,
    };
    let mut rng = PyRng::new(rng)?;
    let edges = match graph_edges(graph, Some(counts)) {
        Ok(edges) => edges,
        Err(e) => return rng.release(Err(e)),
    };
    if max_memberships == 1 {
        let membership: Option<Vec<i64>> =
            start.map(|(offset, labels)| offset[..n as usize].iter().map(|&o| labels[o as usize]).collect());
        let prepared = {
            let slices = Slices::weights(&ew, &nw);
            let g = slices.graph(py, n, edges.edges())?;
            PartitionCall::prepare(&g, &options, membership.as_deref(), None)
        };
        let mut call = match prepared {
            Ok(call) => call,
            Err(e) => return rng.release(Err(to_py(py, e, None))),
        };
        let ran = run_visits(py, &mut rng, |cb| call.run(cb));
        rng.release(ran)?;
        let edges = edges.again(graph)?;
        let result = {
            let slices = Slices::weights(&ew, &nw);
            let g = slices.graph(py, n, edges.edges())?;
            call.finish(&g).map_err(|e| to_py(py, e, None))?
        };
        let list = PyList::new(py, &result.membership)?;
        Ok((list, result.count, result.quality, result.slack).into_pyobject(py)?.into_any().unbind())
    } else {
        let start = start.map(|(offset, labels)| Cover { offset, labels });
        let prepared = {
            let slices = Slices::weights(&ew, &nw);
            let g = slices.graph(py, n, edges.edges())?;
            CoverCall::prepare(&g, &options, start.as_ref(), None)
        };
        let mut call = match prepared {
            Ok(call) => call,
            Err(e) => return rng.release(Err(to_py(py, e, None))),
        };
        let ran = run_visits(py, &mut rng, |cb| call.run(cb));
        rng.release(ran)?;
        let edges = edges.again(graph)?;
        let result = {
            let slices = Slices::weights(&ew, &nw);
            let g = slices.graph(py, n, edges.edges())?;
            call.finish(&g).map_err(|e| to_py(py, e, None))?
        };
        let (rows, clusters) = cover_lists(py, &result.cover, result.count)?;
        Ok((rows, clusters, result.count, result.quality, result.slack).into_pyobject(py)?.into_any().unbind())
    }
}

/// Whether detect() would borrow `graph`'s edges from python-igraph's
/// igraph_t (true) or copy them through get_edgelist() (false). For tests:
/// a python-igraph whose layout changed reads false here.
#[pyfunction]
#[pyo3(name = "_edges_borrowed")]
fn edges_borrowed(graph: &Any<'_>) -> PyResult<bool> {
    Ok(graph_edges(graph, None)?.borrowed())
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
    items.iter().map(long_long).collect()
}

/// _verify_partition(n, edges, membership, edge_weights=None, node_weights=None,
/// resolution=1.0, max_total_communities=None, n_communities=None,
/// allow_isolation=True) -> {'vertices', 'improvable', 'max_certified', 'max_bound'}
///
/// The independent verifier (native/core/check/check.c): every deviation of every vertex
/// scored afresh with certified bounds. For tests.
#[pyfunction]
#[pyo3(name = "_verify_partition", signature = (n, edges, membership, edge_weights=None, node_weights=None, resolution=CArg::ok(1.0), max_total_communities=None, n_communities=None, allow_isolation=CArg::ok(true)))]
#[allow(clippy::too_many_arguments)]
fn verify_partition<'py>(
    py: Python<'py>,
    n: CArg<isize>,
    edges: &Any<'py>,
    membership: &Any<'py>,
    edge_weights: Option<&Any<'py>>,
    node_weights: Option<&Any<'py>>,
    resolution: CArg<f64>,
    max_total_communities: Option<&Any<'py>>,
    n_communities: Option<&Any<'py>>,
    allow_isolation: CArg<bool>,
) -> PyResult<Bound<'py, PyDict>> {
    // As C parses them: in parameter order, before anything else.
    let (n, resolution, allow_isolation) = (n.get()?, resolution.get()?, allow_isolation.get()?);
    let options = Options {
        resolution,
        max_memberships: 1,
        max_total_communities: optional_limit(max_total_communities, "max_total_communities")?,
        n_communities: optional_limit(n_communities, "n_communities")?,
        allow_isolation,
    };
    let plain = PlainGraph::new(n, edges, edge_weights, node_weights)?;
    let membership = read_ints(membership, plain.n as usize, "membership")?;
    let slices = Slices::plain(&plain);
    let graph = slices.plain_graph(py, plain.n)?;
    let report = check_partition(&graph, &options, &membership).map_err(|e| to_py(py, e, None))?;
    report_dict(py, &report)
}

/// _verify_cover(n, edges, memberships, max_memberships, ...): the same for a cover
/// (sorted rows of distinct labels). For tests.
#[pyfunction]
#[pyo3(name = "_verify_cover", signature = (n, edges, memberships, max_memberships, edge_weights=None, node_weights=None, resolution=CArg::ok(1.0), max_total_communities=None, n_communities=None, allow_isolation=CArg::ok(true)))]
#[allow(clippy::too_many_arguments)]
fn verify_cover<'py>(
    py: Python<'py>,
    n: CArg<isize>,
    edges: &Any<'py>,
    memberships: &Any<'py>,
    max_memberships: CArg<i64>,
    edge_weights: Option<&Any<'py>>,
    node_weights: Option<&Any<'py>>,
    resolution: CArg<f64>,
    max_total_communities: Option<&Any<'py>>,
    n_communities: Option<&Any<'py>>,
    allow_isolation: CArg<bool>,
) -> PyResult<Bound<'py, PyDict>> {
    // As C parses them: in parameter order, before anything else.
    let (n, max_memberships, resolution, allow_isolation) = (n.get()?, max_memberships.get()?, resolution.get()?, allow_isolation.get()?);
    let options = Options {
        resolution,
        max_memberships,
        max_total_communities: optional_limit(max_total_communities, "max_total_communities")?,
        n_communities: optional_limit(n_communities, "n_communities")?,
        allow_isolation,
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
    let slices = Slices::plain(&plain);
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
    let counts = graph_counts(graph)?;
    let (n, m) = counts;
    if n == isize::MAX as i64 {
        return Err(PyOverflowError::new_err("Too many audit vertices"));
    }
    let offsets = Buffer::get(offsets, Kind8::Int, Some(n as usize + 1), "offsets")?;
    let labels = audit_buffer(labels, Kind8::Int, "labels")?;
    {
        let (offset, label) = (offsets.ints(), labels.ints());
        if offset.iter().any(|&x| x < 0 || x as usize > label.len()) {
            return Err(PyValueError::new_err("Audit row offset outside labels buffer"));
        }
    }
    let ew = optional_doubles(edge_weights, m as usize, "edge_weights")?;
    let nw = optional_doubles(node_weights, n as usize, "node_weights")?;
    // Prepared with the GIL held: it reads the graph's borrowed edges, and
    // its interruption checks may run Python code, so the edges are read
    // last. (The queries, the long part, run without the GIL.)
    let edges = graph_edges(graph, Some(counts))?;
    let slices = Slices::weights(&ew, &nw);
    let g = slices.graph(py, n, edges.edges())?;
    let (offset, label) = (offsets.ints(), labels.ints());
    let mut signals = Signals::new(py)?;
    let mut check = || signals.attached(py);
    let core = CoreAudit::prepare(&g, &offset, &label, Some(&mut check));
    let core = core.map_err(|e| to_py(py, e, signals.take()))?;
    Ok(AuditState { core })
}

fn audit_options(cap: CArg<i64>, allow_isolation: CArg<bool>, atol: CArg<f64>, rtol: CArg<f64>, prune: CArg<bool>) -> PyResult<AuditOptions> {
    // As C parses them ("Lpddp"): in parameter order.
    Ok(AuditOptions {
        max_memberships: cap.get()?,
        allow_isolation: allow_isolation.get()?,
        atol: atol.get()?,
        rtol: rtol.get()?,
        prune: prune.get()?,
    })
}
/// Fixed-cover audit intervals (private binding).
#[pyfunction]
#[pyo3(name = "_audit_intervals", signature = (state, vertices, max_memberships=CArg::ok(0), allow_isolation=CArg::ok(false), atol=CArg::ok(1e-10), rtol=CArg::ok(1e-9), prune=CArg::ok(true)))]
#[allow(clippy::too_many_arguments)]
fn audit_intervals<'py>(
    py: Python<'py>,
    state: &Bound<'py, AuditState>,
    vertices: &Any<'py>,
    max_memberships: CArg<i64>,
    allow_isolation: CArg<bool>,
    atol: CArg<f64>,
    rtol: CArg<f64>,
    prune: CArg<bool>,
) -> PyResult<Bound<'py, PyList>> {
    let o = audit_options(max_memberships, allow_isolation, atol, rtol, prune)?;
    let selected = audit_buffer(vertices, Kind8::Int, "vertices")?;
    // Copied: the query runs without the GIL, while Python code may write
    // the caller's array.
    let list = selected.ints().into_owned();
    drop(selected);
    let core = &state.get().core;
    let mut signals = Signals::new(py)?;
    let intervals = py.detach(|| {
        let mut check = || signals.detached();
        core.intervals(&o, Selection::List(&list), Some(&mut check))
    });
    let intervals = intervals.map_err(|e| to_py(py, e, signals.take()))?;
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
#[pyo3(name = "_audit_regrets", signature = (state, vertices, resolutions, regret, tolerance, max_memberships=CArg::ok(0), allow_isolation=CArg::ok(false), atol=CArg::ok(1e-10), rtol=CArg::ok(1e-9), prune=CArg::ok(true)))]
#[allow(clippy::too_many_arguments)]
fn audit_regrets<'py>(
    py: Python<'py>,
    state: &Bound<'py, AuditState>,
    vertices: &Any<'py>,
    resolutions: &Any<'py>,
    regret: &Any<'py>,
    tolerance: &Any<'py>,
    max_memberships: CArg<i64>,
    allow_isolation: CArg<bool>,
    atol: CArg<f64>,
    rtol: CArg<f64>,
    prune: CArg<bool>,
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
    // Computed apart and copied in: the outputs may alias each other or the
    // inputs. The inputs are copied too: the query runs without the GIL,
    // while Python code may write the caller's arrays.
    let (mut rv, mut tv) = (vec![0.0f64; count], vec![0.0f64; count]);
    let list = selected.ints().into_owned();
    let grid = gammas.doubles().into_owned();
    let core = &state.get().core;
    let mut signals = Signals::new(py)?;
    let done = py.detach(|| {
        let mut check = || signals.detached();
        core.regrets(&o, Selection::List(&list), &grid, &mut rv, &mut tv, Some(&mut check))
    });
    done.map_err(|e| to_py(py, e, signals.take()))?;
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
///
/// gil_used = false: the module keeps no Python-visible mutable state (its
/// caches are once-initialised, its counters atomics), a call's buffers are
/// borrowed only while no Python code can run, and free-threaded builds read
/// igraph's edges through its API (see GraphEdges).
#[cfg(not(feature = "core-module"))]
#[pymodule(gil_used = false)]
fn hedonic_core_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    module(m)
}

/// The same module as hedonic's own extension, hedonic._core (the package's
/// build: the repository's pyproject.toml).
#[cfg(feature = "core-module")]
#[pymodule(gil_used = false)]
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
    m.add_function(wrap_pyfunction!(edges_borrowed, m)?)?;
    Ok(())
}
