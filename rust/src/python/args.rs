//! The arguments of the module's calls, read as native/binding reads them:
//! typed buffers (arguments.c), the caller's random generator (random.c), the
//! edges of an igraph.Graph (graph_edges.c) and the start rows of detect()
//! (start_rows.c). Every message is the C binding's.

use std::borrow::Cow;
use std::ffi::CStr;

use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::intern;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyDict, PyList, PyTuple, PyType};

use crate::error::{Error, Kind};
use crate::graph::Edges;
use crate::rng::{Mt19937, Rng};

// ---- Buffers -------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Kind8 {
    Int,
    Double,
}

/// A C-contiguous buffer of 8-byte items, released on drop. The `Py_buffer`
/// is boxed: exporters may point into it (`shape = &view->len`).
pub struct Buffer {
    view: Box<ffi::Py_buffer>,
}

impl Drop for Buffer {
    fn drop(&mut self) {
        // SAFETY: the view was filled by PyObject_GetBuffer and is released
        // once; every call that owns a Buffer holds the GIL until it drops.
        unsafe { ffi::PyBuffer_Release(&mut *self.view) }
    }
}

impl Buffer {
    /// `get_buffer` of arguments.c: the format must be an 8-byte integer
    /// (`q`, `l`, `<q`, `=q`) or double (`d`, `<d`, `=d`), and with
    /// `expected`, the length that many entries.
    pub fn get(obj: &Bound<'_, PyAny>, kind: Kind8, expected: Option<usize>, name: &str) -> PyResult<Buffer> {
        let mut view = Box::new(ffi::Py_buffer::new());
        // SAFETY: obj is a live object and view a writable Py_buffer.
        if unsafe { ffi::PyObject_GetBuffer(obj.as_ptr(), &mut *view, ffi::PyBUF_C_CONTIGUOUS | ffi::PyBUF_FORMAT) } < 0 {
            return Err(PyErr::fetch(obj.py()));
        }
        let buffer = Buffer { view };
        let format = if buffer.view.format.is_null() {
            None
        } else {
            // SAFETY: a non-null format is a NUL-terminated string the
            // exporter keeps alive with the view.
            Some(unsafe { CStr::from_ptr(buffer.view.format) }.to_bytes())
        };
        let ok = buffer.view.itemsize == 8
            && match (kind, format) {
                (Kind8::Int, Some(f)) => matches!(f, b"q" | b"l" | b"<q" | b"=q"),
                (Kind8::Double, Some(f)) => matches!(f, b"d" | b"<d" | b"=d"),
                _ => false,
            };
        if !ok {
            return Err(PyTypeError::new_err(format!(
                "{} must be a contiguous buffer of {}",
                name,
                if kind == Kind8::Int { "64-bit integers (array('q'))" } else { "doubles (array('d'))" }
            )));
        }
        if let Some(expected) = expected {
            if buffer.len() != expected {
                return Err(PyValueError::new_err(format!(
                    "{} has {} entries, expected {}",
                    name,
                    buffer.len(),
                    expected
                )));
            }
        }
        Ok(buffer)
    }

    pub fn len(&self) -> usize {
        (self.view.len / 8) as usize
    }

    pub fn readonly(&self) -> bool {
        self.view.readonly != 0
    }

    fn read<T: Copy>(&self) -> Cow<'_, [T]> {
        let (ptr, len) = (self.view.buf as *const T, self.len());
        if len == 0 {
            Cow::Borrowed(&[])
        } else if (ptr as usize) % std::mem::align_of::<T>() == 0 {
            // SAFETY: the exporter keeps len * 8 bytes at buf alive until the
            // release in drop; T is an 8-byte plain type and ptr is aligned.
            Cow::Borrowed(unsafe { std::slice::from_raw_parts(ptr, len) })
        } else {
            // SAFETY: as above, read one unaligned item at a time.
            Cow::Owned((0..len).map(|i| unsafe { ptr.add(i).read_unaligned() }).collect())
        }
    }

    /// The items, copied when `own` (Python code runs during the call).
    pub fn ints_owned(&self, own: bool) -> Cow<'_, [i64]> {
        match self.ints() {
            Cow::Borrowed(b) if own => Cow::Owned(b.to_vec()),
            c => c,
        }
    }

    pub fn doubles_owned(&self, own: bool) -> Cow<'_, [f64]> {
        match self.doubles() {
            Cow::Borrowed(b) if own => Cow::Owned(b.to_vec()),
            c => c,
        }
    }

    pub fn ints(&self) -> Cow<'_, [i64]> {
        self.read::<i64>()
    }

    pub fn doubles(&self) -> Cow<'_, [f64]> {
        self.read::<f64>()
    }

    /// Copies `values` into a writable buffer of doubles of the same length.
    pub fn write_doubles(&self, values: &[f64]) {
        debug_assert!(!self.readonly() && values.len() == self.len());
        // SAFETY: a writable buffer of len * 8 bytes, written byte-wise (no
        // alignment needed, no reference into the buffer is formed).
        unsafe {
            std::ptr::copy_nonoverlapping(values.as_ptr() as *const u8, self.view.buf as *mut u8, values.len() * 8)
        }
    }
}

/// An optional weight buffer: None stays None.
pub fn optional_doubles(obj: Option<&Bound<'_, PyAny>>, expected: usize, name: &str) -> PyResult<Option<Buffer>> {
    match obj {
        Some(o) if !o.is_none() => Ok(Some(Buffer::get(o, Kind8::Double, Some(expected), name)?)),
        _ => Ok(None),
    }
}

/// `optional_limit`: None is no limit (-1); otherwise a positive integer.
pub fn optional_limit(obj: Option<&Bound<'_, PyAny>>, name: &str) -> PyResult<i64> {
    match obj {
        Some(o) if !o.is_none() => {
            let limit: i64 = o.extract()?;
            if limit < 1 {
                return Err(PyValueError::new_err(format!("{name} must be a positive integer or None")));
            }
            Ok(limit)
        }
        _ => Ok(-1),
    }
}

/// The truth of an optional flag ("p" of PyArg_Parse), `default` when absent.
pub fn flag(obj: Option<&Bound<'_, PyAny>>, default: bool) -> PyResult<bool> {
    match obj {
        Some(o) => o.is_truthy(),
        None => Ok(default),
    }
}

/// PySequence_Fast: the items of a list or tuple, else of list(obj);
/// TypeError(message) for an object that is not iterable.
pub fn fast<'py>(obj: &Bound<'py, PyAny>, message: &str) -> PyResult<Vec<Bound<'py, PyAny>>> {
    if let Ok(list) = obj.cast::<PyList>() {
        return Ok(list.iter().collect());
    }
    if let Ok(tuple) = obj.cast::<PyTuple>() {
        return Ok(tuple.iter().collect());
    }
    let iter = obj.try_iter().map_err(|_| PyTypeError::new_err(message.to_string()))?;
    iter.collect()
}

/// The items of a list or tuple, read in place (each a new reference, so
/// Python code run by a check cannot leave one dangling), else of list(obj).
pub enum Items<'py> {
    List(Bound<'py, PyList>),
    Tuple(Bound<'py, PyTuple>),
    Other(Vec<Bound<'py, PyAny>>),
}

impl<'py> Items<'py> {
    /// As `fast`: TypeError(message) for an object that is not iterable.
    pub fn new(obj: &Bound<'py, PyAny>, message: &str) -> PyResult<Items<'py>> {
        if let Ok(list) = obj.cast::<PyList>() {
            return Ok(Items::List(list.clone()));
        }
        if let Ok(tuple) = obj.cast::<PyTuple>() {
            return Ok(Items::Tuple(tuple.clone()));
        }
        Ok(Items::Other(fast(obj, message)?))
    }

    pub fn len(&self) -> usize {
        match self {
            Items::List(list) => list.len(),
            Items::Tuple(tuple) => tuple.len(),
            Items::Other(items) => items.len(),
        }
    }

    pub fn first(&self) -> PyResult<Bound<'py, PyAny>> {
        match self {
            Items::List(list) => list.get_item(0),
            Items::Tuple(tuple) => tuple.get_item(0),
            Items::Other(items) => Ok(items[0].clone()),
        }
    }

    pub fn for_each(&self, mut f: impl FnMut(&Bound<'py, PyAny>) -> PyResult<()>) -> PyResult<()> {
        match self {
            Items::List(list) => list.iter().try_for_each(|item| f(&item)),
            Items::Tuple(tuple) => tuple.iter().try_for_each(|item| f(&item)),
            Items::Other(items) => items.iter().try_for_each(f),
        }
    }

    /// `for_each` over a start row, which must not be empty.
    fn for_each_row(&self, f: impl FnMut(&Bound<'py, PyAny>) -> PyResult<()>) -> PyResult<()> {
        if self.len() == 0 {
            return Err(PyValueError::new_err("each membership row must be non-empty"));
        }
        self.for_each(f)
    }
}

// ---- Errors --------------------------------------------------------------------

/// `raise_error` of _core.c: an exception a callback already set (the
/// generator's, or KeyboardInterrupt from a signal) wins; else the kind's.
pub fn to_py(py: Python<'_>, error: Error, rng_error: Option<PyErr>) -> PyErr {
    if let Some(e) = rng_error {
        return e;
    }
    if let Some(e) = PyErr::take(py) {
        return e;
    }
    match error.kind {
        Kind::NoMem => pyo3::exceptions::PyMemoryError::new_err(()),
        Kind::Overflow => pyo3::exceptions::PyOverflowError::new_err(error.message),
        Kind::Interrupted => pyo3::exceptions::PyKeyboardInterrupt::new_err(error.message),
        Kind::Inval => PyValueError::new_err(error.message),
        Kind::Callback => PyRuntimeError::new_err(error.message),
    }
}

/// The core's interruption check: PyErr_CheckSignals (a handler that raises
/// leaves its exception set, which `to_py` then returns).
pub fn check_signals() -> bool {
    // SAFETY: called only from calls that hold the GIL.
    unsafe { ffi::PyErr_CheckSignals() != 0 }
}

// ---- Random numbers ----------------------------------------------------------

const MT_N: usize = 624;

/// The caller's generator. An exact random.Random runs here on a copy of its
/// state (written back by `release`); any other is called as getrandbits(32).
pub enum PyRng<'py> {
    /// An exact random.Random run natively; `gauss_next` is None when its
    /// state is read and written in place (`direct_layout`).
    Native { mt: Mt19937, rng: Bound<'py, PyAny>, gauss_next: Option<Bound<'py, PyAny>> },
    Calls { getrandbits: Bound<'py, PyAny>, error: Option<PyErr> },
}

impl Rng for PyRng<'_> {
    #[inline]
    fn next32(&mut self) -> crate::error::Result<u32> {
        match self {
            PyRng::Native { mt, .. } => Ok(mt.next()),
            PyRng::Calls { getrandbits, error } => {
                match getrandbits.call1((32,)).and_then(|r| r.extract::<u64>()) {
                    Ok(value) => Ok(value as u32),
                    Err(e) => {
                        *error = Some(e);
                        Err(Error::new(Kind::Callback, "The random number source failed."))
                    }
                }
            }
        }
    }
}

// The generator of CPython's _random.Random is `{PyObject_HEAD; int index;
// uint32_t state[624];}` (Modules/_randommodule.c). Reading and writing it in
// place gives the same state as getstate/setstate (gauss_next, a Python
// attribute, is untouched either way) without two tuples of 625 ints per
// call. The layout is relied on only after a probe at first use agrees with
// getstate and setstate; otherwise the tuples are used, as in the C binding.
const MT_INDEX_AT: usize = std::mem::size_of::<ffi::PyObject>();
const MT_STATE_AT: usize = MT_INDEX_AT + 4;

/// The state of an exact random.Random read in place; None for an index
/// outside 0..=624. Safety: the layout passed `direct_layout`.
unsafe fn read_direct(rng: &Bound<'_, PyAny>) -> Option<Mt19937> {
    let base = rng.as_ptr() as *const u8;
    let index = std::ptr::read_unaligned(base.add(MT_INDEX_AT) as *const i32);
    if !(0..=MT_N as i32).contains(&index) {
        return None;
    }
    let mut mt = Mt19937 { state: [0; MT_N], index: index as usize };
    std::ptr::copy_nonoverlapping(base.add(MT_STATE_AT), mt.state.as_mut_ptr() as *mut u8, 4 * MT_N);
    Some(mt)
}

/// Writes a state in place. Safety: as `read_direct`.
unsafe fn write_direct(rng: &Bound<'_, PyAny>, mt: &Mt19937) {
    let base = rng.as_ptr() as *mut u8;
    std::ptr::write_unaligned(base.add(MT_INDEX_AT) as *mut i32, mt.index as i32);
    std::ptr::copy_nonoverlapping(mt.state.as_ptr() as *const u8, base.add(MT_STATE_AT), 4 * MT_N);
}

/// Whether random.Random has the layout above: its size fits it, and on a
/// probe generator the words read in place are getstate's, words written in
/// place are what getstate then returns, and the next draws follow from them.
fn direct_layout(py: Python<'_>) -> bool {
    static DIRECT: PyOnceLock<bool> = PyOnceLock::new();
    *DIRECT.get_or_init(py, || {
        let probe = || -> PyResult<bool> {
            // A free-threaded build guards the generator with its own lock.
            let sysconfig = py.import("sysconfig")?;
            if sysconfig.call_method1("get_config_var", ("Py_GIL_DISABLED",))?.is_truthy()? {
                return Ok(false);
            }
            let random = py.import("random")?.getattr("Random")?;
            let size: usize = random.getattr("__basicsize__")?.extract()?;
            if size < MT_STATE_AT + 4 * MT_N {
                return Ok(false);
            }
            let a = random.call1((20261020,))?;
            a.call_method1("getrandbits", (32 * 700,))?; // past a refill, index inside
            let state = a.call_method0("getstate")?;
            let words: Vec<u64> = state.get_item(1)?.extract()?;
            // SAFETY: the size covers the layout; a is an exact random.Random.
            let Some(mt) = (unsafe { read_direct(&a) }) else { return Ok(false) };
            if mt.state.iter().map(|&w| w as u64).chain([mt.index as u64]).ne(words.iter().copied()) {
                return Ok(false);
            }
            let b = random.call1((7,))?;
            // SAFETY: as above.
            unsafe { write_direct(&b, &mt) };
            if !b.call_method0("getstate")?.get_item(1)?.eq(state.get_item(1)?)? {
                return Ok(false);
            }
            let (x, y): (u64, u64) = (a.call_method1("getrandbits", (64,))?.extract()?, b.call_method1("getrandbits", (64,))?.extract()?);
            Ok(x == y)
        };
        probe().unwrap_or(false)
    })
}

/// The generator's words and index as the tuple setstate takes.
fn state_words<'py>(py: Python<'py>, mt: &Mt19937) -> PyResult<Bound<'py, PyTuple>> {
    // SAFETY: a new tuple of MT_N + 1 slots, each set once to a new reference
    // (PyTuple_SET_ITEM steals it) before the tuple is used.
    unsafe {
        let words = Bound::from_owned_ptr_or_err(py, ffi::PyTuple_New(MT_N as ffi::Py_ssize_t + 1))?;
        for i in 0..=MT_N {
            let value = if i < MT_N { mt.state[i] as u64 } else { mt.index as u64 };
            let item = ffi::PyLong_FromUnsignedLongLong(value);
            if item.is_null() {
                return Err(PyErr::fetch(py));
            }
            ffi::PyTuple_SET_ITEM(words.as_ptr(), i as ffi::Py_ssize_t, item);
        }
        Ok(words.cast_into_unchecked())
    }
}

/// `mt_take`: the state of an exact random.Random whose methods are the
/// class's, or None when the generator is to be called instead.
fn take_state<'py>(rng: &Bound<'py, PyAny>) -> PyResult<Option<PyRng<'py>>> {
    let py = rng.py();
    static RANDOM: PyOnceLock<Py<PyType>> = PyOnceLock::new();
    let random_type = RANDOM.import(py, "random", "Random")?;
    if !rng.get_type().is(random_type) {
        return Ok(None);
    }
    // SAFETY: rng is a live object; a NULL context is the documented use.
    let dict = unsafe { Bound::from_owned_ptr_or_err(py, ffi::PyObject_GenericGetDict(rng.as_ptr(), std::ptr::null_mut())) };
    let dict = match dict {
        Ok(d) => d,
        Err(_) => return Ok(None),
    };
    if let Ok(dict) = dict.cast::<PyDict>() {
        for name in ["getrandbits", "random", "getstate", "setstate"] {
            if dict.get_item(name).ok().flatten().is_some() {
                return Ok(None);
            }
        }
    }
    if direct_layout(py) {
        // SAFETY: the layout was checked against getstate/setstate; rng is an
        // exact random.Random.
        if let Some(mt) = unsafe { read_direct(rng) } {
            return Ok(Some(PyRng::Native { mt, rng: rng.clone(), gauss_next: None }));
        }
    }
    let got = rng.call_method0(intern!(py, "getstate"))?;
    let Ok(got) = got.cast::<PyTuple>() else { return Ok(None) };
    if got.len() != 3 {
        return Ok(None);
    }
    let version = got.get_item(0)?.extract::<i64>().unwrap_or(-1);
    let words = got.get_item(1)?;
    let Ok(words) = words.cast::<PyTuple>() else { return Ok(None) };
    if version != 3 || words.len() != MT_N + 1 {
        return Ok(None);
    }
    let mut mt = Mt19937 { state: [0; MT_N], index: 0 };
    for i in 0..=MT_N {
        // SAFETY: words is a tuple of MT_N + 1 items (checked above); the
        // item is borrowed while the tuple lives.
        let value = unsafe { ffi::PyLong_AsUnsignedLongLong(ffi::PyTuple_GET_ITEM(words.as_ptr(), i as ffi::Py_ssize_t)) };
        if value == u64::MAX && PyErr::take(py).is_some() {
            return Ok(None);
        }
        if i < MT_N {
            if value > 0xffff_ffff {
                return Ok(None);
            }
            mt.state[i] = value as u32;
        } else {
            if value > MT_N as u64 {
                return Ok(None);
            }
            mt.index = value as usize;
        }
    }
    Ok(Some(PyRng::Native { mt, rng: rng.clone(), gauss_next: Some(got.get_item(2)?) }))
}

impl<'py> PyRng<'py> {
    /// `make_rng` of random.c.
    pub fn new(rng: &Bound<'py, PyAny>) -> PyResult<PyRng<'py>> {
        if let Some(native) = take_state(rng)? {
            return Ok(native);
        }
        Ok(PyRng::Calls { getrandbits: rng.getattr("getrandbits")?, error: None })
    }

    /// Whether every draw is a Python call (which may run any Python code).
    pub fn calls_python(&self) -> bool {
        matches!(self, PyRng::Calls { .. })
    }

    /// The exception the generator raised, if any.
    pub fn take_error(&mut self) -> Option<PyErr> {
        match self {
            PyRng::Calls { error, .. } => error.take(),
            PyRng::Native { .. } => None,
        }
    }

    /// `rng_release`: a copied state goes back to its generator, also after
    /// an error (whose exception is kept).
    pub fn release<T>(self, result: PyResult<T>) -> PyResult<T> {
        if let PyRng::Native { mt, rng, gauss_next } = self {
            let py = rng.py();
            let written = match gauss_next {
                // SAFETY: direct_layout checked this layout; rng is an exact
                // random.Random, alive while bound.
                None => {
                    unsafe { write_direct(&rng, &mt) };
                    Ok(())
                }
                Some(gauss_next) => state_words(py, &mt)
                    .and_then(|words| rng.call_method1(intern!(py, "setstate"), ((3, words, gauss_next),)))
                    .map(drop),
            };
            if let Err(e) = written {
                return Err(match result {
                    Err(first) => first,
                    Ok(_) => e,
                });
            }
        }
        result
    }
}

// ---- The edges of an igraph.Graph ------------------------------------------------

/// The head of igraph's igraph_t (igraph 0.10 and 1.0), as graph_edges.c reads it.
#[repr(C)]
struct IgVector {
    stor_begin: *mut i64,
    stor_end: *mut i64,
    end: *mut i64,
}

#[repr(C)]
struct IgHead {
    n: i64,
    directed: u8,
    from: IgVector,
    to: IgVector,
}

/// The edges of a graph for one call: borrowed from the igraph_t behind the
/// capsule (kept alive here), or copied from get_edgelist().
pub struct GraphEdges<'py> {
    pub n: i64,
    pub m: i64,
    _capsule: Option<Bound<'py, PyAny>>,
    first: *const i64,
    second: *const i64,
    owned: Vec<i64>,
}

impl GraphEdges<'_> {
    /// Copies borrowed edges and lets the capsule go: for calls that run
    /// Python code while the core holds the edges.
    pub fn detach(&mut self) {
        if let Some(capsule) = self._capsule.take() {
            let mut owned = Vec::with_capacity(2 * self.m as usize);
            for e in 0..self.m as usize {
                // SAFETY: e < m, checked in `from_capsule`; the capsule is alive.
                unsafe { owned.extend_from_slice(&[*self.first.add(e), *self.second.add(e)]) };
            }
            self.owned = owned;
            drop(capsule);
        }
    }

    pub fn edges(&self) -> Edges<'_> {
        if self._capsule.is_some() && self.m > 0 {
            // SAFETY: checked in `from_capsule`: m entries at each pointer,
            // owned by the igraph_t the capsule keeps alive. A call whose
            // generator runs Python code detaches first; otherwise only a
            // signal handler could run Python during the call, and it must
            // not modify the graph (as with the C binding).
            unsafe {
                Edges::Split(
                    std::slice::from_raw_parts(self.first, self.m as usize),
                    std::slice::from_raw_parts(self.second, self.m as usize),
                )
            }
        } else {
            Edges::Flat(&self.owned)
        }
    }
}

fn span(v: &IgVector) -> Option<usize> {
    let (begin, end, stor_end) = (v.stor_begin as usize, v.end as usize, v.stor_end as usize);
    if v.stor_begin.is_null() || end < begin || end > stor_end || (end - begin) % 8 != 0 {
        return None;
    }
    Some((end - begin) / 8)
}

/// es[e].tuple == (u, v); Err on a Python error, Ok(None) when it is not a pair.
fn edge_matches(graph: &Bound<'_, PyAny>, e: i64, u: i64, v: i64) -> PyResult<Option<bool>> {
    let pair = graph.getattr("es")?.get_item(e)?.getattr("tuple")?;
    let Ok(pair) = pair.cast::<PyTuple>() else { return Ok(None) };
    if pair.len() != 2 {
        return Ok(None);
    }
    let (a, b): (i64, i64) = (pair.get_item(0)?.extract()?, pair.get_item(1)?.extract()?);
    Ok(Some(a == u && b == v))
}

fn from_capsule<'py>(graph: &Bound<'py, PyAny>, n: i64, m: i64) -> PyResult<Option<GraphEdges<'py>>> {
    let Ok(capsule) = graph.call_method0("__graph_as_capsule") else { return Ok(None) };
    // SAFETY: a capsule's pointer is read only when its name is NULL, as
    // python-igraph makes it; anything else clears the error and falls back.
    let head = unsafe { ffi::PyCapsule_GetPointer(capsule.as_ptr(), std::ptr::null()) } as *const IgHead;
    if head.is_null() {
        let _ = PyErr::take(graph.py());
        return Ok(None);
    }
    // SAFETY: the pointer is the igraph_t the capsule owns; only its head is
    // read, and every field is checked before an edge is read.
    let head = unsafe { &*head };
    if head.n != n || head.directed != 0 || span(&head.from) != Some(m as usize) || span(&head.to) != Some(m as usize) {
        return Ok(None);
    }
    let (first, second) = (head.to.stor_begin as *const i64, head.from.stor_begin as *const i64);
    // Undirected edges come out as (to, from): igraph_edge().
    for e in 0..m as usize {
        // SAFETY: e < m, the length of both vectors checked above.
        let (u, v) = unsafe { (*first.add(e), *second.add(e)) };
        if u < 0 || u >= n || v < 0 || v >= n {
            return Ok(None);
        }
    }
    if m > 0 {
        for probe in [0, m / 2, m - 1] {
            // SAFETY: probe < m.
            let (u, v) = unsafe { (*first.add(probe as usize), *second.add(probe as usize)) };
            if edge_matches(graph, probe, u, v)? != Some(true) {
                return Ok(None);
            }
        }
    }
    Ok(Some(GraphEdges { n, m, _capsule: Some(capsule), first, second, owned: Vec::new() }))
}

/// `graph_edges`: borrowed when the capsule can be read, else get_edgelist().
pub fn graph_edges<'py>(graph: &Bound<'py, PyAny>) -> PyResult<GraphEdges<'py>> {
    let n: i64 = graph.call_method0("vcount")?.extract()?;
    let m: i64 = graph.call_method0("ecount")?.extract()?;
    if n < 0 || m < 0 {
        return Err(PyValueError::new_err("invalid vertex or edge count"));
    }
    if let Some(edges) = from_capsule(graph, n, m)? {
        return Ok(edges);
    }
    let list = graph.call_method0("get_edgelist")?;
    let items = fast(&list, "get_edgelist() must return a sequence")?;
    if items.len() as i64 != m {
        return Err(PyRuntimeError::new_err("get_edgelist() disagrees with ecount()"));
    }
    let mut owned = Vec::new();
    owned.try_reserve_exact(2 * m as usize).map_err(|_| pyo3::exceptions::PyMemoryError::new_err(()))?;
    for pair in items {
        let Ok(pair) = pair.cast::<PyTuple>() else {
            return Err(PyTypeError::new_err("get_edgelist() must return pairs"));
        };
        if pair.len() != 2 {
            return Err(PyTypeError::new_err("get_edgelist() must return pairs"));
        }
        owned.push(pair.get_item(0)?.extract::<i64>()?);
        owned.push(pair.get_item(1)?.extract::<i64>()?);
    }
    Ok(GraphEdges { n, m, _capsule: None, first: std::ptr::null(), second: std::ptr::null(), owned })
}

// ---- Start rows of detect() -----------------------------------------------------

/// A label too large for int64: such a start can never be contiguous.
const LABEL_TOO_LARGE: i64 = i64::MAX;

/// `read_label`: a non-negative integer (bool excluded).
fn read_label(obj: &Bound<'_, PyAny>, integral: &Bound<'_, PyAny>) -> PyResult<i64> {
    let py = obj.py();
    let is_int = obj.is_instance_of::<pyo3::types::PyInt>();
    let ok = !obj.is_instance_of::<pyo3::types::PyBool>() && (is_int || obj.is_instance(integral)?);
    if !ok {
        return Err(PyValueError::new_err("membership labels must be integers"));
    }
    let index = if is_int {
        None
    } else {
        // SAFETY: obj is live; the result is a new reference or NULL with an error.
        Some(unsafe { Bound::from_owned_ptr_or_err(py, ffi::PyNumber_Index(obj.as_ptr()))? })
    };
    let mut overflow: std::os::raw::c_int = 0;
    // SAFETY: obj or index is a live int; overflow is a valid out pointer.
    let value = unsafe { ffi::PyLong_AsLongLongAndOverflow(index.as_ref().unwrap_or(obj).as_ptr(), &mut overflow) };
    if value == -1 {
        if let Some(e) = PyErr::take(py) {
            return Err(e);
        }
    }
    if overflow < 0 || (overflow == 0 && value < 0) {
        return Err(PyValueError::new_err("membership labels must be non-negative"));
    }
    Ok(if overflow > 0 { LABEL_TOO_LARGE } else { value })
}

/// `read_start_rows`: the checks of hedonic._native._start_rows in the same
/// order with the same messages. Rows of distinct labels (in order of first
/// appearance, or sorted for rows above 16 labels) as (offset, labels), or
/// None for an empty start.
pub fn read_start_rows(obj: &Bound<'_, PyAny>, n: i64, max_memberships: i64) -> PyResult<Option<(Vec<i64>, Vec<i64>)>> {
    let py = obj.py();
    let seq = Items::new(obj, "initial_membership must be a sequence")?;
    if seq.len() as i64 != n {
        return Err(PyValueError::new_err(format!("memberships must contain one row per vertex ({n})")));
    }
    if seq.len() == 0 {
        return Ok(None);
    }
    static INTEGRAL: PyOnceLock<Py<PyType>> = PyOnceLock::new();
    let integral = INTEGRAL.import(py, "numbers", "Integral")?;
    let is_row = |o: &Bound<'_, PyAny>| o.is_instance_of::<PyList>() || o.is_instance_of::<PyTuple>();
    let nested = is_row(&seq.first()?);
    let mut labels: Vec<i64> = Vec::with_capacity(seq.len());
    let mut offset: Vec<i64> = Vec::with_capacity(seq.len() + 1);
    let mut max_label: i64 = -1;
    offset.push(0);
    seq.for_each(|entry| {
        if is_row(entry) != nested {
            return Err(PyValueError::new_err("membership cannot mix flat and nested rows"));
        }
        let start = labels.len();
        if nested {
            Items::new(entry, "")?.for_each_row(|item| {
                labels.push(read_label(item, &integral)?);
                Ok(())
            })?;
        } else {
            labels.push(read_label(entry, &integral)?);
        }
        // Distinct labels in order of first appearance (sorted above 16).
        let row = &mut labels[start..];
        let count = row.len();
        let mut largest = row[0];
        let kept = if count <= 16 {
            let mut kept = 0;
            for i in 0..count {
                if !row[..kept].contains(&row[i]) {
                    row[kept] = row[i];
                    largest = largest.max(row[i]);
                    kept += 1;
                }
            }
            kept
        } else {
            row.sort_unstable();
            let mut kept = 1;
            for i in 1..count {
                if row[i] != row[kept - 1] {
                    row[kept] = row[i];
                    kept += 1;
                }
            }
            largest = row[kept - 1];
            kept
        };
        if kept as i64 > max_memberships {
            return Err(PyValueError::new_err(format!(
                "a vertex exceeds max_memberships ({kept} > {max_memberships})"
            )));
        }
        labels.truncate(start + kept);
        max_label = max_label.max(largest);
        offset.push(labels.len() as i64);
        Ok(())
    })?;
    // Contiguous from 0: every label below the largest is used.
    if max_label >= labels.len() as i64 {
        return Err(PyValueError::new_err("membership labels must be contiguous from 0"));
    }
    let mut present = vec![false; max_label as usize + 1];
    let mut distinct = 0i64;
    for &l in &labels {
        if !present[l as usize] {
            present[l as usize] = true;
            distinct += 1;
        }
    }
    if distinct != max_label + 1 {
        return Err(PyValueError::new_err("membership labels must be contiguous from 0"));
    }
    Ok(Some((offset, labels)))
}
