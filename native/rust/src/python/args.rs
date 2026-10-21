//! The arguments of the module's calls, read as native/binding reads them:
//! typed buffers (arguments.c), the caller's random generator (random.c), the
//! edges of an igraph.Graph (graph_edges.c) and the start rows of detect()
//! (start_rows.c), also as the CSR buffers of detect_arrays(). Every message
//! is the C binding's.

use std::borrow::Cow;
use std::ffi::CStr;
use std::time::{Duration, Instant};

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

/// The struct-module prefix of the target's own byte order: a buffer in the
/// other order would be read wrong, so it is refused like any other format.
#[cfg(target_endian = "little")]
const ORDER: u8 = b'<';
#[cfg(target_endian = "big")]
const ORDER: u8 = b'>';

/// A C-contiguous buffer of 8-byte items, released on drop. The `Py_buffer`
/// is boxed: exporters may point into it (`shape = &view->len`). While the
/// view is held the exporter keeps its memory where it is (array.array and
/// NumPy refuse to resize an exported buffer), so slices of it are formed only
/// while the GIL is held and no Python code runs: the calls read their inputs
/// in `prepare` and `finish`, never during the visits.
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
    /// (`q`, `l`, `=q`, or `<q` on a little-endian target, `>q` on a big-endian
    /// one) or double (`d`, `=d`, `<d` or `>d` likewise), and with `expected`,
    /// the length that many entries. arguments.c takes `<q` and `<d`
    /// everywhere; its wheels are all little-endian.
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
                (Kind8::Int, Some(f)) => matches!(f, b"q" | b"l" | b"=q") || f == [ORDER, b'q'],
                (Kind8::Double, Some(f)) => matches!(f, b"d" | b"=d") || f == [ORDER, b'd'],
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
        debug_assert!(std::mem::size_of::<T>() == 8 && self.view.itemsize == 8 && self.view.len >= 0);
        if len == 0 {
            Cow::Borrowed(&[])
        } else if (ptr as usize) % std::mem::align_of::<T>() == 0 {
            debug_assert!(!ptr.is_null());
            // SAFETY: a C-contiguous view of len * 8 bytes at buf (`get`
            // checked the item size; len = view.len / 8), which the exporter
            // keeps alive and in place until the release in drop, which the
            // slice cannot outlive (it borrows self); T is an 8-byte type
            // valid for every bit pattern, and ptr is non-null and aligned.
            Cow::Borrowed(unsafe { std::slice::from_raw_parts(ptr, len) })
        } else {
            // SAFETY: as above; each item is read on its own without alignment.
            Cow::Owned((0..len).map(|i| unsafe { ptr.add(i).read_unaligned() }).collect())
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
        assert!(!self.readonly() && values.len() == self.len());
        // SAFETY: a writable view of len * 8 bytes (asserted above: the
        // callers check both before computing), written byte-wise (no
        // alignment needed, and no reference into the buffer is formed, so
        // inputs that alias it are no concern).
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
            let limit = long_long(o)?;
            if limit < 1 {
                return Err(PyValueError::new_err(format!("{name} must be a positive integer or None")));
            }
            Ok(limit)
        }
        _ => Ok(-1),
    }
}

/// An argument that C's binding parses with PyArg_ParseTupleAndKeywords as
/// "n", "L", "d" or "p", converted the same way but kept with its error until
/// the body takes it (`get`, in parameter order, as C parses them): PyO3
/// prefixes a parameter's TypeError with "argument 'name': ", C does not.
pub struct CArg<T>(PyResult<T>);

impl<T> CArg<T> {
    /// A default value.
    pub const fn ok(value: T) -> CArg<T> {
        CArg(Ok(value))
    }

    pub fn get(self) -> PyResult<T> {
        self.0
    }
}

/// "d": PyFloat_AsDouble, as PyO3's f64 is.
impl<'a, 'py> FromPyObject<'a, 'py> for CArg<f64> {
    type Error = PyErr;
    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        Ok(CArg(obj.extract::<f64>()))
    }
}

/// "L": PyLong_AsLongLong itself, as `long_long`.
impl<'a, 'py> FromPyObject<'a, 'py> for CArg<i64> {
    type Error = PyErr;
    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        Ok(CArg(long_long(&obj)))
    }
}

/// PyLong_AsLongLong, which the C binding calls for every 64-bit integer it
/// reads: PyO3's i64 goes through C long, whose overflow message differs
/// ("Python int too large to convert to C long", not "int too big to
/// convert"), so the binding reads integers here, with C's messages on every
/// CPython.
pub fn long_long(obj: &Bound<'_, PyAny>) -> PyResult<i64> {
    // SAFETY: obj is a live object and the GIL is held (we hold a Bound);
    // PyLong_AsLongLong takes any object (it calls __index__) and reports a
    // failure as -1 with an exception set.
    let value = unsafe { ffi::PyLong_AsLongLong(obj.as_ptr()) };
    match (value, PyErr::take(obj.py())) {
        (-1, Some(e)) => Err(e),
        _ => Ok(value),
    }
}

/// PyLong_AsUnsignedLong, as random.c reads getrandbits(32).
fn unsigned_long(obj: &Bound<'_, PyAny>) -> PyResult<u64> {
    // SAFETY: as `long_long`; failure is (unsigned long) -1 with an
    // exception set.
    let value = unsafe { ffi::PyLong_AsUnsignedLong(obj.as_ptr()) };
    match (value, PyErr::take(obj.py())) {
        (v, Some(e)) if v == std::os::raw::c_ulong::MAX => Err(e),
        _ => Ok(value as u64),
    }
}

/// "n": PyNumber_Index, then PyLong_AsSsize_t, as getargs.c does.
impl<'a, 'py> FromPyObject<'a, 'py> for CArg<isize> {
    type Error = PyErr;
    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let py = obj.py();
        // SAFETY: as above; PyNumber_Index returns a new reference or NULL
        // with an exception set, which from_owned_ptr_or_err takes over.
        let index = match unsafe { Bound::from_owned_ptr_or_err(py, ffi::PyNumber_Index(obj.as_ptr())) } {
            Ok(index) => index,
            Err(e) => return Ok(CArg(Err(e))),
        };
        // SAFETY: index is a live int and the GIL is held; failure is -1
        // with an exception set.
        let value = unsafe { ffi::PyLong_AsSsize_t(index.as_ptr()) };
        match (value, PyErr::take(py)) {
            (-1, Some(e)) => Ok(CArg(Err(e))),
            _ => Ok(CArg(Ok(value))),
        }
    }
}

/// "p": the object's truth (None is false).
impl<'a, 'py> FromPyObject<'a, 'py> for CArg<bool> {
    type Error = PyErr;
    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        Ok(CArg(obj.is_truthy()))
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

/// The core's interruption check: PyErr_CheckSignals, whose exception (a
/// handler's, KeyboardInterrupt by default) is kept for `to_py`.
///
/// With the GIL held (`attached`) it runs at every check, as in the C
/// binding. Without it (`detached`) it re-attaches for the check, at the
/// core's cadence but at most every `RECHECK`, and only on the main thread:
/// CPython runs signal handlers there alone (PyErr_CheckSignals returns 0 on
/// any other thread), so a call on another thread never waits for the GIL.
pub struct Signals {
    main: bool,
    last: Option<Instant>,
    error: Option<PyErr>,
}

/// Without the GIL, the interruption check re-attaches at most this often:
/// Ctrl-C still stops a call within milliseconds, and a call that runs next
/// to busy Python threads does not wait for the GIL at every check.
const RECHECK: Duration = Duration::from_millis(10);

impl Signals {
    pub fn new(py: Python<'_>) -> PyResult<Signals> {
        Ok(Signals { main: on_main_thread(py)?, last: None, error: None })
    }

    /// The check with the GIL held: true to stop.
    pub fn attached(&mut self, py: Python<'_>) -> bool {
        match py.check_signals() {
            Ok(()) => false,
            Err(e) => {
                self.error = Some(e);
                true
            }
        }
    }

    /// The check without the GIL (see above): true to stop.
    pub fn detached(&mut self) -> bool {
        if !self.main {
            return false;
        }
        let now = Instant::now();
        if self.last.is_some_and(|last| now.duration_since(last) < RECHECK) {
            return false;
        }
        self.last = Some(now);
        Python::attach(|py| self.attached(py))
    }

    /// The exception a check raised, if any.
    pub fn take(&mut self) -> Option<PyErr> {
        self.error.take()
    }
}

/// Whether this is the interpreter's main thread (threading.main_thread()),
/// asked once per process (a fork from another thread makes that thread the
/// child's main thread).
fn on_main_thread(py: Python<'_>) -> PyResult<bool> {
    struct Main {
        pid: u32,
        ident: u64,
    }
    static GET_IDENT: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    static MAIN: std::sync::Mutex<Option<Main>> = std::sync::Mutex::new(None);
    let get_ident = GET_IDENT.get_or_try_init(py, || py.import("_thread")?.getattr("get_ident").map(Bound::unbind))?;
    let me: u64 = get_ident.bind(py).call0()?.extract()?;
    let pid = std::process::id();
    // The lock is never held while Python runs (which could wait for a thread
    // that waits for the lock while holding the GIL).
    let known = MAIN.lock().unwrap_or_else(|e| e.into_inner()).as_ref().and_then(|m| (m.pid == pid).then_some(m.ident));
    let ident = match known {
        Some(ident) => ident,
        None => {
            let ident = py.import("threading")?.call_method0("main_thread")?.getattr("ident")?.extract()?;
            *MAIN.lock().unwrap_or_else(|e| e.into_inner()) = Some(Main { pid, ident });
            ident
        }
    };
    Ok(ident == me)
}

// ---- Random numbers ----------------------------------------------------------

const MT_N: usize = 624;

/// The caller's generator. An exact random.Random runs here on a copy of its
/// state (written back by `release`); any other is called as getrandbits(32).
/// One per call, on the stack: the large variant needs no box.
#[allow(clippy::large_enum_variant)]
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
                match getrandbits.call1((32,)).and_then(|r| unsigned_long(&r)) {
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
const MT_END: usize = MT_STATE_AT + 4 * MT_N;

/// The state of an exact random.Random read in place; None for an index
/// outside 0..=624.
///
/// # Safety
/// `direct_layout` returned true, `rng`'s type is exactly random.Random (so
/// its object is at least MT_END bytes with the layout above) and the GIL is
/// held (nothing else touches the generator meanwhile).
unsafe fn read_direct(rng: &Bound<'_, PyAny>) -> Option<Mt19937> {
    debug_assert!(rng.get_type().getattr("__basicsize__").and_then(|b| b.extract::<usize>()).is_ok_and(|b| b >= MT_END));
    let base = rng.as_ptr() as *const u8;
    // SAFETY: the caller's contract: MT_END bytes of the live object at base,
    // read without alignment, as plain integers (any bits are valid).
    let index = unsafe { std::ptr::read_unaligned(base.add(MT_INDEX_AT) as *const i32) };
    if !(0..=MT_N as i32).contains(&index) {
        return None;
    }
    let mut mt = Mt19937 { state: [0; MT_N], index: index as usize };
    // SAFETY: as above; the destination is the 4 * MT_N bytes of mt.state.
    unsafe { std::ptr::copy_nonoverlapping(base.add(MT_STATE_AT), mt.state.as_mut_ptr() as *mut u8, 4 * MT_N) };
    Some(mt)
}

/// Writes a state in place.
///
/// # Safety
/// As `read_direct`.
unsafe fn write_direct(rng: &Bound<'_, PyAny>, mt: &Mt19937) {
    debug_assert!(mt.index <= MT_N);
    let base = rng.as_ptr() as *mut u8;
    // SAFETY: the caller's contract: MT_END bytes of the live object, which
    // no reference of ours points into, written without alignment.
    unsafe {
        std::ptr::write_unaligned(base.add(MT_INDEX_AT) as *mut i32, mt.index as i32);
        std::ptr::copy_nonoverlapping(mt.state.as_ptr() as *const u8, base.add(MT_STATE_AT), 4 * MT_N);
    }
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
            if size < MT_END {
                return Ok(false);
            }
            let a = random.call1((20261020,))?;
            a.call_method1("getrandbits", (32 * 700,))?; // past a refill, index inside
            let state = a.call_method0("getstate")?;
            let words: Vec<u64> = state.get_item(1)?.extract()?;
            // SAFETY: random.Random's basic size covers MT_END bytes and `a`
            // is an exact random.Random; the GIL is held. A wrong layout reads
            // integers that the comparisons below reject.
            let Some(mt) = (unsafe { read_direct(&a) }) else { return Ok(false) };
            if mt.state.iter().map(|&w| w as u64).chain([mt.index as u64]).ne(words.iter().copied()) {
                return Ok(false);
            }
            let b = random.call1((7,))?;
            // SAFETY: as above (the read just matched getstate's words).
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
    let words: Vec<u64> = mt.state.iter().map(|&w| w as u64).chain([mt.index as u64]).collect();
    PyTuple::new(py, words)
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
    // The instance dict (C: PyObject_GenericGetDict): a method set on the
    // instance makes the generator one to call.
    let Ok(dict) = rng.getattr(intern!(py, "__dict__")) else { return Ok(None) };
    if let Ok(dict) = dict.cast::<PyDict>() {
        for name in ["getrandbits", "random", "getstate", "setstate"] {
            if dict.get_item(name).ok().flatten().is_some() {
                return Ok(None);
            }
        }
    }
    if direct_layout(py) {
        // SAFETY: direct_layout checked the layout against getstate/setstate;
        // rng's type is exactly random.Random (above); the GIL is held.
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
    for (i, word) in words.iter().enumerate() {
        // A word that is no int in [0, 2^64) (C: PyLong_AsUnsignedLongLong
        // failing) makes the generator one to call.
        let Ok(value) = word.extract::<u64>() else { return Ok(None) };
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
                None => {
                    // SAFETY: the state was read in place (`take_state`): the
                    // layout passed direct_layout, rng is an exact
                    // random.Random, alive while bound, and the GIL is held.
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
/// capsule, or copied from get_edgelist().
///
/// Borrowed edges are read only while the GIL is held and no Python code
/// runs (`prepare` and `finish` of a call), never during the visits: Python
/// code (another thread while the GIL is released, the generator's
/// getrandbits, a signal handler) may add edges, which reallocates them. So
/// `again` reads the capsule afresh for `finish`.
pub struct GraphEdges {
    pub n: i64,
    pub m: i64,
    borrowed: bool,
    first: *const i64,
    second: *const i64,
    owned: Vec<i64>,
}

impl GraphEdges {
    /// The edge list for `Graph::new`.
    ///
    /// Borrowed, it points into the graph: call it with the GIL held, and use
    /// it only before any Python code can run (it is the pointers read by the
    /// last `graph_edges`/`again`, which checked them).
    pub fn edges(&self) -> Edges<'_> {
        if self.borrowed && self.m > 0 {
            debug_assert!(!self.first.is_null() && !self.second.is_null());
            debug_assert!((self.first as usize) % std::mem::align_of::<i64>() == 0);
            debug_assert!((self.second as usize) % std::mem::align_of::<i64>() == 0);
            // SAFETY: `from_capsule` checked, just before this call's Python-
            // free stretch began, that both pointers are non-null, aligned and
            // start m initialised entries inside their igraph vectors (end -
            // stor_begin = m, end <= stor_end), owned by the igraph_t of a
            // graph the caller keeps alive; no Python code runs while the
            // slices live (see above), so nothing reallocates or writes them.
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

    /// Whether the edges point into the graph (`_edges_borrowed`, tests).
    pub fn borrowed(&self) -> bool {
        self.borrowed
    }

    /// The edges again, for `finish` after Python code may have run: borrowed
    /// ones read afresh (and checked again), copied ones kept. A graph whose
    /// counts changed meanwhile is refused.
    pub fn again(self, graph: &Bound<'_, PyAny>) -> PyResult<GraphEdges> {
        if !self.borrowed {
            return Ok(self);
        }
        graph_edges(graph, Some((self.n, self.m)))
    }
}

/// The length of an igraph vector of i64 whose storage looks sound: non-null
/// and aligned, end within [stor_begin, stor_end], whole items.
fn span(v: &IgVector) -> Option<usize> {
    let (begin, end, stor_end) = (v.stor_begin as usize, v.end as usize, v.stor_end as usize);
    if v.stor_begin.is_null() || begin % std::mem::align_of::<i64>() != 0 || end < begin || end > stor_end {
        return None;
    }
    if (end - begin) % std::mem::size_of::<i64>() != 0 {
        return None;
    }
    Some((end - begin) / std::mem::size_of::<i64>())
}

/// es[e].tuple as a pair of ints; Err on a Python error, Ok(None) when it is
/// not a pair.
fn edge_tuple(graph: &Bound<'_, PyAny>, e: i64) -> PyResult<Option<(i64, i64)>> {
    let pair = graph.getattr("es")?.get_item(e)?.getattr("tuple")?;
    let Ok(pair) = pair.cast::<PyTuple>() else { return Ok(None) };
    if pair.len() != 2 {
        return Ok(None);
    }
    Ok(Some((long_long(&pair.get_item(0)?)?, long_long(&pair.get_item(1)?)?)))
}

/// Whether this free-threaded interpreter runs without the GIL now (then
/// another thread can run igraph code on the graph at any time, and its
/// edges are copied through igraph's own API instead).
fn gil_disabled(py: Python<'_>) -> bool {
    py.import("sys")
        .and_then(|sys| sys.getattr("_is_gil_enabled"))
        .and_then(|f| f.call0())
        .and_then(|on| on.is_truthy())
        .is_ok_and(|on| !on)
}

/// `edges_from_capsule` of graph_edges.c: the edges of the igraph_t behind
/// Graph.__graph_as_capsule(), in get_edgelist()'s order, when everything
/// read from it agrees with the public API (vcount, ecount, undirected, every
/// end a vertex, edges 0, m/2 and m - 1 as graph.es reports them); None (fall
/// back to get_edgelist) otherwise. The Python calls come first: from the
/// read of the struct to the end of the caller's use of the edges no Python
/// code runs, so nothing can change the graph in between.
fn from_capsule(graph: &Bound<'_, PyAny>, n: i64, m: i64) -> PyResult<Option<GraphEdges>> {
    // igraph's integers are as wide as a pointer (IGRAPH_INTEGER_SIZE): on a
    // 32-bit target the head is not the one above, so the edges are copied.
    if cfg!(not(target_pointer_width = "64")) || gil_disabled(graph.py()) {
        return Ok(None);
    }
    let Ok(capsule) = graph.call_method0("__graph_as_capsule") else { return Ok(None) };
    let probes = if m > 0 { [0, m / 2, m - 1] } else { [0; 3] };
    let mut seen = [(0i64, 0i64); 3];
    for (k, &e) in probes.iter().enumerate().take(if m > 0 { 3 } else { 0 }) {
        match edge_tuple(graph, e)? {
            Some(pair) => seen[k] = pair,
            None => return Ok(None),
        }
    }
    // SAFETY: capsule is a live object; PyCapsule_GetPointer checks that it
    // is a capsule named NULL (as python-igraph makes it) and otherwise sets
    // an error, cleared below before falling back. It runs no Python code.
    let head = unsafe { ffi::PyCapsule_GetPointer(capsule.as_ptr(), std::ptr::null()) } as *const IgHead;
    if head.is_null() {
        let _ = PyErr::take(graph.py());
        return Ok(None);
    }
    if (head as usize) % std::mem::align_of::<IgHead>() != 0 {
        return Ok(None);
    }
    // SAFETY: python-igraph's capsule points at the igraph_t inside the graph
    // object, which `graph` keeps alive; only the head's fields (integers and
    // pointers: any bits are valid) are copied out, and every one of them is
    // checked against the public API before an edge is read.
    let head = unsafe { std::ptr::read(head) };
    if head.n != n || head.directed != 0 || span(&head.from) != Some(m as usize) || span(&head.to) != Some(m as usize) {
        return Ok(None);
    }
    let (first, second) = (head.to.stor_begin as *const i64, head.from.stor_begin as *const i64);
    // Undirected edges come out as (to, from): igraph_edge().
    for e in 0..m as usize {
        // SAFETY: e < m, the length of both vectors (`span`: non-null,
        // aligned, m whole items before end <= stor_end).
        let (u, v) = unsafe { (*first.add(e), *second.add(e)) };
        if u < 0 || u >= n || v < 0 || v >= n {
            return Ok(None);
        }
    }
    for (k, &e) in probes.iter().enumerate().take(if m > 0 { 3 } else { 0 }) {
        // SAFETY: as above, e < m.
        let pair = unsafe { (*first.add(e as usize), *second.add(e as usize)) };
        if pair != seen[k] {
            return Ok(None);
        }
    }
    Ok(Some(GraphEdges { n, m, borrowed: true, first, second, owned: Vec::new() }))
}

/// `call_count` of graph_edges.c for vcount and ecount.
pub fn graph_counts(graph: &Bound<'_, PyAny>) -> PyResult<(i64, i64)> {
    let n = long_long(&graph.call_method0("vcount")?)?;
    let m = long_long(&graph.call_method0("ecount")?)?;
    if n < 0 || m < 0 {
        return Err(PyValueError::new_err("invalid vertex or edge count"));
    }
    Ok((n, m))
}

/// `graph_edges`: borrowed when the capsule can be read, else get_edgelist().
/// Called last before the edges are used: with `counts` (read earlier, before
/// other arguments ran Python code), a graph that changed in between is
/// refused.
pub fn graph_edges(graph: &Bound<'_, PyAny>, counts: Option<(i64, i64)>) -> PyResult<GraphEdges> {
    let (n, m) = graph_counts(graph)?;
    if counts.is_some_and(|c| c != (n, m)) {
        return Err(PyRuntimeError::new_err("The graph changed during the call."));
    }
    if let Some(edges) = from_capsule(graph, n, m)? {
        return Ok(edges);
    }
    edges_from_edgelist(graph, n, m)
}

/// `edges_from_edgelist`: a copy of get_edgelist(), checked against ecount().
fn edges_from_edgelist(graph: &Bound<'_, PyAny>, n: i64, m: i64) -> PyResult<GraphEdges> {
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
        owned.push(long_long(&pair.get_item(0)?)?);
        owned.push(long_long(&pair.get_item(1)?)?);
    }
    Ok(GraphEdges { n, m, borrowed: false, first: std::ptr::null(), second: std::ptr::null(), owned })
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

/// Keeps the distinct labels of a non-empty start row at its front, in order of
/// first appearance (sorted for rows above 16 labels); returns how many and the
/// largest.
fn distinct_row(row: &mut [i64]) -> (usize, i64) {
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
    (kept, largest)
}

fn exceeds(kept: usize, max_memberships: i64) -> PyErr {
    PyValueError::new_err(format!("a vertex exceeds max_memberships ({kept} > {max_memberships})"))
}

/// Contiguous from 0: every label below the largest, `max_label`, is used.
fn contiguous(labels: &[i64], max_label: i64) -> PyResult<()> {
    if max_label >= labels.len() as i64 {
        return Err(PyValueError::new_err("membership labels must be contiguous from 0"));
    }
    let mut present = vec![false; max_label as usize + 1];
    let mut distinct = 0i64;
    for &l in labels {
        if !present[l as usize] {
            present[l as usize] = true;
            distinct += 1;
        }
    }
    if distinct != max_label + 1 {
        return Err(PyValueError::new_err("membership labels must be contiguous from 0"));
    }
    Ok(())
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
                labels.push(read_label(item, integral)?);
                Ok(())
            })?;
        } else {
            labels.push(read_label(entry, integral)?);
        }
        let (kept, largest) = distinct_row(&mut labels[start..]);
        if kept as i64 > max_memberships {
            return Err(exceeds(kept, max_memberships));
        }
        labels.truncate(start + kept);
        max_label = max_label.max(largest);
        offset.push(labels.len() as i64);
        Ok(())
    })?;
    contiguous(&labels, max_label)?;
    Ok(Some((offset, labels)))
}

/// The start rows of detect_arrays(): `read_start_rows` over the CSR rows
/// `labels[offset[v]..offset[v + 1]]` (`offset` has n + 1 entries, checked by
/// the caller's buffer): the same checks, in vertex order, with the same
/// messages, and the same rows.
pub fn start_rows_from_csr(offset: &[i64], labels: &[i64], max_memberships: i64) -> PyResult<Option<(Vec<i64>, Vec<i64>)>> {
    let n = offset.len().saturating_sub(1);
    if n == 0 {
        return Ok(None);
    }
    let bounds = offset[0] == 0
        && offset[n] as u64 == labels.len() as u64
        && offset.windows(2).all(|w| w[0] <= w[1]);
    if !bounds {
        return Err(PyValueError::new_err(
            "start_offsets must start at 0, never decrease and end at len(start_labels)",
        ));
    }
    let mut kept_labels: Vec<i64> = Vec::with_capacity(labels.len());
    let mut kept_offset: Vec<i64> = Vec::with_capacity(n + 1);
    let mut max_label: i64 = -1;
    kept_offset.push(0);
    for v in 0..n {
        let row = &labels[offset[v] as usize..offset[v + 1] as usize];
        if row.is_empty() {
            return Err(PyValueError::new_err("each membership row must be non-empty"));
        }
        if row.iter().any(|&l| l < 0) {
            return Err(PyValueError::new_err("membership labels must be non-negative"));
        }
        let start = kept_labels.len();
        kept_labels.extend_from_slice(row);
        let (kept, largest) = distinct_row(&mut kept_labels[start..]);
        if kept as i64 > max_memberships {
            return Err(exceeds(kept, max_memberships));
        }
        kept_labels.truncate(start + kept);
        max_label = max_label.max(largest);
        kept_offset.push(kept_labels.len() as i64);
    }
    contiguous(&kept_labels, max_label)?;
    Ok(Some((kept_offset, kept_labels)))
}
