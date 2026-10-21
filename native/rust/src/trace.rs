//! A traced run (native/core/check/hedonic_check.h, tests only): the start
//! rows in the engine's internal labels, and (v, new row, improvement, bound)
//! for every accepted move.

#[derive(Clone, Debug, Default)]
pub struct TraceMove {
    pub v: i64,
    pub row: Vec<i32>,
    pub improvement: f64,
    pub bound: f64,
}

#[derive(Clone, Debug, Default)]
pub struct Trace {
    pub start: Vec<Vec<i32>>,
    pub moves: Vec<TraceMove>,
}

impl Trace {
    pub fn record_start(&mut self, row: &[i32]) {
        self.start.push(row.to_vec());
    }

    pub fn record_move(&mut self, v: i64, improvement: f64, bound: f64, row: &[i32]) {
        self.moves.push(TraceMove { v, row: row.to_vec(), improvement, bound });
    }
}
