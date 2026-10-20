//! Inputs (native/core/shared/inputs.c): the graph and the options, their
//! checks, the adjacency and the labels of a start.

use crate::error::{filled, Error, Result};

/// Vertex, community and label ids (32 bits, as the C core's `hig_id`).
pub type Id = i32;
pub const ID_MAX: Id = i32::MAX;

/// The edge list: edge e joins `flat[2e]` and `flat[2e + 1]`, or `first[e]`
/// and `second[e]`.
#[derive(Clone, Copy, Debug)]
pub enum Edges<'a> {
    Flat(&'a [i64]),
    Split(&'a [i64], &'a [i64]),
}

/// An undirected graph over plain slices (`hig_graph_t`). The constructor
/// checks that every slice is as long as the counts say, so nothing reads
/// out of bounds.
#[derive(Clone, Copy, Debug)]
pub struct Graph<'a> {
    pub vertex_count: i64,
    pub edge_count: i64,
    pub edges: Edges<'a>,
    pub edge_weights: Option<&'a [f64]>,
    pub node_weights: Option<&'a [f64]>,
}

impl<'a> Graph<'a> {
    pub fn new(
        vertex_count: i64,
        edges: Edges<'a>,
        edge_weights: Option<&'a [f64]>,
        node_weights: Option<&'a [f64]>,
    ) -> Result<Self> {
        let edge_count = match edges {
            Edges::Flat(flat) => {
                if flat.len() % 2 != 0 {
                    return Err(Error::inval("The edge list has an odd number of entries."));
                }
                (flat.len() / 2) as i64
            }
            Edges::Split(first, second) => {
                if first.len() != second.len() {
                    return Err(Error::inval("The two edge arrays differ in length."));
                }
                first.len() as i64
            }
        };
        if let Some(w) = edge_weights {
            if w.len() as i64 != edge_count {
                return Err(Error::inval("edge_weights must have one entry per edge."));
            }
        }
        if let Some(w) = node_weights {
            if vertex_count < 0 || w.len() as i64 != vertex_count {
                return Err(Error::inval("node_weights must have one entry per vertex."));
            }
        }
        Ok(Graph { vertex_count, edge_count, edges, edge_weights, node_weights })
    }

    /// Entry h of the edge list read as one array: edge e joins 2e and 2e + 1.
    #[inline(always)]
    pub fn end(&self, h: i64) -> i64 {
        match self.edges {
            Edges::Flat(flat) => flat[h as usize],
            Edges::Split(first, second) => {
                let e = (h >> 1) as usize;
                if h & 1 != 0 {
                    second[e]
                } else {
                    first[e]
                }
            }
        }
    }

    #[inline(always)]
    pub fn edge_weight(&self, e: i64) -> f64 {
        self.edge_weights.map_or(1.0, |w| w[e as usize])
    }

    #[inline(always)]
    pub fn node_weight(&self, v: i64) -> f64 {
        self.node_weights.map_or(1.0, |w| w[v as usize])
    }

    /// The sum of the edge weights, in edge order (W).
    pub fn total_edge_weight(&self) -> f64 {
        let mut total = 0.0;
        for e in 0..self.edge_count {
            total += self.edge_weight(e);
        }
        total
    }
}

/// The options both engines share (`hig_options_t` without the callbacks).
#[derive(Clone, Copy, Debug)]
pub struct Options {
    /// gamma, finite.
    pub resolution: f64,
    /// Labels per vertex at most: 1 for partitions.
    pub max_memberships: i64,
    /// Occupied communities at most; <= 0: no limit.
    pub max_total_communities: i64,
    /// Exactly this many communities; <= 0: no limit.
    pub n_communities: i64,
    /// May a vertex move to an empty community.
    pub allow_isolation: bool,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            resolution: 1.0,
            max_memberships: 1,
            max_total_communities: -1,
            n_communities: -1,
            allow_isolation: true,
        }
    }
}

#[inline(always)]
pub fn limit_set(limit: i64) -> bool {
    limit > 0
}

/// Any sum of payoff terms stays below DBL_MAX / 2^12, so no computation of a
/// payoff or of its error bound overflows.
pub const ARITHMETIC_BOUND: f64 = f64::MAX / 4096.0;

/// Checks the graph and the options both engines share; `cover` adds the
/// cover's domain (non-negative edge weights, no loops, at least one edge).
/// Normalizes the count limits (<= 0 becomes -1).
pub fn check_input(graph: &Graph, options: &mut Options, cover: bool) -> Result<()> {
    let n = graph.vertex_count;
    let m = graph.edge_count;
    let game = if cover { "The overlapping game" } else { "The hedonic game" };
    let mut abs_edge_weight = 0.0f64;
    let mut total_edge_weight = 0.0f64;
    let mut total_node_weight = 0.0f64;

    if n < 0 || m < 0 {
        return Err(Error::inval("Vertex and edge counts must be non-negative."));
    }
    if n > (ID_MAX - 1) as i64 {
        return Err(Error::overflow(format!("The core handles at most {} vertices.", ID_MAX - 1)));
    }
    for h in 0..2 * m {
        let end = graph.end(h);
        if end < 0 || end >= n {
            return Err(Error::inval(format!(
                "Edge {} has an end outside [0, {}).",
                h / 2,
                n
            )));
        }
    }
    if !options.resolution.is_finite() {
        return Err(Error::inval(format!("{game} resolution must be finite.")));
    }
    if options.max_memberships < 1 {
        return Err(Error::inval("max_memberships must be at least 1."));
    }
    if options.max_total_communities <= 0 {
        options.max_total_communities = -1;
    }
    if options.n_communities <= 0 {
        options.n_communities = -1;
    }
    if limit_set(options.max_total_communities)
        && options.n_communities > options.max_total_communities
    {
        return Err(Error::inval("n_communities must not exceed max_total_communities."));
    }
    if cover {
        if n < 1 || m < 1 {
            return Err(Error::inval(format!(
                "{game} requires a nonempty graph with at least one edge."
            )));
        }
        for e in 0..m {
            if graph.end(2 * e) == graph.end(2 * e + 1) {
                return Err(Error::inval(format!("{game} requires a loopless graph.")));
            }
        }
        if options.max_memberships > n {
            return Err(Error::inval(format!(
                "{game} max_memberships must not exceed the number of vertices."
            )));
        }
        if options.n_communities > n.saturating_mul(options.max_memberships) {
            return Err(Error::inval(
                "n_communities exceeds n * max_memberships, the largest number of communities a cover can occupy.",
            ));
        }
    } else {
        if options.max_memberships != 1 {
            return Err(Error::inval("A partition has max_memberships = 1; use hig_cover."));
        }
        if options.n_communities > n {
            return Err(Error::inval(
                "n_communities must not exceed the number of vertices for a partition.",
            ));
        }
    }
    for e in 0..m {
        let w = graph.edge_weight(e);
        if !w.is_finite() || (cover && w < 0.0) {
            return Err(Error::inval(if cover {
                format!("{game} edge weights must be finite and non-negative.")
            } else {
                format!("{game} edge weights must be finite.")
            }));
        }
        abs_edge_weight += w.abs();
        total_edge_weight += w;
    }
    if !(abs_edge_weight <= ARITHMETIC_BOUND) {
        return Err(Error::overflow(format!("{game} total edge weight would overflow.")));
    }
    if cover && !(total_edge_weight > 0.0) {
        return Err(Error::inval(format!("{game} requires positive finite total edge weight.")));
    }
    for v in 0..n {
        let w = graph.node_weight(v);
        if !w.is_finite() || w < 0.0 {
            return Err(Error::inval(format!(
                "{game} node weights must be finite and non-negative."
            )));
        }
        total_node_weight += w;
    }
    // |gamma| N^2 m bounds every crowding sum (m labels of mass <= N each).
    let scale = if total_node_weight > 1.0 { total_node_weight } else { 1.0 };
    let cap = ARITHMETIC_BOUND / scale / scale / options.max_memberships as f64;
    if !total_node_weight.is_finite() || options.resolution.abs() > cap {
        return Err(Error::overflow(format!("{game} crowding term would overflow.")));
    }
    Ok(())
}

/// The adjacency of the game: v's neighbours are `adj[start[v]..start[v+1]]`,
/// in increasing edge id, self-loops left out.
#[derive(Default)]
pub struct Adjacency {
    pub n: Id,
    pub start: Vec<i64>,
    pub adj: Vec<Id>,
    /// Empty when every edge weight is 1.
    pub weight: Vec<f64>,
    pub weighted: bool,
    /// Empty when every node weight is 1.
    pub node_weight: Vec<f64>,
    pub node_weighted: bool,
    /// N.
    pub total_node_weight: f64,
    pub max_degree: i64,
}

impl Adjacency {
    /// Counting sort of the half-edges by owner, edges placed from the last
    /// to the first so each block lists v's edges in increasing edge id.
    pub fn build(graph: &Graph) -> Result<Adjacency> {
        let n = graph.vertex_count as Id;
        let m = graph.edge_count;
        let unit = match graph.edge_weights {
            Some(w) => w.iter().all(|&x| x == 1.0),
            None => true,
        };
        let node_weighted = match graph.node_weights {
            Some(w) => w.iter().any(|&x| x != 1.0),
            None => false,
        };
        let mut a = Adjacency { n, weighted: !unit, node_weighted, ..Default::default() };
        if node_weighted {
            a.node_weight = graph.node_weights.unwrap().to_vec();
        }
        for v in 0..n as usize {
            a.total_node_weight += if node_weighted { a.node_weight[v] } else { 1.0 };
        }
        let mut start = filled(n as usize + 1, 0i64)?;
        for e in 0..m {
            let (x, y) = (graph.end(2 * e), graph.end(2 * e + 1));
            if x != y {
                start[x as usize + 1] += 1;
                start[y as usize + 1] += 1;
            }
        }
        for v in 0..n as usize {
            if start[v + 1] > a.max_degree {
                a.max_degree = start[v + 1];
            }
            start[v + 1] += start[v];
        }
        let half_edges = start[n as usize];
        a.adj = filled(half_edges as usize, 0 as Id)?;
        if !unit {
            a.weight = filled(half_edges as usize, 0.0)?;
        }
        // start[v + 1] is the end of v's block; filling backwards moves it to
        // the block's beginning, so afterwards start[v + 1] = begin(v).
        for e in (0..m).rev() {
            let (x, y) = (graph.end(2 * e), graph.end(2 * e + 1));
            if x != y {
                start[x as usize + 1] -= 1;
                start[y as usize + 1] -= 1;
                let ia = start[x as usize + 1] as usize;
                let ib = start[y as usize + 1] as usize;
                a.adj[ia] = y as Id;
                a.adj[ib] = x as Id;
                if !unit {
                    let w = graph.edge_weight(e);
                    a.weight[ia] = w;
                    a.weight[ib] = w;
                }
            }
        }
        start.copy_within(1..=n as usize, 0);
        start[n as usize] = half_edges;
        a.start = start;
        Ok(a)
    }

    #[inline(always)]
    pub fn node_weight(&self, v: Id) -> f64 {
        if self.node_weighted {
            self.node_weight[v as usize]
        } else {
            1.0
        }
    }

    #[inline(always)]
    pub fn range(&self, v: Id) -> std::ops::Range<usize> {
        self.start[v as usize] as usize..self.start[v as usize + 1] as usize
    }
}

/// Maps non-negative labels to 0 .. distinct - 1, preserving their order.
pub fn compact_labels(input: &[i64], out: &mut [Id]) -> Result<i64> {
    let count = input.len();
    let largest = input.iter().copied().max().unwrap_or(-1).max(-1);
    let mut next: i64 = 0;
    if largest < 4 * count as i64 + 64 {
        // Dense: a rank per possible label.
        let mut rank = filled((largest + 1) as usize, 0 as Id)?;
        for &x in input {
            rank[x as usize] = 1;
        }
        for c in 0..=largest {
            let r = &mut rank[c as usize];
            if *r != 0 {
                if next >= ID_MAX as i64 {
                    return Err(Error::overflow("Too many distinct labels."));
                }
                *r = next as Id;
                next += 1;
            } else {
                *r = -1;
            }
        }
        for (o, &x) in out.iter_mut().zip(input) {
            *o = rank[x as usize];
        }
    } else {
        let mut sorted = input.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        next = sorted.len() as i64;
        if next > ID_MAX as i64 {
            return Err(Error::overflow("Too many distinct labels."));
        }
        for (o, x) in out.iter_mut().zip(input) {
            *o = sorted.partition_point(|y| y < x) as Id;
        }
    }
    Ok(next)
}
