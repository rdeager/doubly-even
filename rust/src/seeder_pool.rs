//! D16 seeder helper pool: parallelises the seeder's σ_Q work.
//!
//! Post-D15 the parallel seeder's serial span is the Amdahl ceiling
//! (workers ~54 % busy at N=26 d=5; `architecture/08-post-d15-profile.md`
//! §5), and ~89 % of that span is the low-rank σ_Q orbit-min BFS — large
//! quotient dimension `L = N − 2k`, latency-bound bitset probes. The main
//! worker pool is starved exactly while the seeder runs, so this module
//! spawns a small persistent helper pool that the seeder's
//! `doubly_even_candidates_q_pooled` dispatches BFS-level and Gray-walk
//! ranges onto. The pool is created right before `traverse_seed` and
//! dropped right after, so helpers never compete with the worker-dominated
//! tail of the run.
//!
//! A persistent pool (not `std::thread::scope` per call) because the
//! seeder makes thousands of 1–10 ms σ_Q calls — per-call thread spawn
//! would eat the win.
//!
//! Workers stay sequential per-call: they are already saturated; pooling
//! inside them would oversubscribe.

use std::sync::atomic::{AtomicU64, Ordering};

type Task = Box<dyn FnOnce() + Send + 'static>;

/// Persistent FIFO helper pool. `execute` never blocks (unbounded task
/// channel); synchronisation back to the caller rides per-call result
/// channels, whose send/recv edges also provide the happens-before
/// ordering the atomic bitset claims rely on between BFS levels.
pub struct SeederPool {
    task_tx: Option<crossbeam_channel::Sender<Task>>,
    handles: Vec<std::thread::JoinHandle<()>>,
    size: usize,
    /// Minimum quotient dimension `L` for pooled σ_Q stages. Default 22
    /// (`DOUBLY_EVEN_SEEDER_PAR_MIN_L`): the D16 ship A/B at N=26 showed
    /// the pool wins 1.4–1.5× on the seeder's earliest, largest calls
    /// (L ≥ 22 — workers still idle, helpers get free cores) but LOSES
    /// 0.7–0.8× once workers are saturated and helpers contend (L ≤ 20).
    /// The L threshold doubles as an "early window" gate: at larger N
    /// more of the low-rank walk clears it, which is exactly when the
    /// seeder span grows.
    pub min_l: u32,
}

impl SeederPool {
    pub fn new(size: usize, min_l: u32) -> Self {
        let (task_tx, task_rx) = crossbeam_channel::unbounded::<Task>();
        let handles = (0..size)
            .map(|_| {
                let rx = task_rx.clone();
                std::thread::spawn(move || {
                    while let Ok(task) = rx.recv() {
                        task();
                    }
                })
            })
            .collect();
        Self {
            task_tx: Some(task_tx),
            handles,
            size,
            min_l,
        }
    }

    /// Resolve `(seeder_threads, min_l)` from the environment.
    /// `DOUBLY_EVEN_SEEDER_THREADS` defaults to `num_threads`; `0` or `1`
    /// disables the pool (exact pre-D16 seeder behaviour).
    pub fn env_defaults(num_threads: usize) -> (usize, u32) {
        let threads = std::env::var("DOUBLY_EVEN_SEEDER_THREADS")
            .ok()
            .and_then(|s| s.trim().parse::<usize>().ok())
            .unwrap_or(num_threads);
        let min_l = std::env::var("DOUBLY_EVEN_SEEDER_PAR_MIN_L")
            .ok()
            .and_then(|s| s.trim().parse::<u32>().ok())
            .unwrap_or(22);
        (threads, min_l)
    }

    pub fn size(&self) -> usize {
        self.size
    }

    pub fn execute(&self, task: Task) {
        self.task_tx
            .as_ref()
            .expect("SeederPool used after drop")
            .send(task)
            .expect("seeder helper thread exited early");
    }
}

impl Drop for SeederPool {
    fn drop(&mut self) {
        drop(self.task_tx.take());
        for h in self.handles.drain(..) {
            h.join().expect("seeder helper thread panicked");
        }
    }
}

/// Fixed-size concurrent bitset with an exactly-once `claim` primitive.
///
/// `Relaxed` suffices: the only invariant is single-location atomicity of
/// `fetch_or` (exactly one caller observes the bit unset), and all data
/// hand-off between BFS levels rides crossbeam channel edges, which are
/// acquire/release. Upgrading to `AcqRel` is a zero-risk fallback if any
/// doubt arises in review.
pub struct AtomicBitset {
    words: Vec<AtomicU64>,
}

impl AtomicBitset {
    pub fn new(bits: usize) -> Self {
        let n_words = bits.div_ceil(64);
        let mut words = Vec::with_capacity(n_words);
        words.resize_with(n_words, || AtomicU64::new(0));
        Self { words }
    }

    /// Set bit `i`; true iff THIS call flipped it (exactly-once winner).
    #[inline]
    pub fn claim(&self, i: usize) -> bool {
        let bit = 1u64 << (i & 63);
        self.words[i >> 6].fetch_or(bit, Ordering::Relaxed) & bit == 0
    }

    #[inline]
    pub fn contains(&self, i: usize) -> bool {
        self.words[i >> 6].load(Ordering::Relaxed) & (1u64 << (i & 63)) != 0
    }
}
