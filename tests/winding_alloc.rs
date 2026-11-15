//! The memory of the argument-principle count's first grids where the budget runs out in them: a polygon of many long
//! edges at a fine h0 is refused after the budget's worth of points, without the grids of every edge. The file has one
//! test, since its allocator counts the bytes of the whole process.
use multishoot::winding::{winding_batch, WindOpts, C64};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

/// the system allocator, with the bytes in use and their peak counted
struct Counting;
static NOW: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        let p = unsafe { System.alloc(l) };
        if !p.is_null() {
            let now = NOW.fetch_add(l.size(), Relaxed) + l.size();
            PEAK.fetch_max(now, Relaxed);
        }
        p
    }

    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        unsafe { System.dealloc(p, l) };
        NOW.fetch_sub(l.size(), Relaxed);
    }
}

#[global_allocator]
static COUNTING: Counting = Counting;

/// f's result and the peak bytes in use above those at its start
fn peak<T>(f: impl FnOnce() -> T) -> (T, usize) {
    let base = NOW.load(Relaxed);
    PEAK.store(base, Relaxed);
    let r = f();
    (r, PEAK.load(Relaxed) - base)
}

/// Pass, written before the first run: a regular polygon of 100 unit edges at h0 1e-4 (about 10^4 samples an edge)
/// with max_evals 20000 runs out of the budget in its second edge's grid. The error is the budget's, f is given 20000
/// points, and the call's peak heap above its start is below 8 MB, where the grids of all 100 edges (8 MB of t) and
/// their points (16 MB) take 24 MB. The control feeds the memory check the peak of allocating those grids and points
/// and must be rejected.
#[test]
fn budget_bounds_the_first_grids_memory() {
    let k = 100;
    let r = 0.5 / (std::f64::consts::PI / k as f64).sin();
    let poly: Vec<C64> = (0..k)
        .map(|i| {
            let a = std::f64::consts::TAU * i as f64 / k as f64;
            (r * a.cos(), r * a.sin())
        })
        .collect();
    let opts = WindOpts {
        h0: 1e-4,
        max_evals: 20000,
        ..WindOpts::default()
    };
    let mut given = 0;
    let f = |xs: &[C64]| {
        given += xs.len();
        vec![Ok((0.0, 0.0)); xs.len()]
    };
    let (e, bytes) = peak(|| winding_batch(f, &poly, &opts).unwrap_err());
    let (_, all) = peak(|| {
        let t = vec![0.0f64; k * 10_001];
        let xs = vec![(0.0f64, 0.0f64); k * 10_001];
        std::hint::black_box((t, xs));
    });
    let ok = |b: usize| b < 8 << 20;
    println!("{e}; f given {given} points; peak {bytes} bytes, control {all} bytes");
    assert!(e == "winding: more than 20000 evaluations" && given == 20000 && ok(bytes));
    assert!(!ok(all), "control did not fire");
}
