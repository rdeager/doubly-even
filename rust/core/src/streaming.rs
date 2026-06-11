//! Binary streaming output for the enumerate kernel (N ≥ 28 unblock).
//!
//! Per-worker file format consumed by the Python `stream_reader` and
//! `merge_stream.py` validation harness:
//!
//! ```text
//! file := header record*
//! header := b"DEKS" version:u32 n:u32 worker_id:u32       // 16 bytes, LE
//! record := k:u8 aut_order:u128 basis:[u64; k]            // 1 + 16 + 8k bytes, LE
//! ```
//!
//! At N=29 we estimate ~134 B/record × ~81 M classes ≈ ~11 GB total
//! across all worker files. A 256 KB buffer keeps syscalls amortised
//! even at the parallel kernel's ~10 µs/emit cadence, and 384 workers
//! still fit comfortably in RAM (~100 MB total of buffer space).
//!
//! Errors are surfaced as `io::Result`; the caller panics the worker
//! on any write failure — silently dropping records would invalidate
//! the post-run `Σ N!/|Aut| == gaborit_sigma(N, k)` assertion that the
//! streaming kernel relies on for correctness.

use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;

use crate::types::BinVec;

pub const MAGIC: [u8; 4] = *b"DEKS";
pub const VERSION: u32 = 1;

/// Header is fixed 16 bytes: magic (4) + version (4) + N (4) + worker_id (4).
pub const HEADER_LEN: usize = 16;

/// 256 KB buffer per worker. Sized for the parallel kernel's emit
/// cadence (~10 µs/class) — at 256 KB / (134 B/record) ≈ 2000 records
/// per syscall, syscalls happen at ~50 Hz per worker.
const BUFFER_SIZE: usize = 256 * 1024;

pub struct BinaryWriter<W: Write> {
    inner: BufWriter<W>,
}

impl BinaryWriter<File> {
    /// Open `path` (truncating any existing file), write the file
    /// header, and return a writer ready for `write_class` calls.
    pub fn create(path: &Path, n: u32, worker_id: u32) -> io::Result<Self> {
        let file = File::create(path)?;
        let inner = BufWriter::with_capacity(BUFFER_SIZE, file);
        let mut w = Self { inner };
        w.write_header(n, worker_id)?;
        Ok(w)
    }
}

impl<W: Write> BinaryWriter<W> {
    fn write_header(&mut self, n: u32, worker_id: u32) -> io::Result<()> {
        self.inner.write_all(&MAGIC)?;
        self.inner.write_all(&VERSION.to_le_bytes())?;
        self.inner.write_all(&n.to_le_bytes())?;
        self.inner.write_all(&worker_id.to_le_bytes())?;
        Ok(())
    }

    /// Append one canonical class. `rref.len()` must fit in a `u8` —
    /// always true since `k ≤ MAX_N/2 ≤ 32` (debug-asserted).
    #[inline]
    pub fn write_class(&mut self, aut_order: u128, rref: &[BinVec]) -> io::Result<()> {
        let k = rref.len();
        debug_assert!(k <= u8::MAX as usize, "k = {k} exceeds u8 range");
        self.inner.write_all(&[k as u8])?;
        self.inner.write_all(&aut_order.to_le_bytes())?;
        for &row in rref {
            self.inner.write_all(&row.to_le_bytes())?;
        }
        Ok(())
    }

    pub fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}

// No explicit `Drop` impl: `BufWriter::Drop` already flushes and swallows
// errors. Adding one here would forbid moving the inner buffer out of
// `BinaryWriter` (E0509), which the round-trip tests rely on.

#[cfg(test)]
mod tests {
    use super::*;

    fn write_records(records: &[(u128, Vec<BinVec>)], n: u32, worker_id: u32) -> Vec<u8> {
        let mut w = BinaryWriter {
            inner: BufWriter::new(Vec::new()),
        };
        w.write_header(n, worker_id).unwrap();
        for (aut, rref) in records {
            w.write_class(*aut, rref).unwrap();
        }
        w.flush().unwrap();
        w.inner.into_inner().unwrap()
    }

    fn parse_header(buf: &[u8]) -> (u32, u32, u32) {
        assert_eq!(&buf[0..4], &MAGIC);
        let version = u32::from_le_bytes(buf[4..8].try_into().unwrap());
        let n = u32::from_le_bytes(buf[8..12].try_into().unwrap());
        let worker_id = u32::from_le_bytes(buf[12..16].try_into().unwrap());
        (version, n, worker_id)
    }

    fn parse_records(buf: &[u8]) -> Vec<(u128, Vec<BinVec>)> {
        let mut out = Vec::new();
        let mut i = HEADER_LEN;
        while i < buf.len() {
            let k = buf[i] as usize;
            i += 1;
            let aut = u128::from_le_bytes(buf[i..i + 16].try_into().unwrap());
            i += 16;
            let mut basis = Vec::with_capacity(k);
            for _ in 0..k {
                basis.push(u64::from_le_bytes(buf[i..i + 8].try_into().unwrap()));
                i += 8;
            }
            out.push((aut, basis));
        }
        assert_eq!(i, buf.len(), "trailing bytes after last record");
        out
    }

    #[test]
    fn header_only_round_trip() {
        let buf = write_records(&[], 29, 17);
        assert_eq!(buf.len(), HEADER_LEN);
        let (version, n, worker_id) = parse_header(&buf);
        assert_eq!(version, VERSION);
        assert_eq!(n, 29);
        assert_eq!(worker_id, 17);
    }

    #[test]
    fn single_record_round_trip() {
        let rec = (12_345u128, vec![0xDEAD_BEEFu64, 0xCAFE_BABEu64, 0x1234u64]);
        let buf = write_records(&[rec.clone()], 22, 0);
        let recs = parse_records(&buf);
        assert_eq!(recs, vec![rec]);
    }

    #[test]
    fn fuzz_k_range() {
        // Span k = 1..=28 — covers production max_k at every N up to 64.
        let records: Vec<(u128, Vec<BinVec>)> = (1..=28u32)
            .map(|k| {
                let aut = (k as u128) * 1_000_000 + 7;
                let basis = (0..k).map(|i| (i as u64) * 0xABCDEF + k as u64).collect();
                (aut, basis)
            })
            .collect();
        let buf = write_records(&records, 64, 99);
        let recs = parse_records(&buf);
        assert_eq!(recs, records);
    }

    #[test]
    fn u128_aut_order_extremes() {
        // |Aut| at N=29 is bounded by 29! ≈ 8.84e30 ≪ u128::MAX ≈ 3.4e38.
        // Confirm the wire format survives both u128::MAX and 1.
        let records = vec![
            (u128::MAX, vec![0xFFFF_FFFF_FFFF_FFFFu64]),
            (1u128, vec![1u64]),
        ];
        let buf = write_records(&records, 64, 0);
        let recs = parse_records(&buf);
        assert_eq!(recs, records);
    }

    #[test]
    fn file_round_trip() {
        // Exercise the BinaryWriter<File> create() path end-to-end.
        let mut path = std::env::temp_dir();
        path.push(format!(
            "doubly_even_streaming_test_{}.bin",
            std::process::id()
        ));
        {
            let mut w = BinaryWriter::create(&path, 24, 3).unwrap();
            w.write_class(48, &[0b1111u64]).unwrap();
            w.write_class(96, &[0b0011u64, 0b1100u64]).unwrap();
            w.flush().unwrap();
        }
        let buf = std::fs::read(&path).unwrap();
        let (version, n, worker_id) = parse_header(&buf);
        assert_eq!(version, VERSION);
        assert_eq!(n, 24);
        assert_eq!(worker_id, 3);
        let recs = parse_records(&buf);
        assert_eq!(
            recs,
            vec![(48u128, vec![0b1111u64]), (96u128, vec![0b0011u64, 0b1100u64])],
        );
        std::fs::remove_file(&path).ok();
    }
}
