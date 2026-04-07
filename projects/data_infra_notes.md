# Data Infrastructure Notes

Technical notes on data storage, processing infrastructure, and pipeline concepts used in lumaverse.

---

## Lance Fragments

A **fragment** is a Lance storage unit. A Lance dataset is stored as multiple fragments (files on disk/S3), each containing a subset of rows.

- Each incremental write (e.g. each `commit_percentage` flush during a pipeline run) creates a new fragment
- A dataset written with `commit_percentage=0.01` over 1M rows could produce hundreds of fragments
- **Reading with randomization across many fragments is slow** because it requires random I/O across all of them
- The multilingual podcast table (`whisperx__multilingual_v1.lance`) had 54,905 fragments — too many for randomized reads with `limit=3M`
- **Compacting** merges fragments into fewer, larger ones for faster reads

**Error encountered**:

```
[WARNING] Reading a dataset of 54905 fragments with limit=3000000 and randomized=True is slow!
Consider disabling randomization with --no-randomized OR better, compact the dataset ;)
```

**Fix**: Use `--no-randomized` flag, or compact the dataset beforehand.

---

## Ray Data Blocks

A **block** is a Ray Data processing unit. Ray Data groups rows into blocks (batches) for parallel processing through the pipeline.

- When the progress bar says `Completed 12 / 59`, it means 12 out of 59 blocks have been fully processed through all stages (read → process → write)
- Block size is influenced by `read_control_row_based_batch_size` (e.g. 256 in the fidelity pipeline) and Ray's internal partitioning
- The `Completed` counter only increments when a block is flushed through the entire pipeline — it can stay at 0 for a long time during model loading / cold start
- Blocks flow through stages independently: while block N is being written, block N+1 can be processing, and block N+2 can be reading

---

## Relationship Between Fragments and Blocks

| Concept | Layer | What it represents |
|---------|-------|--------------------|
| **Fragment** | Storage (Lance) | A file on S3 containing a subset of rows |
| **Block** | Processing (Ray Data) | A batch of rows flowing through the pipeline |

Fragments are about how data is **stored**. Blocks are about how data **flows through processing**. They are independent — Ray Data reads from fragments and groups rows into blocks based on its own batching logic.

### Mental Model: Bookshelf and Reading List

**Fragments = books on a shelf.** Each book (fragment) contains some pages (rows). Books can be different sizes — some thick, some thin. This is how the data is physically stored.

**Blocks = chapters you're assigned to read.** The reading coordinator (Ray Data) says "read 256 pages at a time." It pulls pages off the shelf sequentially, and every 256 pages it hands you a stack (block) to process. It doesn't care where one book ends and another begins — it just fills up the stack to the target size.

```
Shelf (Lance dataset):
  [Fragment A: 100 rows] [Fragment B: 500 rows] [Fragment C: 200 rows] ...

Reading coordinator (Ray Data, batch_size=256):
  Block 1: [A: all 100] + [B: first 156]     = 256 rows
  Block 2: [B: next 256]                      = 256 rows
  Block 3: [B: last 88] + [C: first 168]      = 256 rows
  Block 4: [C: last 32] + [next fragment...]   = ...
```

There is no 1:1 mapping between fragments and blocks:

- **Large fragments** (e.g. 10,000 rows) → split into many blocks
- **Small fragments** (e.g. 50 rows) → merged into fewer blocks
- Ray just streams rows from fragments and cuts them into blocks at the configured batch size

### Why Randomization + Many Fragments Is Slow

Imagine the books are scattered across different warehouses (S3 objects):

- **Sequential reading** = visit warehouses in order, grab whole books. Fast — large sequential reads.
- **Randomized reading with 54K tiny books** = randomly jump between warehouses to grab a few pages from each. Massive I/O overhead from thousands of small random reads.

This is why the 3M multilingual job failed with 54,905 fragments + randomization. Fix: `--no-randomized` or compact the dataset first.
