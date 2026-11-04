# EventStoryLine input caches

This folder contains the two feature caches used by the current extraction code.
The files were copied without modification. manifest.json records their sizes,
SHA-256 hashes, and audit statistics. splits/ contains document IDs and cache
indices for the current seed-209 five-fold protocol.

The train cache is the 233-document training/test pool. The dev cache is the
25-document fixed validation set. These are processed inputs, not a new original
dataset release. Their positive-pair inventory differs from the manuscript table.

Corpus: EventStoryLine v0.9, by Tommaso Caselli and Piek Vossen (2017).
Original source: https://github.com/tommasoc80/EventStoryLine
Corpus license and attribution terms:
https://github.com/tommasoc80/EventStoryLine/blob/master/LICENSE.md
https://creativecommons.org/licenses/by/3.0/
Inherited feature-processing implementation: https://github.com/LchengC/iLIF

The caches contain derived tokenized corpus material. The upstream dataset terms
and underlying source-text notices continue to apply; no new dataset license is
assigned by this package. Changes from the corpus distribution include tokenization,
event-pair feature construction, and separation into train/test-pool and dev caches.
The raw-corpus-to-XXX.npy step is still missing from the inherited preprocessing
script. Do not claim that these scripts alone rebuild the inputs from raw data.
