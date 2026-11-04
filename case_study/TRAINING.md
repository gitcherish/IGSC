# Five-fold training outputs

See [README.md](README.md) for commands. Each fold saves a complete best-validation
model, the actual configuration, cache indices, and corresponding test documents.
The later epoch wins validation-F1 ties. The wrapper inserts recording callbacks
into an in-memory copy of main.py and checks that the rest of the syntax tree is
unchanged. It does not add proj_head to the original optimizer.

The output captures Python/package versions and input/source hashes. Keep these
with the weights. The saved models support evaluation and trajectory export, not
resuming training with identical optimizer, scheduler, and RNG state.

Five-fold completion is required before automatic held-out trajectory export.
The exported cases are newly generated run outputs. They must not be described
as the manuscript's original cases without matching historical evidence.
