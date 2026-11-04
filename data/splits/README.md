# Document splits

seed209/fold1.json through fold5.json record the five folds generated from the
included train cache using sklearn.model_selection.KFold with shuffle=True and
random_state=209. Indices are zero-based and refer to the exact cache order.
Each file includes both indices and topic/document identifiers. Validation
documents are in seed209/validation.json and are fixed across all five folds.

The five test partitions cover the 233-document pool exactly once, and the
25 validation documents are disjoint from that pool. The split unit is a document;
topics can overlap between training and test partitions. These files reconstruct
the current code protocol and do not establish the identity of historical runs
that did not record their exact inputs and folds.
