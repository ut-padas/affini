"""Leaf statuses, sample purposes and the depth limit shared by the region-mode modules."""

# Leaf statuses, stored as int8 codes (index into these tuples).
TERMINAL_REASONS = ("accepted", "max_depth", "split_failed")
VALIDATION_STATUSES = ("not_performed", "passed", "failed", "insufficient_sampling")
ACCEPTED, MAX_DEPTH, SPLIT_FAILED = 0, 1, 2
NOT_PERFORMED, PASSED, FAILED, INSUFFICIENT_SAMPLING = 0, 1, 2, 3

PURPOSE_TRAIN, PURPOSE_VALIDATION, PURPOSE_DIRECTION = 1, 2, 3
MAX_DEPTH_LIMIT = 62  # path codes are positive int64 with a leading sentinel bit
