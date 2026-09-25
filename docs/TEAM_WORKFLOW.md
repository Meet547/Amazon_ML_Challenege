# Team Workflow

## Ownership

### Meet
- Phase 1
- Phase 2
- Normalization
- Blocking
- Candidate generation

### Teammate
- Phase 3
- Phase 4
- Phase 5
- Matching
- Optimization
- Decision layer

### Shared
- Evaluation
- Error analysis
- Integration
- Final validation
- Submission


## Integration Rule

Each phase communicates through stable file contracts.

Phase 3 must consume Phase 2 output.

Phase 4 must consume Phase 3 output.

Phase 5 must consume Phase 4 output.

Evaluation reads outputs but should not secretly modify
the prediction pipeline.


## Git Workflow

Each teammate works in their own branch.

Example:

meet/phase1-phase2
teammate/phase3-phase4-phase5

Do not directly edit another person's phase unless explicitly agreed.

Before merging:

1. Pull latest main.
2. Rebase or merge main into your branch.
3. Run tests.
4. Run the pipeline on a small sample.
5. Run validate_submission.py.
6. Open a pull request.
7. Review changed files.
8. Merge only after integration passes.


## Important

Never commit:

- Competition datasets
- Large generated candidate files
- Generated model binaries
- Experiment logs containing huge artifacts
- Secrets
- AWS credentials
