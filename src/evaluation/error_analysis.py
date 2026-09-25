"""
Error analysis.

Separate errors into:

1. Blocking failures
   - true pair never entered candidate set

2. Matching failures
   - candidate existed but matcher scored it incorrectly

3. Decision failures
   - matcher score was reasonable but final threshold/margin
     decision produced the wrong output

4. Ambiguous entities
5. Hard negatives
6. False merges
7. Missed matches
"""
