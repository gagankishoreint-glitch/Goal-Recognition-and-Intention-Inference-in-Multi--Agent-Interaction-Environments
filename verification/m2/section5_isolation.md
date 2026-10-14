# End-to-End Information Isolation (Section 5)

Pipeline exercised: scenario sampling -> policy reset -> trajectory generation ->
observable/privileged separation -> Condition A -> Condition B projection ->
JSON serialization -> reload -> batch validation -> model input tensor.

Assertions (all implemented in `verification/m2/section5_isolation_test.py`):

- no privileged field in a Condition A batch
- no privileged field and no peer-derived field in a Condition B batch
- Condition B schema is a strict subset of the Condition A schema
- injecting a privileged field raises `InformationIsolationError`
- serialized observable files contain no privileged key names (grep)
- observations up to t do not depend on the suffix after t
- a model input tensor cannot carry privileged or peer columns in Condition B

Result: PASS (141 passed in 160.70s (0:02:40))
