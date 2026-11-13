# theoros-models requirements

| ID | Requirement | Verification |
|---|---|---|
| MOD-REQ-001 | A flight model shall have fewer than 1,000,000 parameters after BatchNorm folding and D4 expansion. | Analysis, CI |
| MOD-REQ-002 | Its int8 weights and int32 biases shall fit in 1 MiB. | Analysis, CI |
| MOD-REQ-003 | The same trained weights shall compile for any sensor with 3 to 13 bands between 400 and 2400 nm. | Test |
| MOD-REQ-004 | A head shall be disabled when the sensor lacks the spectral bands its task physically needs. | Test |
| MOD-REQ-005 | The compiled network shall use only the operators in the flight operator set. | Test |
| MOD-REQ-006 | The compiled network shall match the training network within 1e-4 before quantisation. | Test |
| MOD-REQ-007 | The lift layer shall be exactly equivariant to the 8 symmetries of the square. | Test |
| MOD-REQ-008 | Every released model shall ship a model card with accuracy, size and latency per target. | Inspection |

Status: baseline, November 2026. Changes need approval from the CTO.
