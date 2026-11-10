# Attribution

This is the official GLVQ reimplementation for open-source release. The
original implementation used an internal company framework; this distribution
redevelops the method using the public engineering base identified below.
The core learnable lattice is included. Bit allocation and PV-Tuning are not
included in this release.

This distribution derives from AQLM (https://github.com/Vahe1994/AQLM),
upstream commit e79a896, under the Apache License 2.0 retained in LICENSE.

Local additions and modifications include the GLVQ lattice implementation,
Hadamard transforms, companding, activation-aware fitting, fixed-code block
and end-to-end fine-tuning, checkpoint handling, data preparation and evaluation.
The distribution adds a portable fixed-recipe launcher and documentation.
The release retains the AQLM-derived calibration, data and distributed-training
infrastructure. The exported interface supports the fixed GLVQ E8P/full recipe.

No model weights or datasets are distributed. Their respective licenses apply
when obtained separately.
