# dscribe_qm9_testing
Preliminary tests on modifications to the dscribe library using the QM9 dataset (and optionally other datasets)

This repository contains some quick-and-dirty tests to modifications to the dscribe library, specifically
to the SOAP features, that incorporate various forms of feature compression. The SOAP features are useful as
representations of atomic environments for input to deep learning and / or Gaussian process models.
The number of features in a  SOAP descriptor vector typically scales quadratically with the number of
elements present and with the number
of radial basis functions, resulting in feature vectors of extraordinary and impractical length for
datasets with many different elements (e.g. SPICE) or where a large # of radial basis functions is necessary
for high accuracy. We hope here to show how to avoid this problem with modifications to dscribe.

### Installation and use

To run this demo, with a Python 3.9 venv active, run the `install.sh` script. Next, run `construct_features.sh`. Finally,
use the Jupyter "qm9_fitting" notebook. Note that dowloading the qm9 dataset and generating the features can take
a minute.
