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

This demo runs on Linux only, with a GPU with CUDA >= 10.0 and with Python 3.9 or better.

To run this demo, with a venv active, run the `install.sh` script. Next, run `setup_qm9.sh`. This will
download the qm9 dataset and generate the SOAP features. Finally,
use the Jupyter "qm9_fitting" notebook under notebooks. Note that dowloading the
qm9 dataset and generating the features can take a minute and -- once all features are generated --
close to 50 GB of disk space.
