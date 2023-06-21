"""Generate soap features from raw xyz files, using the yvalues
obtained from molecule_net as labels
(we do this for consistency, to be certain we are using the same
y-values as other publications).
The xyz files need to be 'cleaned' first (i.e. remove comment lines
at the end of the xyz file)."""
import os
from ase.io import read
import numpy as np
import pandas as pd
from dscribe.descriptors import SOAP

BATCH_SIZE = 500
atom_key = {"C":0, "N":1, "O":2, "F":3, "H":4}


def get_yvalue_dict():
    """Build a dictionary mapping molecule ids to all
    of the y-values of interest."""
    molnet_data = pd.read_csv("qm9_.csv")
    yvalues = molnet_data["u298_atom"].values
    yvalue_dict = dict(zip([int(z.split("_")[1])
            for z in molnet_data["mol_id"].tolist()],
            list(yvalues) ))
    return yvalue_dict


def get_id(fname):
    """Extracts the id number for a molecule
    from the xyz file."""
    with open(fname, "r") as fhandle:
        _ = fhandle.readline()
        metadata = fhandle.readline().strip().split()
    return int(metadata[1])



def partition_input_filelist(input_filelist, batch_name):
    """Breaks an input filelist up into subgroups of
    file lists, each containing only molecules with a set
    number of atoms. Mols in QM9 have up to 29 atoms."""

    print(f"Now loading the input files for group {batch_name}...")
    molgroups = {k:[] for k in range(1,30,1)}
    for xyz_file in input_filelist:
        struct = read(xyz_file)
        n_atoms = len(struct.get_chemical_symbols())
        molgroups[n_atoms].append((xyz_file, struct))
    return molgroups




def featurize_split(input_filelist, yval_dict, batch_name):
    """Generates .npy files containing 'chunked' data
    for the input list of xyz files.
    Args:
        input_filelist (list): A list of xyz files.
        yval_dict (dict): A dictionary mapping mol ids
            to y-values.
        batch_name (str): Train, valid or test.
    """
    molgroups = partition_input_filelist(input_filelist, batch_name)

    species = {"C", "H", "O", "N", "F"}

    soaper = SOAP(species=species,
                periodic=False,
                sigma = 0.25,
                n_max=8, l_max=5,
                weighting={"function":"pow",
                    "r0":1.5, "m":8, "c":1, "d":1},
                average = "m1n1_compression",
                r_cut = 4,
                sparse=False)
    batchnum = 0
    for _, molgroup in molgroups.items():
        if len(molgroup) == 0:
            continue
        structs, yvals = [], []
        for (filename, struct)  in molgroup:
            mol_id = get_id(filename)
            structs.append(struct)
            yvals.append(yval_dict[mol_id])
            if len(structs) >= BATCH_SIZE:
                blend_soap(soaper, structs, yvals, batchnum)
                structs, yvals = [], []
                batchnum += 1
        if len(structs) > 0:
            blend_soap(soaper, structs, yvals, batchnum)
            batchnum += 1


def blend_soap(soaper, structs, yvals, batchnum):
    """Performs the actual work of converting a minibatch of molecules
    into soap descriptors and saving to file.
    Args:
        soaper: The object that will generate the SOAP descriptors for
            use by convolution kernels.
        structs (list): A list of ase atoms objects containing the
            molecules to be processed.
        yvals (list): A list of shape M arrays where M is the number
            of properties we are predicting.
        batchnum (int): The number of this minibatch; used to generate
            the output filename.
    """
    xmats = soaper.create(structs, n_jobs=1)
    if len(xmats.shape) == 2:
        xmats = xmats.reshape((1,xmats.shape[0], xmats.shape[1]))

    xmats /= np.linalg.norm(xmats, axis=2)[:,:,None]
    ydata = np.array(yvals)

    print(f"Saving another batch, size {xmats.shape}...", flush=True)
    np.save(f"qm9_{batchnum}_xfolded.npy", xmats)
    np.save(f"qm9_{batchnum}_yvalues.npy", ydata)



def featurize_xyzfiles(target_dir, chemdata_path):
    """Obtains a list of xyz files for a target directory, splits them
    up into train - valid - test, and sets them up for feature generation."""

    bad_mols = set()
    with open("inconsistent_geom_mols.txt", "r") as fhandle:
        for line in fhandle:
            bad_mols.add(int(line.split()[0]))

    os.chdir(chemdata_path)
    yval_dict = get_yvalue_dict()
    os.chdir("cleaned_qm9_mols")
    raw_xyz_files = [os.path.abspath(f) for f in os.listdir() if f.endswith("xyz")]
    xyz_files = [xyz for xyz in raw_xyz_files if get_id(xyz) not in bad_mols]
    xyz_files.sort()
    print(f"There are {len(xyz_files)} files.")
    rng = np.random.default_rng(123)

    idx = rng.permutation(len(xyz_files))
    cutoff_valid, cutoff_test = 110000, 120000
    cv_splits = [idx[:cutoff_valid], idx[cutoff_valid:cutoff_test],
                    idx[cutoff_test:]]
    cv_splits = [[xyz_files[i] for i in s.tolist()] for s in cv_splits]

    os.chdir(target_dir)

    for batch_dir, cv_split in zip(["train", "valid", "test"], cv_splits):
        if batch_dir not in os.listdir():
            os.mkdir(batch_dir)
        os.chdir(batch_dir)
        featurize_split(cv_split, yval_dict, batch_dir)
        os.chdir("..")

    print("All done.")

if __name__ == "__main__":
    start_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(os.path.join(start_dir, "..", "qm9_data"))
    if "qm9_soap_features" not in os.listdir():
        os.mkdir("qm9_soap_features")
    output_dir = os.path.join(os.getcwd(), "qm9_soap_features")
    featurize_xyzfiles(output_dir, os.getcwd())
