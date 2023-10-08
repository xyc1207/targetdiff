import os
import argparse
import multiprocessing as mp
import pickle
import shutil
import pandas as pd
import urllib.request

from functools import partial
from tqdm.auto import tqdm
from utils.data import PDBProtein, parse_sdf_file
from openbabel import openbabel 
from Bio.PDB import PDBParser, PDBIO, Select


def is_het(residue, name=None):
    res = residue.id[0]
    if name is None:
        return res != " " and res != "W"
    return res != " " and res != "W" and name.lower() in residue.id[0].lower()


def convert_pdb_to_sdf(pdb_file, sdf_file):  
    obConversion = openbabel.OBConversion()  
    obConversion.SetInAndOutFormats("pdb", "sdf")  
  
    mol = openbabel.OBMol()  
    obConversion.ReadFile(mol, pdb_file)  
    obConversion.WriteFile(mol, sdf_file)  


class ResidueSelect(Select):
    # the code is from https://stackoverflow.com/questions/61390035/how-to-save-each-ligand-from-a-pdb-file-separately-with-bio-pdb
    def __init__(self, chain, residue):
        self.chain = chain
        self.residue = residue

    def accept_chain(self, chain):
        return chain.id == self.chain.id

    def accept_residue(self, residue):
        """ Recognition of heteroatoms - Remove water molecules """
        return residue == self.residue and is_het(residue)


def extract_ligands(pdb_file, ligand_file, ligand_name=None):
    """ Extraction of the heteroatoms of .pdb files """
    ret_ligand_fn = []
    i = 0
    pdb_code = pdb_file[:-4]
    pdb = PDBParser().get_structure(pdb_code, pdb_file)
    io = PDBIO()
    io.set_structure(pdb)
    for model in pdb:
        for chain in model:
            for residue in chain:
                if not is_het(residue, ligand_name):
                    continue
                print(f"saving {chain} {residue}")
                pdb_ligand_fn = f"/tmp/tmppdb.pdb"
                io.save(pdb_ligand_fn, ResidueSelect(chain, residue))
                lfn = ligand_file.replace("ligand.sdf", f"ligand{i}.sdf")
                ret_ligand_fn.append(lfn)
                convert_pdb_to_sdf(pdb_ligand_fn, lfn)  
                i += 1
    return ret_ligand_fn


def load_item(item):
    pdb_path = item[0]
    sdf_path = item[1]
    with open(pdb_path, 'r') as f:
        pdb_block = f.read()
    with open(sdf_path, 'r') as f:
        sdf_block = f.read()
    return pdb_block, sdf_block


def process_item(item, args):
    try:
        pdb_block, sdf_block = load_item(item)
        protein = PDBProtein(pdb_block)
        # ligand = parse_sdf_block(sdf_block)
        ligand = parse_sdf_file(item[1])

        pdb_block_pocket = protein.residues_to_pdb_block(
            protein.query_residues_ligand(ligand, args.radius)
        )
        
        ligand_fn = item[1]
        pocket_fn = ligand_fn[:-4] + '_pocket%d.pdb' % args.radius
        ligand_dest = os.path.join(args.dest, os.path.basename(ligand_fn))
        pocket_dest = os.path.join(args.dest, os.path.basename(pocket_fn))
        os.makedirs(os.path.dirname(ligand_dest), exist_ok=True)

        shutil.copyfile(
            src=ligand_fn,
            dst=os.path.join(args.dest, os.path.basename(ligand_fn))
        )
        with open(pocket_dest, 'w') as f:
            f.write(pdb_block_pocket)
        return pocket_fn, ligand_fn, item[0], item[2]  # item[0]: original protein filename; item[2]: rmsd.
    except Exception:
        print('Exception occurred.', item)
        return None, item[1], item[0], item[2]


def retrieve_pdb(pdbfolder, pdbid):
    pdbid = pdbid.lower()
    fn = os.path.join(pdbfolder, pdbid +".pdb")
    if os.path.exists(fn):
        print(fn, 'exists')
        return fn
    urllib.request.urlretrieve(f"http://files.rcsb.org/download/{pdbid}.pdb", fn)
    return fn


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdblist', type=str, default=r"./examples/pdb_list.csv")
    parser.add_argument('--pdbfolder', type=str, default=r"./data/pdbfolder")
    parser.add_argument('--dest', type=str, required=True)
    parser.add_argument('--radius', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=1)
    args = parser.parse_args()

    os.makedirs(args.pdbfolder, exist_ok=True)
    df = pd.read_csv(args.pdblist)
    
    for idx, row in df.iterrows():
        receptor_fn = retrieve_pdb(args.pdbfolder, row['pdb_id'])
        ligand_fn_sdf = receptor_fn.replace(".pdb", ".ligand.sdf")
        ret_ligand_fn = extract_ligands(receptor_fn, ligand_fn_sdf, row["ligand_id"])
        for fn in ret_ligand_fn:
            item = [receptor_fn, fn, 0.0]
            process_item(item, args)