#!/usr/bin/env python3

import sys
import os
import subprocess
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
import pyrosetta
import ViennaRNA

def progress(message):
    with open("progress.txt", "a") as f:
        f.write(message + '\n')

def console(message):
    print("----------------------------------------------------------------------------------------")
    print(message)
    print("****************************************************************************************")

def run_command(command, error_message):
    print(f"command: {command}")
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        progress(error_message)
        progress(result.stderr)
        print(error_message)
        print(result.stderr)
        sys.exit(1)
    progress(result.stdout)
    return result

def setup_paths():
    console("Setting up paths")
    rosetta_path = "/home/gxs/rosetta/rosetta.source.release-371"
    pyrossetta_path = "/home/gxs/PyRosetta"
    rna_tools_path = f"{rosetta_path}/main/tools/rna_tools"

    exportcmd = (
        f"export ROSETTA={rosetta_path} && "
        f"export ROSETTA3={rosetta_path}/main/source && "
        f"export PYROSETTA={pyrossetta_path} && "
        f"export PATH=$PATH:{rosetta_path}/main/source/bin:{rna_tools_path}/bin && "
        f"export PYTHONPATH=$PYTHONPATH:{rna_tools_path}/bin/ && "
        f"export PYTHONPATH=$PYTHONPATH:{pyrossetta_path}/python && "
        f"python {rna_tools_path}/sym_link.py"
    )

    return exportcmd

def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)
        console(f"Directory created: {path}")
    else:
        console(f"Directory already exists: {path}")

def transcribe_sequence(sequence):
    if 'T' in sequence:
        console("Transcribing DNA to RNA")
        progress('2')
        return Seq(sequence).transcribe()
    return Seq(sequence)

def save_fasta(sequence, filename):
    console("Saving sequence in FASTA format")
    progress('3')
    rec = SeqRecord(seq=sequence, id="aptamer", name='aptamer1', description='Sequence DNA')
    try:
        SeqIO.write(rec, filename, "fasta")
        console(f"FASTA file saved: {filename}")
    except Exception as e:
        progress(f"!FASTA saving error: {e}")
        sys.exit(1)

def search_secondary_structure(sequence, sec_struct, fasta_file):
    if len(sec_struct) != len(sequence):
        console("Predicting secondary structure")
        progress('4')
        result = run_command(f"RNAfold < {fasta_file} -o dotbracket", "!RNAfold error")
        with open("dotbracket_aptamer.fold", "r") as f:
            data = f.readlines()
        primary_structure = data[0].strip()
        secondary_structure = data[1].strip()
        with open("aptamer.dtb", "w") as f:
            f.write(secondary_structure + '\n')
        os.remove("dotbracket_aptamer.fold")
        console("Secondary structure predicted and saved")
        return primary_structure, secondary_structure
    else:
        with open("aptamer.dtb", "w") as f:
            f.write(sec_struct + '\n')
        console("Secondary structure loaded from input")
        return sequence, sec_struct

def identify_helices(exportcmd, path):
    console("Identifying helices")
    progress('5')
    command = f"{exportcmd} && python $ROSETTA/main/tools/rna_tools/job_setup/helix_preassemble_setup.py -secstruct {path}/aptamer.dtb -fasta {path}/aptamer.fa"
    run_command(command, "!No CMDLINES file created")
    console("Helices identified")

def preprocess_helices(exportcmd, path):
    console("Preprocessing helices")
    progress('6')
    run_command(f"{exportcmd} && /bin/bash {path}/CMDLINES", "!Unable to run CMDLINES")
    console("Helices preprocessed")

def compute_structure(exportcmd, path, primary_structure, ncycles, option):
    console("Computing structure (this may take a while)")
    progress('7')
    command = f"{exportcmd} && $ROSETTA3/bin/rna_denovo.linuxgccrelease -fasta {path}/aptamer.fa -secstruct_file {path}/aptamer.dtb -fixed_stems -tag aptamer -working_res 1-{len(primary_structure)} -cycles {ncycles} -ignore_zero_occupancy false -include_neighbor_base_stacks -minimize_rna false {option}"
    run_command(command, "!Computation failed")
    console("Structure computed")

def minimize_structure(exportcmd, path):
    console("Minimizing structure")
    progress('8')
    command = f"{exportcmd} && $ROSETTA/main/tools/rna_tools/job_setup/parallel_min_setup.py -silent {path}/aptamer.out -tag aptamer_min -nstruct 100 -out_folder {path}/min -out_script {path}/min_cmdline ''"
    run_command(command, "!No minimization command-lines created")
    run_command(f"/bin/bash {path}/min_cmdline", "!Unable to run minimization")
    console("Structure minimized")

def select_best_structures(exportcmd, path):
    console("Selecting the 5 best structures")
    progress('9')
    command = f"{exportcmd} && $ROSETTA/main/tools/rna_tools/silent_util/silent_file_sort_and_select.py {path}/min/0/aptamer_min.out -select 1-5 -o {path}/aptamer_best.out"
    run_command(command, "!Unable to select the best files")
    console("Best structures selected")

def convert_to_pdb(exportcmd, path):
    console("Converting to PDB")
    progress('10')
    command = f"{exportcmd} && $ROSETTA/main/tools/rna_tools/silent_util/extract_lowscore_decoys.py {path}/aptamer_best.out 5"
    run_command(command, "!Cannot convert .out to .pdb")
    console("Converted to PDB")

def convert_to_dna(exportcmd, path):
    console("Converting to DNA")
    progress('11')
    for k in range(1, 6):
        with open(f"{path}/aptamer_best.out.{k}.pdb", "r") as f:
            atoms = f.readlines()
        output = []
        for atom in atoms:
            if "O2'" not in atom:
                if ('H' in atom[-5:] and "U" in atom and "H5 " in atom):
                    line = atom.split()
                    line[2] = 'C7'
                    line[-1] = 'C'
                    atom = ' '.join(line)
                if "U" in atom:
                    atom = atom.replace("U", "T")
            output.append(atom)
        with open(f"{path}/pre-DNA_aptamer_best.out.{k}.pdb", "w") as f:
            f.writelines(output)
        os.remove(f"{path}/aptamer_best.out.{k}.pdb")

    console("Re-minimizing DNA structures")
    command = f"{exportcmd} && $ROSETTA3/bin/score.linuxgccrelease -in:file:s {path}/pre-DNA_aptamer_best.out.*.pdb -no_optH false -output"
    run_command(command, "!Cannot minimize the converted to DNA file")
    os.remove(f"{path}/default.sc")

    for k in range(1, 6):
        os.remove(f"{path}/pre-DNA_aptamer_best.out.{k}.pdb")
    console("DNA structures converted and minimized")

def main():
    if len(sys.argv) < 6:
        print("SYNTAX: ./1.py SEQ folder returnDNA sectruct ncycles")
        sys.exit(0)

    sequence = sys.argv[1]
    folder = sys.argv[2]
    return_dna = int(sys.argv[3])
    sec_struct = sys.argv[4]
    ncycles = int(sys.argv[5])

    path = os.path.expanduser(f"~/aptamers/{folder}")
    create_directory(path)
    os.chdir(path)

    with open("progress.txt", "w") as f:
        f.write("1\n")

    exportcmd = setup_paths()

    try:
        sequence_rna = transcribe_sequence(sequence)
        save_fasta(sequence_rna, "aptamer.fa")
        primary_structure, secondary_structure = search_secondary_structure(sequence, sec_struct, "aptamer.fa")
        identify_helices(exportcmd, path)
        preprocess_helices(exportcmd, path)
        compute_structure(exportcmd, path, primary_structure, ncycles, option="")
        minimize_structure(exportcmd, path)
        select_best_structures(exportcmd, path)
        convert_to_pdb(exportcmd, path)
        
        if return_dna:
            convert_to_dna(exportcmd, path)

        console("Completed.")
        progress('ok')

    except Exception as e:
        console(f"Error: {e}")
        progress(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
