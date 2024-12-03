import subprocess
import os
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO

# 定义二进制文件和脚本的路径
BIN_PATH = "/home/gxs/rosetta/rosetta.source.release-371/main/source/bin"
TOOLS_PATH = "/home/gxs/rosetta/rosetta.source.release-371/main/tools/rna_tools"

def run_subprocess(command, input_data=None):
    result = subprocess.run(command, input=input_data, text=True, capture_output=True)
    if result.returncode != 0:
        print(f"Error running command: {' '.join(command)}")
        print(result.stderr)
    return result.stdout

def save_fasta(sequence, file_name):
    record = SeqRecord(Seq(sequence), id="RNA_sequence")
    SeqIO.write(record, file_name, "fasta")
    print(f"FASTA file saved: {file_name}")

def predict_secondary_structure(fasta_file):
    print("Predicting secondary structure...")
    result = run_subprocess(["RNAfold", fasta_file])
    ss = ""
    mfe = 0.0
    for line in result.split('\n'):
        if line.startswith(">") or line.strip() == "":
            continue
        if '(' in line and ')' in line:
            ss = line.split()[0]
            mfe = float(line.split()[1].strip("()"))
    print(f"Predicted Secondary Structure: {ss}")
    print(f"Minimum Free Energy (MFE): {mfe} kcal/mol")
    return ss

def identify_helices(fasta_file, ss):
    print("Identifying helices...")
    helix_setup_script = os.path.join(TOOLS_PATH, "job_setup/helix_preassemble_setup.py")
    ss_file = fasta_file.replace(".fasta", ".ss")
    with open(ss_file, "w") as f:
        f.write(f">{fasta_file}\n{ss}\n")
    result = run_subprocess(["python3", helix_setup_script, ss_file])
    helices = []
    for line in result.split('\n'):
        if line.startswith("Helix"):
            start, end = map(int, line.split()[2:4])
            helices.append((start, end))
    print(f"Identified helices: {helices}")
    return helices

def compute_structure(fasta_file, ss):
    print("Computing RNA structure...")
    de_novo_bin = os.path.join(BIN_PATH, "rna_denovo.linuxgccrelease")
    command = [de_novo_bin, "-s", fasta_file, "-secstruct", ss]
    run_subprocess(command)
    print(f"Structure computation completed for {fasta_file}")

def minimize_structure(fasta_file):
    print("Minimizing structure...")
    min_setup_script = os.path.join(TOOLS_PATH, "job_setup/parallel_min_setup.py")
    command = ["python3", min_setup_script, fasta_file]
    run_subprocess(command)
    print(f"Structure minimization completed for {fasta_file}")

def select_best_structures(fasta_file):
    print("Selecting best structures...")
    sort_select_script = os.path.join(TOOLS_PATH, "silent_util/silent_file_sort_and_select.py")
    silent_file = fasta_file.replace(".fasta", ".out")
    command = ["python3", sort_select_script, silent_file]
    run_subprocess(command)
    print(f"Best structure selection completed for {fasta_file}")

def extract_lowscore_decoys(fasta_file):
    print("Extracting lowest score decoys...")
    extract_script = os.path.join(TOOLS_PATH, "silent_util/extract_lowscore_decoys.py")
    silent_file = fasta_file.replace(".fasta", ".out")
    command = ["python3", extract_script, silent_file, "-pdb"]
    run_subprocess(command)
    print(f"Extraction of lowest score decoys completed for {fasta_file}")

def main():
    rna_sequences = [
        "GCGCUUCGCGC",
        # 添加更多RNA序列
    ]
    
    for i, rna_sequence in enumerate(rna_sequences):
        print(f"\nProcessing RNA sequence {i+1}/{len(rna_sequences)}: {rna_sequence}")
        fasta_file = f"rna_sequence_{i+1}.fasta"
        save_fasta(rna_sequence, fasta_file)

        ss = predict_secondary_structure(fasta_file)
        helices = identify_helices(fasta_file, ss)

        # 计算结构
        compute_structure(fasta_file, ss)
        minimize_structure(fasta_file)
        select_best_structures(fasta_file)
        extract_lowscore_decoys(fasta_file)

        print(f"Completed processing for RNA sequence {i+1}")

if __name__ == "__main__":
    main()
