import subprocess
import os

def write_sequence_file(sequence_a, sequence_b, file_path):
    with open(file_path, 'w') as f:
        f.write(">t\n")
        f.write(f"{sequence_a}&{sequence_b}\n")

def write_concentration_file(file_path):
    with open(file_path, 'w') as f:
        c = 1e-07
        while c < 0.2:
            f.write(f"{c}\t{c}\n")
            c *= 1.71

def run_rna_cofold(sequence_a, sequence_b, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    sequence_file = os.path.join(output_dir, 't.seq')
    conc_file = os.path.join(output_dir, 'concfile')
    cofold_output = os.path.join(output_dir, 'cofold.out')

    write_sequence_file(sequence_a, sequence_b, sequence_file)
    write_concentration_file(conc_file)
    
    # Run RNAcofold
    subprocess.run(f"RNAcofold -p < {sequence_file}", shell=True)
    subprocess.run(f"RNAcofold -f {conc_file} < {sequence_file} > {cofold_output}", shell=True)
    
    return cofold_output

def parse_cofold_output(output_file):
    mfe_freq = None
    delta_g_binding = None
    with open(output_file, 'r') as f:
        lines = f.readlines()
        # Example of parsing some of the relevant results
        for line in lines:
            if line.startswith('frequency of mfe structure in ensemble'):
                mfe_freq = line.strip().split()[-1]
                print(f"Frequency of MFE structure: {mfe_freq}")
            if line.startswith('delta G binding'):
                delta_g_binding = line.strip().split('=')[-1]
                print(f"Delta G binding: {delta_g_binding}")
    return mfe_freq, delta_g_binding

def batch_process(rna_pairs, output_dir):
    results = []
    for i, (seq_a, seq_b) in enumerate(rna_pairs):
        pair_output_dir = os.path.join(output_dir, f"pair_{i+1}")
        cofold_output = run_rna_cofold(seq_a, seq_b, pair_output_dir)
        mfe_freq, delta_g_binding = parse_cofold_output(cofold_output)
        results.append((cofold_output, mfe_freq, delta_g_binding))
    return results

def run_rna_plot(output_dir):
    # Find the generated PostScript files
    for file in os.listdir(output_dir):
        if file.endswith("_ss.ps") or file.endswith("_dp.ps"):
            ps_file = os.path.join(output_dir, file)
            subprocess.run(f"RNAplot < {ps_file}", shell=True)

def batch_process_with_plot(rna_pairs, output_dir):
    results = []
    for i, (seq_a, seq_b) in enumerate(rna_pairs):
        pair_output_dir = os.path.join(output_dir, f"pair_{i+1}")
        cofold_output = run_rna_cofold(seq_a, seq_b, pair_output_dir)
        mfe_freq, delta_g_binding = parse_cofold_output(cofold_output)
        results.append((cofold_output, mfe_freq, delta_g_binding))
        
        # Run RNAplot to visualize the results
        run_rna_plot(pair_output_dir)
    
    return results

if __name__ == "__main__":
    rna_pairs = [
        ("GCGCUUCGCCGCGCGCC", "GCUAGCAUGCUACGUCAAGCUAGCUAUGCAUA"),
        # Add more RNA pairs here
    ]
    output_dir = "rna_cofold_results"
    results = batch_process_with_plot(rna_pairs, output_dir)
