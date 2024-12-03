import subprocess
import os
import itertools

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
        for line in lines:
            if "frequency of mfe structure in ensemble" in line:
                mfe_freq = line.strip().split()[-1].strip(',')
            if "delta G binding" in line:
                try:
                    delta_g_binding = float(line.strip().split('=')[-1].strip())
                except ValueError:
                    delta_g_binding = None
    return mfe_freq, delta_g_binding

def run_rna_plot(output_dir, plot_id):
    for file in os.listdir(output_dir):
        if file.endswith("_ss.ps") or file.endswith("_dp.ps"):
            ps_file = os.path.join(output_dir, file)
            plot_output = os.path.join(output_dir, f"{plot_id}_{file.replace('.ps', '.png')}")
            subprocess.run(f"RNAplot < {ps_file}", shell=True)
            pdf_output = plot_output.replace(".png", ".pdf")
            subprocess.run(f"ps2pdf {ps_file} {pdf_output}", shell=True)
            subprocess.run(f"convert -density 150 {pdf_output} -quality 90 {plot_output}", shell=True)

def batch_process_rna_pairs(rna_pairs, output_dir):
    results = []
    for i, (seq_a, seq_b) in enumerate(rna_pairs):
        pair_output_dir = os.path.join(output_dir, f"pair_{i+1}")
        cofold_output = run_rna_cofold(seq_a, seq_b, pair_output_dir)
        mfe_freq, delta_g_binding = parse_cofold_output(cofold_output)
        
        if delta_g_binding is not None:
            results.append((seq_a, seq_b, mfe_freq, delta_g_binding, pair_output_dir))
    
    results.sort(key=lambda x: x[3])
    
    return results[:5]

def main(rna_sequences, output_dir):
    rna_pairs = list(itertools.combinations(rna_sequences, 2))
    results = batch_process_rna_pairs(rna_pairs, output_dir)
    
    if results:
        with open(os.path.join(output_dir, 'results.txt'), 'w') as f:
            for idx, (seq_a, seq_b, mfe_freq, delta_g_binding, pair_output_dir) in enumerate(results, start=1):
                f.write(f"Pair: {seq_a} & {seq_b}\n")
                f.write(f"Frequency of MFE structure: {mfe_freq}\n")
                f.write(f"Delta G binding: {delta_g_binding}\n")
                f.write(f"Output Directory: {pair_output_dir}\n\n")
                
                run_rna_plot(pair_output_dir, f"pair_{idx}")
            
        print("Processing complete. Results saved to results.txt.")
    else:
        print("No valid results found.")

if __name__ == "__main__":
    rna_sequences = [
        "GCGCUUCGCCGCGCGCC",
        "GCUAGCAUGCUACGUCAAGCUAGCUAUGCAUA",
        "AUCGUAGCUAGUAUGCAUCGUAGUCUGACGUAUGCUGCAUGCUAGCUA"
    ]
    output_dir = "rna_cofold_results"
    main(rna_sequences, output_dir)
