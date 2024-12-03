import subprocess
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

def run_subprocess(command, input_data=None):
    result = subprocess.run(command, input=input_data, text=True, capture_output=True)
    if result.returncode != 0:
        print(f"Error running command: {' '.join(command)}")
        print(result.stderr)
    return result.stdout

def predict_rna_structure(rna_sequence):
    # 使用RNAfold生成MFE结构和配对概率图
    fold_output = run_subprocess(["RNAfold", "--noPS"], input_data=rna_sequence)
    
    ss = ""
    mfe = 0.0
    for line in fold_output.split('\n'):
        if line.startswith(rna_sequence):
            continue
        if line.startswith("("):
            pos = line.find(' ')
            ss = line[0: pos]
            mfe = float(line[pos+1: ].strip("()"))

    print(f"Predicted Secondary Structure: {ss}")
    print(f"Minimum Free Energy (MFE): {mfe} kcal/mol")

    # 使用RNAsubopt生成次优结构
    subopt_output = run_subprocess(["RNAsubopt", "-e", "5.0"], input_data=rna_sequence)
    subopts = []
    for line in subopt_output.split('\n'):
        if line and not line.startswith(">"):
            parts = line.split()
            if len(parts) == 2:
                subopts.append((parts[0], float(parts[1])))

    # 生成二级结构图
    output_file = "rna_structure.ps"
    plot_rna_structure(rna_sequence, ss, output_file)

    return ss, mfe, subopts

def plot_rna_structure(rna_sequence, secondary_structure, output_file):
    # 创建RNAplot输入文件
    with open("rna-sec.txt", "w") as f:
        f.write(f"{rna_sequence}\n{secondary_structure}\n")
    
    # 生成二级结构图
    run_subprocess(["RNAplot"], input_data=open("rna-sec.txt").read())
    
    # 确保生成的文件存在于当前目录
    if os.path.exists("rna.ps"):
        os.rename("rna.ps", output_file)
    
    # 使用Ghostscript将PS文件转换为PNG
    png_output_file = output_file.replace(".ps", ".png")
    run_subprocess(["gs", "-sDEVICE=pngalpha", "-o", png_output_file, "-r144", output_file])

    # 加载并显示图片
    img = mpimg.imread(png_output_file)
    plt.imshow(img)
    plt.axis('off')
    plt.show()

def main():
    # 输入RNA序列列表
    rna_sequences = [
        "GCGCUUCGCCGAUCCAGCAGCCGUGCGC",
        # 可以添加更多RNA序列
    ]
    
    for i, rna_sequence in enumerate(rna_sequences):
        print(f"\nProcessing RNA sequence {i+1}/{len(rna_sequences)}: {rna_sequence}")
        ss, mfe, subopts = predict_rna_structure(rna_sequence)
        
        # 输出次优结构
        print("Suboptimal Structures within 5 kcal/mol:")
        for structure, energy in subopts:
            print(f"Structure: {structure}, Energy: {energy} kcal/mol")

if __name__ == "__main__":
    main()
