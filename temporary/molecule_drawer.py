import tkinter as tk
from tkinter import messagebox, Canvas, Button, Label, Frame, filedialog
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw, AllChem, inchi
from PIL import Image, ImageTk
import io
import os

# 尝试导入pyperclip，如果不存在则提供替代方案
try:
    import pyperclip  # 用于复制到剪贴板
    PYPERCLIP_AVAILABLE = True
except ImportError:
    PYPERCLIP_AVAILABLE = False
    print("警告: pyperclip库未安装，复制到剪贴板功能将不可用")
    print("可以使用以下命令安装: pip install pyperclip")

class MoleculeDrawer:
    def __init__(self, root):
        self.root = root
        self.root.title("分子结构绘制器")
        self.root.geometry("1000x700")
        
        # 初始化分子数据
        self.atoms = []  # 存储原子 [(x, y, 元素符号), ...]
        self.bonds = []  # 存储键 [(atom1_idx, atom2_idx, 键类型), ...]
        self.current_element = "C"  # 默认元素
        self.current_bond_type = 1  # 默认键类型 (单键)
        self.selected_atom = None
        self.drawing_bond = False
        self.temp_line = None
        self.current_mol = None  # 存储当前分子对象
        self.current_smiles = ""  # 存储当前SMILES
        self.current_inchi = ""  # 存储当前InChI
        
        # 创建界面
        self.create_ui()
        
    def create_ui(self):
        # 主框架
        main_frame = Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 左侧工具栏
        tools_frame = Frame(main_frame, width=200, bg="#f0f0f0", padx=10, pady=10)
        tools_frame.pack(side=tk.LEFT, fill=tk.Y)
        
        # 元素选择
        Label(tools_frame, text="选择元素:", bg="#f0f0f0").pack(anchor=tk.W, pady=(0, 5))
        elements_frame = Frame(tools_frame, bg="#f0f0f0")
        elements_frame.pack(fill=tk.X, pady=(0, 10))
        
        common_elements = ["C", "H", "O", "N", "P", "S", "F", "Cl", "Br", "I"]
        row, col = 0, 0
        for elem in common_elements:
            btn = Button(elements_frame, text=elem, width=3, 
                         command=lambda e=elem: self.set_element(e))
            btn.grid(row=row, column=col, padx=2, pady=2)
            col += 1
            if col > 4:
                col = 0
                row += 1
        
        # 键类型选择
        Label(tools_frame, text="选择键类型:", bg="#f0f0f0").pack(anchor=tk.W, pady=(10, 5))
        bond_frame = Frame(tools_frame, bg="#f0f0f0")
        bond_frame.pack(fill=tk.X, pady=(0, 10))
        
        bond_types = [("单键", 1), ("双键", 2), ("三键", 3), ("芳香键", 4)]
        for i, (name, bond_type) in enumerate(bond_types):
            btn = Button(bond_frame, text=name, 
                         command=lambda bt=bond_type: self.set_bond_type(bt))
            btn.pack(fill=tk.X, pady=2)
        
        # 操作按钮
        actions_frame = Frame(tools_frame, bg="#f0f0f0")
        actions_frame.pack(fill=tk.X, pady=(10, 0))
        
        Button(actions_frame, text="清除画布", command=self.clear_canvas).pack(fill=tk.X, pady=2)
        Button(actions_frame, text="生成SMILES", command=self.generate_smiles).pack(fill=tk.X, pady=2)
        Button(actions_frame, text="生成InChI", command=self.generate_inchi).pack(fill=tk.X, pady=2)
        Button(actions_frame, text="生成2D图像", command=self.generate_2d_image).pack(fill=tk.X, pady=2)
        Button(actions_frame, text="复制SMILES", command=lambda: self.copy_to_clipboard(self.current_smiles)).pack(fill=tk.X, pady=2)
        Button(actions_frame, text="复制InChI", command=lambda: self.copy_to_clipboard(self.current_inchi)).pack(fill=tk.X, pady=2)
        Button(actions_frame, text="保存2D图像", command=self.save_2d_image).pack(fill=tk.X, pady=2)
        
        # SMILES显示区域
        Label(tools_frame, text="SMILES:", bg="#f0f0f0").pack(anchor=tk.W, pady=(10, 5))
        self.smiles_label = Label(tools_frame, text="", bg="white", wraplength=180, 
                                 justify=tk.LEFT, relief=tk.SUNKEN, height=3)
        self.smiles_label.pack(fill=tk.X, pady=(0, 10))
        
        # InChI显示区域
        Label(tools_frame, text="InChI:", bg="#f0f0f0").pack(anchor=tk.W, pady=(10, 5))
        self.inchi_label = Label(tools_frame, text="", bg="white", wraplength=180, 
                                justify=tk.LEFT, relief=tk.SUNKEN, height=3)
        self.inchi_label.pack(fill=tk.X, pady=(0, 10))
        
        # 绘图区域
        canvas_frame = Frame(main_frame, bg="white", padx=10, pady=10)
        canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.canvas = Canvas(canvas_frame, bg="white", width=700, height=600)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # 绑定鼠标事件
        self.canvas.bind("<Button-1>", self.canvas_click)
        self.canvas.bind("<Motion>", self.canvas_motion)
        
        # 2D分子图像显示区域
        self.image_label = Label(canvas_frame, bg="white")
        self.image_label.pack(side=tk.BOTTOM, pady=(10, 0))
        
    def set_element(self, element):
        self.current_element = element
        
    def set_bond_type(self, bond_type):
        self.current_bond_type = bond_type
        
    def canvas_click(self, event):
        x, y = event.x, event.y
        
        # 检查是否点击了现有原子
        clicked_atom = self.find_atom_at_position(x, y)
        
        if clicked_atom is not None:
            # 如果已经选择了一个原子，则创建一个键
            if self.selected_atom is not None and self.selected_atom != clicked_atom:
                # 检查这两个原子之间是否已经有键
                bond_exists = False
                for i, (a1, a2, _) in enumerate(self.bonds):
                    if (a1 == self.selected_atom and a2 == clicked_atom) or \
                       (a2 == self.selected_atom and a1 == clicked_atom):
                        # 更新现有键的类型
                        self.bonds[i] = (a1, a2, self.current_bond_type)
                        bond_exists = True
                        break
                
                if not bond_exists:
                    # 创建新键
                    self.bonds.append((self.selected_atom, clicked_atom, self.current_bond_type))
                
                # 重置选择状态
                self.selected_atom = None
                self.drawing_bond = False
                if self.temp_line:
                    self.canvas.delete(self.temp_line)
                    self.temp_line = None
                
                # 重绘画布
                self.redraw_canvas()
            else:
                # 选择这个原子开始画键
                self.selected_atom = clicked_atom
                self.drawing_bond = True
        else:
            # 在空白处添加新原子
            atom_idx = len(self.atoms)
            self.atoms.append((x, y, self.current_element))
            
            # 如果正在画键，则连接到新原子
            if self.drawing_bond and self.selected_atom is not None:
                self.bonds.append((self.selected_atom, atom_idx, self.current_bond_type))
                self.selected_atom = None
                self.drawing_bond = False
                if self.temp_line:
                    self.canvas.delete(self.temp_line)
                    self.temp_line = None
            
            # 重绘画布
            self.redraw_canvas()
    
    def canvas_motion(self, event):
        if self.drawing_bond and self.selected_atom is not None:
            # 更新临时线条
            x, y = event.x, event.y
            atom_x, atom_y, _ = self.atoms[self.selected_atom]
            
            if self.temp_line:
                self.canvas.delete(self.temp_line)
            
            self.temp_line = self.canvas.create_line(atom_x, atom_y, x, y, fill="gray", dash=(4, 4))
    
    def find_atom_at_position(self, x, y, tolerance=10):
        for i, (ax, ay, _) in enumerate(self.atoms):
            if abs(x - ax) <= tolerance and abs(y - ay) <= tolerance:
                return i
        return None
    
    def redraw_canvas(self):
        self.canvas.delete("all")
        
        # 绘制键
        for atom1_idx, atom2_idx, bond_type in self.bonds:
            x1, y1, _ = self.atoms[atom1_idx]
            x2, y2, _ = self.atoms[atom2_idx]
            
            if bond_type == 1:  # 单键
                self.canvas.create_line(x1, y1, x2, y2, width=2)
            elif bond_type == 2:  # 双键
                # 计算平行线的偏移
                dx = x2 - x1
                dy = y2 - y1
                length = (dx**2 + dy**2)**0.5
                offset = 3
                
                # 计算垂直于键的方向
                nx = -dy / length * offset
                ny = dx / length * offset
                
                # 绘制两条平行线
                self.canvas.create_line(x1 + nx, y1 + ny, x2 + nx, y2 + ny, width=2)
                self.canvas.create_line(x1 - nx, y1 - ny, x2 - nx, y2 - ny, width=2)
            elif bond_type == 3:  # 三键
                # 计算平行线的偏移
                dx = x2 - x1
                dy = y2 - y1
                length = (dx**2 + dy**2)**0.5
                offset = 4
                
                # 计算垂直于键的方向
                nx = -dy / length * offset
                ny = dx / length * offset
                
                # 绘制三条平行线
                self.canvas.create_line(x1, y1, x2, y2, width=2)
                self.canvas.create_line(x1 + nx, y1 + ny, x2 + nx, y2 + ny, width=2)
                self.canvas.create_line(x1 - nx, y1 - ny, x2 - nx, y2 - ny, width=2)
            elif bond_type == 4:  # 芳香键
                # 绘制虚线表示芳香键
                self.canvas.create_line(x1, y1, x2, y2, width=2, dash=(5, 3))
        
        # 绘制原子
        for i, (x, y, element) in enumerate(self.atoms):
            # 为不同元素设置不同颜色
            color = self.get_element_color(element)
            
            # 绘制原子圆圈
            radius = 15
            self.canvas.create_oval(x-radius, y-radius, x+radius, y+radius, 
                                   fill=color, outline="black")
            
            # 绘制元素符号
            self.canvas.create_text(x, y, text=element, fill="white" if element in ["C", "N", "O"] else "black")
    
    def get_element_color(self, element):
        # 为常见元素设置颜色
        colors = {
            "C": "#333333",  # 碳 - 深灰色
            "H": "#FFFFFF",  # 氢 - 白色
            "O": "#FF0000",  # 氧 - 红色
            "N": "#3050F8",  # 氮 - 蓝色
            "P": "#FF8000",  # 磷 - 橙色
            "S": "#FFFF30",  # 硫 - 黄色
            "F": "#90E050",  # 氟 - 浅绿色
            "Cl": "#1FF01F", # 氯 - 绿色
            "Br": "#A62929", # 溴 - 棕色
            "I": "#940094"   # 碘 - 紫色
        }
        return colors.get(element, "#808080")  # 默认灰色
    
    def clear_canvas(self):
        self.atoms = []
        self.bonds = []
        self.selected_atom = None
        self.drawing_bond = False
        self.current_mol = None
        self.current_smiles = ""
        self.current_inchi = ""
        if self.temp_line:
            self.canvas.delete(self.temp_line)
            self.temp_line = None
        self.redraw_canvas()
        self.smiles_label.config(text="")
        self.inchi_label.config(text="")
        self.image_label.config(image="")
    
    def generate_smiles(self):
        if not self.atoms:
            messagebox.showinfo("提示", "请先绘制分子结构")
            return
        
        try:
            # 创建RDKit分子对象
            mol = self.create_rdkit_molecule()
            if mol is None:
                messagebox.showerror("错误", "无法创建有效的分子结构")
                return
            
            # 保存当前分子对象
            self.current_mol = mol
            
            # 生成SMILES
            smiles = Chem.MolToSmiles(mol)
            self.current_smiles = smiles
            self.smiles_label.config(text=smiles)
            
            return smiles
        except Exception as e:
            messagebox.showerror("错误", f"生成SMILES时出错: {str(e)}")
            return None
    
    def generate_inchi(self):
        if not self.atoms:
            messagebox.showinfo("提示", "请先绘制分子结构")
            return
        
        try:
            # 如果没有当前分子，先创建
            if self.current_mol is None:
                mol = self.create_rdkit_molecule()
                if mol is None:
                    messagebox.showerror("错误", "无法创建有效的分子结构")
                    return
                self.current_mol = mol
            
            # 生成InChI
            inchi_str = inchi.MolToInchi(self.current_mol)
            self.current_inchi = inchi_str
            self.inchi_label.config(text=inchi_str)
            
            return inchi_str
        except Exception as e:
            messagebox.showerror("错误", f"生成InChI时出错: {str(e)}")
            return None
    
    def generate_2d_image(self):
        smiles = self.generate_smiles()
        if not smiles:
            return
        
        try:
            # 从SMILES创建分子
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                messagebox.showerror("错误", "无法从SMILES创建分子")
                return
            
            # 生成2D坐标
            AllChem.Compute2DCoords(mol)
            
            # 绘制分子
            self.mol_image = Draw.MolToImage(mol, size=(300, 300))
            
            # 显示图像
            photo = ImageTk.PhotoImage(self.mol_image)
            self.image_label.config(image=photo)
            self.image_label.image = photo  # 保持引用
        except Exception as e:
            messagebox.showerror("错误", f"生成2D图像时出错: {str(e)}")
    
    def copy_to_clipboard(self, text):
        """复制文本到剪贴板"""
        if PYPERCLIP_AVAILABLE:
            pyperclip.copy(text)
            messagebox.showinfo("复制成功", f"已复制到剪贴板")
        else:
            # 如果pyperclip不可用，显示文本供用户手动复制
            copy_window = tk.Toplevel(self.root)
            copy_window.title("复制文本")
            copy_window.geometry("500x150")
            
            Label(copy_window, text="pyperclip库未安装，请手动复制以下文本:").pack(pady=(10, 5))
            text_field = tk.Text(copy_window, height=3, width=50)
            text_field.pack(padx=10, pady=5)
            text_field.insert(tk.END, text)
            text_field.config(state=tk.DISABLED)  # 设为只读
            
            Button(copy_window, text="关闭", command=copy_window.destroy).pack(pady=10)
        if not text:
            messagebox.showinfo("提示", "没有可复制的内容")
            return
        
        try:
            pyperclip.copy(text)
            messagebox.showinfo("成功", "已复制到剪贴板")
        except Exception as e:
            messagebox.showerror("错误", f"复制到剪贴板时出错: {str(e)}")
    
    def save_2d_image(self):
        if not hasattr(self, 'mol_image') or self.mol_image is None:
            messagebox.showinfo("提示", "请先生成2D图像")
            return
        
        try:
            # 打开文件对话框
            file_path = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG图像", "*.png"), ("JPEG图像", "*.jpg"), ("所有文件", "*.*")]
            )
            
            if file_path:
                self.mol_image.save(file_path)
                messagebox.showinfo("成功", f"图像已保存到: {file_path}")
        except Exception as e:
            messagebox.showerror("错误", f"保存图像时出错: {str(e)}")
    
    def create_rdkit_molecule(self):
        if not self.atoms or not self.bonds:
            messagebox.showinfo("提示", "请绘制至少一个原子和一个键")
            return None
        
        try:
            # 创建空分子
            mol = Chem.RWMol()
            
            # 添加原子
            atom_map = {}
            for i, (_, _, element) in enumerate(self.atoms):
                atom = Chem.Atom(element)
                atom_idx = mol.AddAtom(atom)
                atom_map[i] = atom_idx
            
            # 添加键
            for atom1_idx, atom2_idx, bond_type in self.bonds:
                # 转换键类型
                if bond_type == 1:
                    rdkit_bond_type = Chem.BondType.SINGLE
                elif bond_type == 2:
                    rdkit_bond_type = Chem.BondType.DOUBLE
                elif bond_type == 3:
                    rdkit_bond_type = Chem.BondType.TRIPLE
                elif bond_type == 4:
                    rdkit_bond_type = Chem.BondType.AROMATIC
                else:
                    rdkit_bond_type = Chem.BondType.SINGLE
                
                mol.AddBond(atom_map[atom1_idx], atom_map[atom2_idx], rdkit_bond_type)
            
            # 转换为不可变分子
            mol = mol.GetMol()
            
            # 为每个原子设置显式价态
            for atom in mol.GetAtoms():
                atom.UpdatePropertyCache(strict=False)
            
            try:
                # 尝试清理分子，但捕获可能的异常
                Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL^Chem.SanitizeFlags.SANITIZE_KEKULIZE)
            except Exception as e:
                print(f"警告: 分子清理过程中出现问题: {str(e)}")
                # 继续处理，尝试获取SMILES
            
            # 添加氢原子
            try:
                mol = Chem.AddHs(mol)
            except Exception as e:
                print(f"警告: 添加氢原子时出现问题: {str(e)}")
                # 继续处理，尝试获取SMILES
            
            return mol
        except Exception as e:
            messagebox.showerror("错误", f"创建分子时出错: {str(e)}")
            return None

if __name__ == "__main__":
    root = tk.Tk()
    app = MoleculeDrawer(root)
    root.mainloop()