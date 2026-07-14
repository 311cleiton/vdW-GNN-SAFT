"""
SMILES Property Calculator (MW & vdW_VM) with Status Tracking

This script provides a Tkinter-based graphical user interface (GUI) to select an input 
CSV file containing SMILES strings in its first column. It utilizes RDKit to calculate 
the Molecular Weight (MW) and van der Waals Molecular Volume (vdW_VM) for each unique 
SMILES structure, caching results for efficiency, and exports the final data into a 
new CSV file complete with column headers and a success/fail status column.
"""

import os
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors

# --- Global variables for file paths ---
INPUT_CSV = ""
OUTPUT_CSV = ""


def calculate_properties(smiles):
    """
    Calculates molecular weight and 3D molecular volume for a single SMILES string.
    Returns (MW, Volume, Status).
    Status is "success" if fully identified with zero errors, otherwise "fail".
    """
    if not isinstance(smiles, str) or not smiles.strip():
        return None, None, "fail"
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        # If RDKit cannot resolve the SMILES at all, it returns None
        if mol is None:
            return None, None, "fail"
        
        # 1. Calculate Molecular Weight (2D property)
        mol_weight = Descriptors.MolWt(mol)
        
        # 2. Add Hydrogens for accurate 3D volume
        mol_with_hs = Chem.AddHs(mol)
        
        # 3. Embed the molecule to generate a 3D conformation
        embed_status = AllChem.EmbedMolecule(mol_with_hs, randomSeed=42)
        
        # Fallback: if standard embedding fails, try with random coordinates
        if embed_status == -1:
            embed_status = AllChem.EmbedMolecule(mol_with_hs, randomSeed=42, useRandomCoords=True)
            
        # If embedding still fails, volume cannot be computed accurately
        if embed_status == -1:
            print(f"Warning: 3D embedding failed for '{smiles}'. Volume skipped.")
            return mol_weight, None, "fail"

        # 4. Compute the molecular volume with high precision
        volume = AllChem.ComputeMolVolume(
            mol_with_hs, 
            confId=-1, 
            gridSpacing=0.1, 
            boxMargin=2.0
        )
        
        # If we reached this point, everything resolved perfectly
        return mol_weight, volume, "success"

    except Exception as e:
        print(f"Error processing SMILES '{smiles}': {e}")
        return None, None, "fail"


def process_smiles_data():
    """
    Executes the core RDKit and Pandas data processing logic.
    """
    if not os.path.exists(INPUT_CSV):
        print(f"Error: The file '{INPUT_CSV}' was not found. Please check your path.")
        return

    print(f"Reading data from '{INPUT_CSV}'...")
    # Read CSV. header=None assumes columns are positional (0-indexed).
    df = pd.read_csv(INPUT_CSV, header=None)
    
    if df.shape[1] < 1:
        print("Error: The CSV file must have at least 1 column containing SMILES strings.")
        return

    # Keep only the first column containing SMILES data
    df = df[[0]].copy()

    # -------------------------------------------------------------------------
    # EFFICIENCY STEP: Identify unique SMILES to avoid duplicate calculations
    # -------------------------------------------------------------------------
    unique_smiles = list(df[0].dropna().unique())

    print(f"Found {len(unique_smiles)} unique SMILES strings to process.")

    # Dictionary to serve as our look-up cache: { smiles: {"mw": val, "vol": val, "status": val} }
    cache = {}

    # Run calculations exactly once per unique SMILES
    for i, smiles in enumerate(unique_smiles, 1):
        print(f"[{i}/{len(unique_smiles)}] Calculating: {smiles}")
        mw, vol, status = calculate_properties(smiles)
        cache[smiles] = {"mw": mw, "vol": vol, "status": status}

    print("\nMapping calculated values back to the table layout...")
    
    # -------------------------------------------------------------------------
    # LAYOUT GENERATION: Populate columns 2, 3, and 4 (0-indexed as 1, 2, and 3)
    # -------------------------------------------------------------------------
    df[1] = df[0].map(lambda x: cache.get(x, {}).get("mw") if pd.notna(x) else None)
    df[2] = df[0].map(lambda x: cache.get(x, {}).get("vol") if pd.notna(x) else None)
    df[3] = df[0].map(lambda x: cache.get(x, {}).get("status") if pd.notna(x) else "fail")

    # Format numbers to match original precision requirements (MW: .3f, Vol: .2f)
    df[1] = df[1].map(lambda x: f"{x:.3f}" if pd.notna(x) and isinstance(x, (int, float)) else "")
    df[2] = df[2].map(lambda x: f"{x:.2f}" if pd.notna(x) and isinstance(x, (int, float)) else "")

    # Assign column headers (titles)
    df.columns = ['SMILES', 'MW', 'vdW_VM', 'Status']

    # Save output to file with column titles included
    df.to_csv(OUTPUT_CSV, header=True, index=False)
    print(f"Success! Processing complete. Output saved to '{OUTPUT_CSV}'.")


def main():
    global INPUT_CSV, OUTPUT_CSV

    # --- GUI START ---
    root = tk.Tk()
    root.title("SMILES Property Calculator Configuration")
    root.geometry("700x200")
    
    # GUI Variables
    in_var = tk.StringVar(value="Select input.csv...")
    out_var = tk.StringVar(value="Select output_processed.csv...")

    def browse_open_file(var, title, filetypes):
        path = filedialog.askopenfilename(title=title, filetypes=filetypes)
        if path:
            var.set(path)

    def browse_save_file(var, title, filetypes):
        path = filedialog.asksaveasfilename(title=title, filetypes=filetypes, defaultextension=".csv")
        if path:
            var.set(path)
            
    # Layout Frame
    frame = tk.Frame(root, padx=20, pady=10)
    frame.pack(fill=tk.BOTH, expand=True)

    # 1. Input CSV Layout
    tk.Label(frame, text="Input CSV:", font=("Arial", 10, "bold")).grid(row=0, column=0, sticky="e", pady=10, padx=5)
    tk.Entry(frame, textvariable=in_var, width=55).grid(row=0, column=1, padx=5)
    tk.Button(frame, text="Browse", command=lambda: browse_open_file(in_var, "Select Input CSV", [("CSV Files", "*.csv"), ("All Files", "*.*")])).grid(row=0, column=2)

    # 2. Output CSV Layout
    tk.Label(frame, text="Output CSV:", font=("Arial", 10, "bold")).grid(row=1, column=0, sticky="e", pady=10, padx=5)
    tk.Entry(frame, textvariable=out_var, width=55).grid(row=1, column=1, padx=5)
    tk.Button(frame, text="Browse", command=lambda: browse_save_file(out_var, "Save Processed CSV As", [("CSV Files", "*.csv"), ("All Files", "*.*")])).grid(row=1, column=2)

    # State flag to verify if user executed via the run button
    run_flag = [False]

    def on_run():
        # Quick validation check
        if "Select" in in_var.get() or "Select" in out_var.get():
            messagebox.showwarning("Missing Input", "Please select both input and output paths before running.")
            return

        global INPUT_CSV, OUTPUT_CSV
        INPUT_CSV = in_var.get()
        OUTPUT_CSV = out_var.get()
        
        run_flag[0] = True
        root.destroy()
        
    tk.Button(root, text="Run Property Calculator", command=on_run, bg="#4CAF50", fg="white", font=("Arial", 10, "bold"), width=25).pack(pady=15)
    
    # Start GUI loop
    root.mainloop()
    
    # Check if processing was cancelled
    if not run_flag[0]:
        print("SMILES processing cancelled by user.")
        return
    # --- GUI END ---

    # Execute calculation flow with the selected variables
    process_smiles_data()


if __name__ == "__main__":
    main()