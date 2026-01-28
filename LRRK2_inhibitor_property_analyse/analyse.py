import os
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, rdMolDescriptors
import seaborn as sns
import matplotlib.pyplot as plt

# ===============================
# Settings (you can tweak these)
# ===============================
SAVE_DIR = "plots"
LABEL_SIZE = 28        # x/y axis label font size (bigger)
TICK_LABELSIZE = 24    # tick label font size
SPINE_WIDTH = 1.6
SPINE_COLOR = "#333333"  # deep gray
DPI = 300

# Ensure output dir exists
os.makedirs(SAVE_DIR, exist_ok=True)

# Force matplotlib / seaborn to use Arial for all text
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["axes.labelsize"] = LABEL_SIZE
plt.rcParams["xtick.labelsize"] = TICK_LABELSIZE
plt.rcParams["ytick.labelsize"] = TICK_LABELSIZE
plt.rcParams["legend.fontsize"] = 22

sns.set(style="whitegrid")  # keep a clean grid background

# ===============================
# 1. Calculate molecular properties
# ===============================
def calc_properties(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    props = {}
    props["MW"] = Descriptors.MolWt(mol)
    props["LogP"] = Crippen.MolLogP(mol)
    props["TPSA"] = rdMolDescriptors.CalcTPSA(mol)
    props["RotBond"] = Lipinski.NumRotatableBonds(mol)
    props["HBD"] = Lipinski.NumHDonors(mol)
    props["RingCount"] = rdMolDescriptors.CalcNumRings(mol)
    return props

# ===============================
# 2. Read data, compute pIC50, and classify activities
# ===============================
def prepare_dataset(file_path):
    df = pd.read_csv(file_path)

    # Calculate properties
    props_list = []
    for smi in df["SMILES"]:
        p = calc_properties(smi)
        if p:
            props_list.append(p)
        else:
            props_list.append({k: np.nan for k in ["MW","LogP","TPSA","RotBond","HBD","RingCount"]})

    props_df = pd.DataFrame(props_list)
    df = pd.concat([df, props_df], axis=1)

    # compute pIC50
    df["IC50_numeric"] = pd.to_numeric(df["IC50"], errors="coerce")
    df["pIC50"] = np.nan
    mask_valid = ~df["IC50_numeric"].isna()
    df.loc[mask_valid, "pIC50"] = -np.log10(df.loc[mask_valid, "IC50_numeric"] * 1e-9)

    # classify activities
    conditions = [
        df["pIC50"] >= 7,
        (df["pIC50"] >= 6) & (df["pIC50"] < 7),
        df["pIC50"] < 6
    ]
    choices = ["High", "Medium", "Low"]
    df["Activity"] = np.select(conditions, choices, default="Unknown")

    return df

# ===============================
# 3. Visualization helpers
# ===============================
def plot_property_distributions(df,
                                props=["MW","LogP","TPSA","RotBond","HBD","RingCount"],
                                save_dir=SAVE_DIR,
                                dpi=DPI,
                                labelsize=LABEL_SIZE,
                                tick_labelsize=TICK_LABELSIZE,
                                spine_width=SPINE_WIDTH,
                                spine_color=SPINE_COLOR):
    sns.set(style="whitegrid")
    palette = {"High": "#2ca02c", "Medium": "#1f77b4", "Low": "#d62728"}  # custom colors

    for prop in props:
        fig, ax = plt.subplots(figsize=(8, 6))

        # stripplot with jitter
        sns.stripplot(x="Activity", y=prop, data=df,
                      order=["High", "Medium", "Low"],
                      jitter=True, alpha=0.85,
                      palette=palette, size=5, ax=ax)

        # add legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=palette["High"], markersize=10, label="High"),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=palette["Medium"], markersize=10, label="Medium"),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=palette["Low"], markersize=10, label="Low"),
        ]
        ax.legend(handles=legend_elements, loc="upper right", fontsize=tick_labelsize, frameon=True, )

        # Axis labels
        ax.set_xlabel("Activity", fontsize=labelsize, fontname="Arial")
        ax.set_ylabel(prop, fontsize=labelsize, fontname="Arial")
        # Set y-axis ranges for specific properties to improve readability
        if prop == "MW":
            ax.set_ylim(0, 1200)
        elif prop == "LogP":
            ax.set_ylim(None, 11)
        elif prop == "TPSA":
            ax.set_ylim(0, 250)
        elif prop == "RotBond":
            ax.set_ylim(0, 26)
        elif prop == "HBD":
            ax.set_ylim(0, 6)
        elif prop == "RingCount":
            ax.set_ylim(0, 10)
            
        

        # Tick styling
        ax.tick_params(axis='x', labelsize=tick_labelsize)
        ax.tick_params(axis='y', labelsize=tick_labelsize)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontname("Arial")

        # Spine styling
        for spine_name, spine in ax.spines.items():
            spine.set_linewidth(spine_width)
            spine.set_edgecolor(spine_color)

        # Save figure
        safe_prop = prop.replace("/", "_").replace(" ", "_")
        out_path = os.path.join(save_dir, f"{safe_prop}_scatter.png")
        plt.tight_layout()
        fig.savefig(out_path, dpi=dpi)
        plt.close(fig)
        print(f"Saved: {out_path}")

# ===============================
# 4. Define thresholds based on mean ± std of the high-activity group
# ===============================
def thresholds_from_errorbar(df, props=["MW","LogP","TPSA","RotBond","HBD","RingCount"]):
    high = df[df["Activity"] == "High"]
    final_thresholds = {}

    for prop in props:
        mean_val = high[prop].mean()
        std_val = high[prop].std()
        if np.isnan(mean_val) or np.isnan(std_val):
            final_thresholds[prop] = {"Lower": np.nan, "Upper": np.nan}
        else:
            lower = int(round(mean_val - std_val))
            upper = int(round(mean_val + std_val))
            final_thresholds[prop] = {"Lower": lower, "Upper": upper}

    return final_thresholds

# ===============================
# Main
# ===============================
if __name__ == "__main__":
    # LRRK2 inhibitors activity dataset
    infile = "LRRK2_inhibitor_bindingdb.csv"
    df = prepare_dataset(infile)

    # plotting
    plot_property_distributions(df)

