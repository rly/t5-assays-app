"""Human-readable descriptions for known assay data columns.

Used for AG Grid header tooltips and AI system prompt context.
"""

COLUMN_DESCRIPTIONS: dict[str, str] = {
    # SPR columns
    "GroupName": "Location of the analyte in the 96-well plate",
    "Channel": "Flow cell/channel the measurement was taken from",
    "Injection": "Which injection in the run sequence",
    "Cycle": "Measurement cycle (association + dissociation)",
    "Analyte": "Same as GroupName — the analyte identifier",
    "IDNUMBER": "Compound name/identifier",
    "Box/Plate": "Plate/box the sample came from",
    "Well": "Specific well position on the plate",
    "Concentration[M]": "Analyte concentration used for the injection (molar, M)",
    "kA[1/(M\u00b7s)]": "Association (on-rate) constant. Higher = binds faster",
    "kD[1/s]": "Dissociation (off-rate) constant. Higher = falls off faster",
    "Rmax_RU": "Fitted maximum binding response (in RU)",
    "KD_M": "Binding affinity (KD) in molar. Lower = tighter binding",
    "RMSE_RU": "Average fit error (RU). Lower = better fit. Filter: < 10",
    "Chi2_ndof_RU2": "Reduced chi-squared (RU\u00b2). Lower = better fit. Filter: < 10",

    # AI docking columns
    "VEEV - AI Binding Score": "AI-predicted binding score for VEEV macrodomain. More negative = stronger predicted binding",
    "VEEV - Binding Score": "AI-predicted binding score for VEEV macrodomain. More negative = stronger predicted binding",
    "EEEV - Binding Score": "AI-predicted binding score for EEEV macrodomain",
    "CHK - Binding Score": "AI-predicted binding score for Chikungunya virus macrodomain",
    "Covid - Binding Score": "AI-predicted binding score for SARS-CoV-2",

    # Fluorescence Polarization columns
    "FP binding (uM)": "Fluorescence polarization binding affinity (\u00b5M). Lower = stronger binding",
    "PARG Number": "PARG compound identifier",
    "PARG Number FP": "PARG compound identifier from the FP assay sheet",

    # Compound properties
    "Name": "Compound name/identifier",
    "Structure": "SMILES string — click to view 2D structure",
    "Chemical formula": "Molecular formula",
    "Molecular weight": "Molecular weight (Da)",
    "LogP": "Lipophilicity. Higher = more lipophilic",
    "TPSA": "Topological polar surface area (\u00c5\u00b2)",
    "Fsp\u00b3": "Fraction of sp3-hybridized carbons",
    "Heavy atoms": "Number of non-hydrogen atoms",
    "Rotatable bonds": "Number of rotatable bonds. Fewer = more rigid",
    "Hydrogen bonds donors": "H-bond donor count",
    "Hydrogen bonds acceptors": "H-bond acceptor count",
    "Molar refractivity": "Molar refractivity — related to molecular size and polarizability",
    "Solubility": "Predicted aqueous solubility",

    # Drug-likeness
    "Lipinski rule": "Lipinski's Rule of Five compliance",
    "Veber's(GSK) rule": "Veber's rule compliance (oral bioavailability)",
    "Druglikeness": "Overall drug-likeness score",
    "Synthesis complexity": "Synthetic accessibility score",

    # ADMET
    "Metabolic stability (H)": "Human metabolic stability prediction",
    "Metabolic stability (M)": "Mouse metabolic stability prediction",
    "PAMPA": "Passive membrane permeability (PAMPA assay prediction)",
    "BBB penetration": "Blood-brain barrier penetration prediction",
    "P-gp inhibitor": "P-glycoprotein inhibitor potential",
    "hERG inhibitor": "hERG channel inhibition risk (cardiac toxicity)",

    # DLS columns
    "n_kept": "Number of replicate wells retained after DLS QC",
    "mean_Range1 Radius (I) (0.1-10nm)": "Intensity-weighted hydrodynamic radius of the protein population (nm)",
    "mean_Radius (nm)": "Intensity-weighted whole-sample hydrodynamic radius (including aggregates)",
    "mean_Range1 %Intensity (I)": "Scattering intensity attributed to Range1 population",
    "mean_SOS": "Sum-of-squares quality metric for correlation fit (should be < 100)",
    "mean_Baseline": "Baseline level of the autocorrelation function",
    "mean_Intensity (Cnt/s)": "Total detected scattering intensity (counts per second)",
    "mean_DLS Temp (C)": "DLS measurement temperature (\u00b0C)",
    "mean_Amplitude": "Correlation intercept",
    "mean_Mw-R (kDa)": "Apparent molecular weight from overall distribution (kDa)",
    "mean_Range1 Mw-R (I) (kDa)": "Apparent molecular weight for Range1 population (kDa)",
    "mean_Range1 %Pd (I)": "Percent polydispersity for Range1 population",
    "mean_Range1 %Mass (I)": "Estimated mass fraction assigned to Range1 population",
    "agg_ratio": "Aggregation index = Radius / Range1 Radius. Higher = more large-species contribution",
    "dls_z_score": "Z-score: (Rh_range1 - control_mean) / control_sd",
    "dls_bad_by_zscore": "Outlier flag based on |Z| > 2",
}
