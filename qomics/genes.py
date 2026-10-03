"""Curated multi-omics feature panel for endometrial carcinoma.

A whole-transcriptome basis change (20 530 genes) is neither interpretable
nor encodable in a 6-8 qubit register.  Q-OMICS therefore operates on a
biologically curated panel spanning the pathways that drive and modulate
endometrial carcinogenesis.

Symbols are stored as HGNC gene symbols and resolved to Entrez identifiers
through the cBioPortal gene index at run time, which keeps the panel
self-validating: a symbol that has been retired simply drops out and is
reported, instead of silently producing an all-zero column.

The panel is grouped by axis so that per-modality importances remain
attributable to a biological process rather than to an opaque index.
"""

from __future__ import annotations

# --------------------------------------------------------------------------
# Genomic layer - genes recurrently mutated / amplified / deleted in UCEC
# --------------------------------------------------------------------------

GENOME_PANEL: dict[str, tuple[str, ...]] = {
    "pi3k_pten": (
        "PTEN", "PIK3CA", "PIK3R1", "PIK3R2", "AKT1", "AKT2", "MTOR",
        "RPS6KB1", "PDPK1", "TSC1", "TSC2", "INPP4B", "PIK3C2G",
    ),
    "ras_mapk": (
        "KRAS", "NRAS", "BRAF", "MAP2K1", "MAP2K2", "MAP2K4",
        "MAP3K1", "RASA1", "ERBB2", "ERBB3", "EGFR", "IGF1R",
    ),
    "cell_cycle_cdk": (
        "CDKN1A", "CDKN1B", "CDKN2A", "CDKN2B", "CCND1", "CCND2",
        "CCNE1", "CCNE2", "CDK4", "CDK6", "RB1", "E2F1", "CDC20",
        "CDK1", "CDK2", "CHEK1", "CHEK2", "MYC", "MYCN", "TERT",
    ),
    "tumor_suppressors": (
        "TP53", "TP53BP1", "ATM", "ATR", "CHEK1", "BARD1", "BRCA1",
        "BRCA2", "PALB2", "RAD51", "STK11", "NF1", "NF2", "SETD2",
        "BAP1", "ARID1A", "ARID1B", "ARID2", "SMARCA4", "SMARCB1",
        "SMARCE1", "PPP2R1A", "FBXW7", "RNF43", "ZNRF3", "SOX17",
        "KMT2C", "KMT2D", "CREBBP", "EP300", "MED12", "NCOA2",
        "NCOA3", "NCOR1", "NCOR2", "ERBB2", "NBN", "MRE11", "RAD50",
    ),
    "dna_mismatch_repair": (
        "MLH1", "MSH2", "MSH3", "MSH6", "PMS1", "PMS2", "EPCAM",
        "POLD1", "POLE", "POLE2", "MSH2",
    ),
    "wnt_beta_catenin": (
        "CTNNB1", "AXIN1", "AXIN2", "APC", "TCF7", "LEF1", "WNT5A",
        "FZD2", "LRP5", "LRP6", "DVL2", "GSK3B", "CDK1",
    ),
    "notch_esri": (
        "NOTCH1", "NOTCH2", "JAG1", "HES1", "HEY1", "ESR1", "ESR2",
        "PGR", "ARID1A", "GATA3", "FOXO1", "GREB1", "TFF1", "PDZK1",
    ),
    "histone_methylation": (
        "EZH2", "SUZ12", "EED", "KDM6A", "KDM5C", "SETD1B", "SETD2",
        "KMT2A", "IDH1", "IDH2",
    ),
    "other_recurrent": (
        "KRAS", "FBXW7", "TGFBR2", "TGFBR1", "SMAD2", "SMAD4", "KMT2C",
        "JUN", "ERBB4", "MACF1", "RPL22", "KRAS", "CTNNB1", "NCOA2",
    ),
}

# --------------------------------------------------------------------------
# Transcriptomic layer - pathway-resolved expression markers
# --------------------------------------------------------------------------

TRANSCRIPTOME_PANEL: dict[str, tuple[str, ...]] = {
    # PI3K / AKT / mTOR signalling
    "pi3k_pten": (
        "PTEN", "PIK3CA", "PIK3R1", "AKT1", "AKT2", "MTOR", "RPS6KB1",
        "PDPK1", "TSC1", "TSC2", "IGF1", "IGF1R", "INSR", "PPP2R1A",
    ),
    # RAS / MAPK
    "ras_mapk": (
        "KRAS", "NRAS", "BRAF", "MAP2K1", "MAP2K2", "DUSP1", "DUSP4",
        "DUSP6", "SOS1", "ERBB2", "EGFR", "GRB2", "SHC1",
    ),
    # Cell cycle and proliferation
    "cell_cycle": (
        "MKI67", "PCNA", "TOP2A", "CCNA2", "CCNB1", "CCND1", "CDK1",
        "CDK2", "CDK4", "CDKN1A", "CDKN1B", "CDKN2A", "E2F1", "BUB1",
        "AURKA", "AURKB", "PLK1", "FOXM1", "MYBL2", "CDC6", "CDCA3",
        "TYMS", "RRM2", "MCM2", "MCM4", "MCM5", "MCM7", "PCLAF",
    ),
    # Hormone signalling - the dominant UCEC axis
    "estrogen_progesterone": (
        "ESR1", "ESR2", "PGR", "GATA3", "FOXO1", "GREB1", "TFF1",
        "TFF3", "PDZK1", "IGFBP1", "PGRMC1", "NCOR1", "NCOR2",
        "FOXA1", "FOXC1", "HNF4A", "CDX2", "MUC1", "EPCAM", "KRT19",
        "ELF3", "TACSTD2", "AQP5", "PODXL", "AGR2", "SPINT2",
    ),
    # Apoptosis and DNA damage response
    "apoptosis_ddr": (
        "BAX", "BAK1", "BCL2", "BCL2L1", "BCL2L11", "BID", "CASP3",
        "CASP8", "CASP9", "APAF1", "BIRC5", "XIAP", "DIABLO", "TP53",
        "MDM2", "ATM", "CHEK1", "CHEK2", "GADD45A", "HMOX1", "SESN1",
        "SESN2", "SERPINB5", "S100A10",
    ),
    # WNT / beta-catenin / stemness
    "wnt_stemness": (
        "CTNNB1", "AXIN2", "TCF7", "LEF1", "GATA4", "NANOG", "POU5F1",
        "SOX2", "ALDH1A1", "ALDH1A3", "LGR5", "PROM1", "CD44", "SPP1",
        "WNT5A", "FZD7", "RSPO2", "ASCL1", "MYC",
    ),
    # EMT, invasion and remodelling
    "emt_invasion": (
        "CDH1", "CDH2", "VIM", "FN1", "SNAI1", "SNAI2", "TWIST1",
        "ZEB1", "ZEB2", "MMP2", "MMP7", "MMP9", "TIMP1", "SERPINE1",
        "ITGA1", "ITGB1", "ITGAV", "LAMA1", "LAMB1", "THY1", "NT5E",
    ),
    # Angiogenesis and hypoxia
    "angiogenesis_hypoxia": (
        "VEGFA", "VEGFB", "VEGFC", "KDR", "FLT1", "ANGPT1", "ANGPT2",
        "TEK", "ESM1", "CA9", "HIF1A", "LDHA", "SLC2A1", "PGK1", "ADM",
        "PECAM1", "CDH5", "APLNR",
    ),
    # Immune context and checkpoint
    "immune": (
        "CD8A", "CD8B", "CD3E", "CD3D", "CD2", "PTPRC", "LCK", "ZAP70",
        "FOXP3", "IL2RA", "CTLA4", "PDCD1", "CD274", "LAG3", "HAVCR2",
        "TIGIT", "IDO1", "CD19", "MS4A1", "CD79A", "CD79B", "CD22",
        "CD68", "CD163", "MRC1", "LYZ", "CD1C", "FCGR3A", "S100A8",
        "S100A9", "CXCL9", "CXCL10", "CXCL12", "CCL5", "IL6", "TGFB1",
    ),
    # Metabolic reprogramming
    "metabolism": (
        "HK1", "HK2", "G6PD", "PFKL", "ALDOA", "GAPDH", "PKM", "LDHA",
        "SLC16A1", "SLC2A1", "ACLY", "ACACA", "FASN", "SCD", "MLXIPL",
        "CPT1A", "PPARG", "SREBF1", "HMGCR", "IDH1", "FH", "SDHA",
        "GLS", "GSR", "SOD1", "SOD2", "CAT", "GPX1", "NFE2L2",
    ),
    # Oestrogen/metabolic and xenobiotic response (core TCGA signatures)
    "xenobiotic_response": (
        "CYP1A1", "CYP1B1", "CYP3A4", "NR1I2", "AHR", "POR", "NQO1",
        "GCLC", "GCLM", "UGT1A1",
    ),
    # ECM, hypoxia and PI3K-mTOR gene sets used by the TCGA UCEC signatures
    "tgf_b_myogenesis": (
        "TGFB1", "TGFB2", "TGFB3", "TGFBR1", "TGFBR2", "SMAD2", "SMAD3",
        "SMAD4", "TGFBI", "THBS1", "COL1A1", "COL1A2", "COL3A1",
        "ACTA2", "TAGLN", "MYL9", "MYH11", "CNN1",
    ),
    # DNA repair / replication stress
    "dna_repair": (
        "BRCA1", "BRCA2", "RAD51", "RAD51C", "PALB2", "BARD1", "MRE11",
        "NBN", "RAD50", "XRCC1", "XRCC2", "XRCC5", "LIG4", "MUTYH",
        "MLH1", "MSH2", "MSH6", "PMS2", "POLE", "POLD1", "PCNA",
        "RFC3", "MCM2", "MCM6", "TOP1", "PARP1", "H2AX",
    ),
    # ECM / collagen and tumour-stroma interaction
    "ecm_stroma": (
        "COL1A1", "COL1A2", "COL4A1", "COL4A2", "COL6A1", "COL6A2",
        "COL6A3", "FN1", "POSTN", "FBN1", "LOX", "LOXL1", "SPARC",
        "SERPINH1", "MMP14", "CTSK", "PDGFRB", "FAP", "VIM", "ACTA2",
    ),
    # MTORC1 / ribosomal biogenesis
    "mtorc1_ribosome": (
        "MTOR", "RPTOR", "RICTOR", "TSC1", "TSC2", "RPS6KB1", "EIF4EBP1",
        "RPL3", "RPL5", "RPL11", "RPS3", "RPS6", "RPSA", "RPL22",
        "RRP12", "NOP2", "NOLC1", "DDX21", "MRPL44", "TCOF1",
    ),
}

# --------------------------------------------------------------------------
# Proteomic layer - RPPA antibody panel (handled separately, see xena.py)
# --------------------------------------------------------------------------

#: Fraction of the RPPA panel retained after the variance filter.
PROTEOME_TOP_FRAC = 0.85

#: Pathway annotations for the most clinically relevant antibodies, used to
#: label proteomic biomarkers in the report.
PROTEIN_PATHWAY_HINTS: dict[str, str] = {
    "p53": "p53 pathway",
    "p16": "cell cycle",
    "rb": "cell cycle",
    "cyclin": "cell cycle",
    "cdk": "cell cycle",
    "stat3": "JAK/STAT",
    "stat1": "JAK/STAT",
    "stat5": "JAK/STAT",
    "jak": "JAK/STAT",
    "akt": "PI3K/AKT/mTOR",
    "mts1": "PI3K/AKT/mTOR",
    "s6k1": "PI3K/AKT/mTOR",
    "4e-bp1": "PI3K/AKT/mTOR",
    "ampk": "energy sensing",
    "acco": "energy sensing",
    "ulk1": "energy sensing",
    "mapk": "RAS/MAPK",
    "raf": "RAS/MAPK",
    "mek": "RAS/MAPK",
    "erk": "RAS/MAPK",
    "p38": "stress kinase",
    "mTOR": "PI3K/AKT/mTOR",
    "er-": "hormone receptor",
    "er_": "hormone receptor",
    "her2": "RTK signalling",
    "egfr": "RTK signalling",
    "insulin": "metabolic",
    "irs1": "metabolic",
    "glucose": "metabolic",
    "caveolin": "adhesion",
    "cadherin": "adhesion",
    "claudin": "adhesion",
    "vimentin": "EMT",
    "twist": "EMT",
    "snail": "EMT",
    "slug": "EMT",
    "vegf": "angiogenesis",
    "notch": "Notch signalling",
    "nfkb": "NF-kB",
    "ikb": "NF-kB",
    "bcl": "apoptosis",
    "bad": "apoptosis",
    "bcl-xl": "apoptosis",
    "caspase": "apoptosis",
    "mcl": "apoptosis",
    "h2ax": "DNA damage",
    "dna-pk": "DNA damage",
    "53bp1": "DNA damage",
    "rad51": "DNA repair",
    "yap1": "Hippo pathway",
    "lfa1": "adhesion",
}


def flatten(panel: dict[str, tuple[str, ...]]) -> list[str]:
    """Return a de-duplicated, order-preserving symbol list from a panel."""
    seen: set[str] = set()
    out: list[str] = []
    for group in panel.values():
        for symbol in group:
            if symbol not in seen:
                seen.add(symbol)
                out.append(symbol)
    return out


def build_symbol_panels() -> dict[str, list[str]]:
    """Assemble the per-layer HGNC symbol panels.

    The genome layer is the union of the genomic and transcriptomic panels:
    a feature counts for the genomic layer when it is mutated or copy-number
    altered, and for the transcriptomic layer when it is expressed.
    """
    return {
        "genome": flatten(GENOME_PANEL),
        "transcriptome": flatten(TRANSCRIPTOME_PANEL),
    }


def symbol_to_axis(symbol: str) -> str:
    """Return the pathway axis a symbol belongs to (first match wins)."""
    for axis in TRANSCRIPTOME_PANEL:
        if symbol in TRANSCRIPTOME_PANEL[axis]:
            return axis
    for axis in GENOME_PANEL:
        if symbol in GENOME_PANEL[axis]:
            return axis
    return "unclassified"


def all_panels() -> dict[str, dict[str, tuple[str, ...]]]:
    return {"genome": GENOME_PANEL, "transcriptome": TRANSCRIPTOME_PANEL}


__all__ = [
    "GENOME_PANEL",
    "TRANSCRIPTOME_PANEL",
    "PROTEOME_TOP_FRAC",
    "PROTEIN_PATHWAY_HINTS",
    "flatten",
    "build_symbol_panels",
    "symbol_to_axis",
    "all_panels",
]
