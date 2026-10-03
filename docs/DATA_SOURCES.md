# Data sources

Q-OMICS uses only open-access, public data. No patient data is collected,
stored or redistributed by this project.

## Primary cohort

**TCGA-UCEC** — Uterine Corpus Endometrial Carcinoma, The Cancer Genome Atlas.

| Attribute | Value |
|---|---|
| Samples in analysis | 545 tumour samples |
| Patients | 544 |
| Target | FIGO stage I (early) vs stage II–IV (advanced) |
| Class balance | 340 stage I / 205 stage II–IV (37.6 % events) |
| Tissue | Primary tumour only |
| Modality layers | Genome, transcriptome, proteome |

Stage distribution in the analysis cohort: I = 340, II = 52, III = 124, IV = 29.
437 of the 545 samples carry RPPA proteome data; the remainder are median-imputed
within the proteome block.

## Layer-by-layer provenance

| Layer | Content | Source | Access |
|---|---|---|---|
| **Genome** | Somatic mutations (variant-level, `MUTATION_EXTENDED`) | cBioPortal `ucec_tcga_gdc_mutations` | Public REST, no auth |
| **Genome** | Copy number (GISTIC discrete, −2…+2) | cBioPortal `ucec_tcga_gdc_cna` | Public REST, no auth |
| **Genome** | Tumour mutational burden, fraction genome altered | cBioPortal clinical attributes `TMB_NONSYNONYMOUS`, `FRACTION_GENOME_ALTERED`, `MUTATION_COUNT` | Public REST, no auth |
| **Transcriptome** | Gene-level RNA-seq, TPM z-scores against the TCGA cohort | cBioPortal `ucec_tcga_gdc_mrna_seq_tpm_Zscores` | Public REST, no auth |
| **Proteome** | RPPA antibody panel, 189 antibodies (total and phospho) | UCSC Xena `TCGA.UCEC.sampleMap/RPPA` | Public HTTPS |
| **Clinical** | FIGO stage, histology, sample type, age | cBioPortal `FIGO_GRADE`; Xena `TCGA.UCEC.sampleMap/UCEC_clinicalMatrix` | Public |

### Exact endpoints

```
# cBioPortal
GET  https://www.cbioportal.org/api/studies/ucec_tcga_gdc/clinical-data?clinicalDataType=PATIENT
GET  https://www.cbioportal.org/api/studies/ucec_tcga_gdc/clinical-data?clinicalDataType=SAMPLE
GET  https://www.cbioportal.org/api/sample-lists/ucec_tcga_gdc_tpm/sample-ids
POST https://www.cbioportal.org/api/molecular-profiles/ucec_tcga_gdc_mrna_seq_tpm_Zscores/molecular-data/fetch
GET  https://www.cbioportal.org/api/molecular-profiles/ucec_tcga_gdc_mutations/mutations
POST https://www.cbioportal.org/api/molecular-profiles/ucec_tcga_gdc_cna/molecular-data/fetch
GET  https://www.cbioportal.org/api/genes/{HGNC_SYMBOL}

# UCSC Xena
GET  https://tcga.xenahubs.net/download/TCGA.UCEC.sampleMap%2FRPPA.gz
GET  https://tcga.xenahubs.net/download/TCGA.UCEC.sampleMap%2FUCEC_clinicalMatrix
```

## Feature panel

The analysis uses a curated panel of **375 HGNC gene symbols** spanning
seventeen biological axes, defined in `qomics/genes.py`:

PI3K/PTEN · RAS/MAPK · cell cycle · tumour suppressors · DNA mismatch repair ·
WNT/β-catenin · NOTCH/ESR1 · histone methylation · oestrogen–progesterone
signalling · apoptosis and DDR · WNT stemness · EMT and invasion · angiogenesis
and hypoxia · immune context · metabolism · xenobiotic response · TGF-β and
myogenesis · DNA repair · ECM/stroma · mTORC1 and ribosome biogenesis.

Symbols are resolved to Entrez identifiers through the cBioPortal gene index
at run time, so a retired symbol drops out and is logged rather than silently
producing an all-zero column. At the time of the recorded run **375 of 375
symbols resolved**.

The proteomic layer is the full Xena RPPA panel after a variance filter
(245 antibodies → 189 retained).

## Why a curated panel rather than whole-transcriptome?

Three reasons, in order of importance:

1. **Interpretability.** A biomarker panel of named, pathway-annotated
   molecules is a result a clinician can act on. Twenty thousand anonymous
   gene indices are not.
2. **Register width.** The quantum branch encodes each sample into a
   six-qubit register (64-dimensional Hilbert space). A 20 000-dimensional
   input cannot be encoded without a lossy compression that would dominate
   the comparison with classical kernels.
3. **Multiple-testing burden.** Screening 20 000 features to find a
   significant association is achievable at any threshold. Screening a
   biologically motivated panel with FDR control is a defensible inference.

## Caching

All responses are cached under `data/cache/` as gzip-compressed JSON,
keyed by endpoint and parameters. A warm cache makes cohort assembly
deterministic and offline. The `data/` directory is git-ignored; nothing
large is committed.

## Citation

If you use Q-OMICS or its cohort, please cite both this project and the
underlying data:

**The cohort**
> The Cancer Genome Atlas Research Network. *Integrated Genomic Characterization
> of Uterine Corpus Endometrial Carcinoma.* Nature 487, 291–297 (2013).
> https://doi.org/10.1038/nature12113

> The Cancer Genome Atlas Research Network. *TCGA PanCanAtlas: An integrated
> portrait of endometrial carcinoma.* (2018).

**The portal and hub**
> Cerami et al. *The cBioPortal for Cancer Genomics: an open platform for
> exploring patient genomic data.* Cell Reports 2, 212–218 (2018).
> https://doi.org/10.1016/j.celrep.2016.12.028

> Goldman et al. *Visualizing and interpreting cancer genomics data via the
> Xena platform.* Nature Biotechnology 38, 675–678 (2020).
> https://doi.org/10.1038/s41587-020-0543-7

## Extending to another cohort

`CohortConfig` in `qomics/config.py` is the single switch point. Set a
different `study_id` and the matching molecular profile / sample list IDs and
Q-OMICS will retarget the same protocol. Cohorts with clinical attributes in
a different format need only a new parser in
`qomics/data/cohort.py::parse_figo_stage`, or an analogous function for the
target ontology of the new disease.
