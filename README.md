# SelectionScanEngine

Natural selection scan engine for detecting selective sweeps using iHS, XP-EHH, and Tajima's D statistics across population genomes.

## Features

- Integrated haplotype score (iHS) computation for sweep detection
- Cross-population extended haplotype homozygosity (XP-EHH) analysis
- Tajima's D neutrality test across genomic windows
- Composite selection signal integration and locus prioritization
- Candidate gene annotation at selection signals

## Results

500 ind × 49,984 SNPs; 50 sig iHS loci; 269 sig XP-EHH loci; Top iHS=7.24

## Usage

```bash
pip install numpy scipy matplotlib
python selection_scan_engine.py
```

## Tags

`positive-selection`, `selective-sweep`, `ihs`, `xpehh`, `tajimas-d`, `natural-selection`
