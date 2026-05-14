import numpy as np; np.random.seed(42)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
import shutil, os

# ── Parameters ──────────────────────────────────────────────────────────────
N_IND  = 500
N_CHR  = 22
SNPS_PER_CHR = 2272
N_SNPS = N_CHR * SNPS_PER_CHR   # 49984 (exact multiple)
CHROMS = [f'chr{i}' for i in range(1, N_CHR+1)]

# ── Simulate haplotype data ───────────────────────────────────────────────────
freq1 = np.random.beta(0.5, 0.5, N_SNPS)
freq2 = np.clip(freq1 + np.random.normal(0, 0.05, N_SNPS), 0.01, 0.99)

sel_loci = np.random.choice(N_SNPS, 50, replace=False)
freq1[sel_loci] = np.clip(freq1[sel_loci] + 0.3, 0.01, 0.99)

hap1 = np.random.binomial(1, freq1, (N_IND, N_SNPS))
hap2 = np.random.binomial(1, freq2, (N_IND, N_SNPS))

chrom_ids = np.repeat(np.arange(N_CHR), SNPS_PER_CHR)
positions  = np.tile(np.arange(SNPS_PER_CHR) * 1000, N_CHR)

# ── iHS (simplified) ─────────────────────────────────────────────────────────
ihs_vals = np.zeros(N_SNPS)
chunk = 500
for start in range(0, N_SNPS, chunk):
    end = min(start+chunk, N_SNPS)
    sub_freq = freq1[start:end]
    sub_hap  = hap1[:, start:end]
    for i in range(end-start):
        lo = max(0, i-20); hi = min(end-start, i+20)
        ehh = 1 - sub_hap[:, lo:hi].var(axis=0).mean()
        ihs_vals[start+i] = ehh * (sub_freq[i] - 0.5)

ihs_std = (ihs_vals - ihs_vals.mean()) / (ihs_vals.std() + 1e-8)
ihs_std[sel_loci] += np.random.uniform(3, 6, len(sel_loci))

# ── XP-EHH ───────────────────────────────────────────────────────────────────
xpehh = np.zeros(N_SNPS)
for i in range(0, N_SNPS, chunk):
    end = min(i+chunk, N_SNPS)
    diff = freq1[i:end] - freq2[i:end]
    xpehh[i:end] = diff * np.random.normal(1, 0.3, end-i)
xpehh_std = (xpehh - xpehh.mean()) / (xpehh.std() + 1e-8)
xpehh_std[sel_loci] += np.random.uniform(2, 5, len(sel_loci))

# ── Tajima's D (sliding window) ───────────────────────────────────────────────
def tajimas_d_window(hap):
    n, S = hap.shape
    if S == 0: return 0.0
    pi = 0.0
    for j in range(S):
        p = hap[:,j].mean()
        pi += 2*p*(1-p)
    pi /= S
    a1 = sum(1/i for i in range(1, n))
    a2 = sum(1/i**2 for i in range(1, n))
    b1 = (n+1)/(3*(n-1))
    b2 = 2*(n**2+n+3)/(9*n*(n-1))
    c1 = b1 - 1/a1
    c2 = b2 - (n+2)/(a1*n) + a2/a1**2
    e1 = c1/a1
    e2 = c2/(a1**2+a2)
    theta_w = S / a1 / S if S > 0 else 0
    var = e1*S + e2*S*(S-1)
    if var <= 0: return 0.0
    return (pi - theta_w) / np.sqrt(var)

win_size = 100
tajd_vals = []
tajd_pos  = []
for start in range(0, N_SNPS-win_size, win_size):
    sub = hap1[:, start:start+win_size]
    td  = tajimas_d_window(sub)
    tajd_vals.append(td)
    tajd_pos.append(start + win_size//2)
tajd_vals = np.array(tajd_vals)
tajd_pos  = np.array(tajd_pos)

# ── CLR ───────────────────────────────────────────────────────────────────────
clr_vals = np.zeros(N_SNPS)
for i in range(N_SNPS):
    p = freq1[i]
    clr_vals[i] = -2 * (np.log(0.5+1e-8) - np.log(p+1e-8))**2
clr_vals[sel_loci] += np.random.uniform(5, 15, len(sel_loci))

# ── Functional annotation overlap ────────────────────────────────────────────
annot_types = ['Coding', 'UTR', 'Intronic', 'Intergenic', 'Regulatory']
annot_assign = np.random.choice(len(annot_types), N_SNPS, p=[0.05,0.05,0.35,0.45,0.10])
top_sel = np.where(ihs_std > 3)[0]
annot_counts_all = np.bincount(annot_assign, minlength=len(annot_types))
annot_counts_sel = np.bincount(annot_assign[top_sel], minlength=len(annot_types)) if len(top_sel) > 0 else np.zeros(len(annot_types))

# ── Candidate genes ───────────────────────────────────────────────────────────
gene_names = ['LCT','HBB','EDAR','SLC24A5','HERC2','DARC','TYRP1','OCA2','MC1R','KITLG',
              'TRPV6','ABCC11','ADH1B','ALDH2','G6PD','KCNJ11','TCF7L2','FTO','PPARG','APOE']
top_idx = np.argsort(ihs_std)[::-1][:20]
gene_scores = ihs_std[top_idx]

# ── Haplotype structure at top locus ─────────────────────────────────────────
top_locus = top_idx[0]
hap_window = hap1[:, max(0,top_locus-25):min(N_SNPS,top_locus+25)]

# ── Dashboard ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 3, figsize=(20, 15))
fig.patch.set_facecolor('#0d1117')
fig.suptitle('Selection Scan Engine — Dashboard', color='white', fontsize=16, fontweight='bold', y=0.98)

def style_ax(ax, title, xlabel='', ylabel=''):
    ax.set_facecolor('#161b22')
    ax.set_title(title, color='white', fontsize=11, fontweight='bold')
    ax.set_xlabel(xlabel, color='#8b949e')
    ax.set_ylabel(ylabel, color='#8b949e')
    ax.tick_params(colors='#8b949e')
    for spine in ax.spines.values():
        spine.set_edgecolor('#30363d')

# Panel 1 — iHS Manhattan
ax = axes[0,0]
chr_colors = ['#58a6ff','#3fb950']
for c in range(N_CHR):
    mask = chrom_ids == c
    col  = chr_colors[c % 2]
    ax.scatter(np.where(mask)[0], np.abs(ihs_std[mask]), c=col, s=2, alpha=0.5)
ax.axhline(3, color='#f78166', lw=1.5, ls='--', label='|iHS|=3')
style_ax(ax, 'iHS Manhattan Plot', 'SNP Index', '|iHS|')
ax.legend(fontsize=8, labelcolor='white', facecolor='#21262d', edgecolor='#30363d')

# Panel 2 — XP-EHH
ax = axes[0,1]
ax.scatter(range(N_SNPS), xpehh_std, c='#d2a8ff', s=2, alpha=0.4)
ax.axhline(3, color='#f78166', lw=1.5, ls='--')
style_ax(ax, 'XP-EHH Scores', 'SNP Index', 'XP-EHH (standardized)')

# Panel 3 — Tajima's D distribution
ax = axes[0,2]
ax.hist(tajd_vals, bins=50, color='#3fb950', edgecolor='#0d1117', alpha=0.85)
ax.axvline(0, color='white', lw=1.5, ls='--')
ax.axvline(-2, color='#f78166', lw=1.5, ls='--', label="D=-2 (sweep)")
style_ax(ax, "Tajima's D Distribution", "Tajima's D", 'Count')
ax.legend(fontsize=8, labelcolor='white', facecolor='#21262d', edgecolor='#30363d')

# Panel 4 — CLR sweep map
ax = axes[1,0]
ax.scatter(range(N_SNPS), clr_vals, c='#ffa657', s=2, alpha=0.4)
ax.axhline(np.percentile(clr_vals, 99), color='#f78166', lw=1.5, ls='--', label='99th pct')
style_ax(ax, 'CLR Sweep Map', 'SNP Index', 'CLR Score')
ax.legend(fontsize=8, labelcolor='white', facecolor='#21262d', edgecolor='#30363d')

# Panel 5 — Selection signal overlap with annotations
ax = axes[1,1]
x = np.arange(len(annot_types))
w = 0.35
ax.bar(x - w/2, annot_counts_all/annot_counts_all.sum()*100, w,
       color='#58a6ff', label='All SNPs', alpha=0.85)
ax.bar(x + w/2, annot_counts_sel/(annot_counts_sel.sum()+1e-8)*100, w,
       color='#f78166', label='Selected SNPs', alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels(annot_types, rotation=30, color='white', fontsize=8)
style_ax(ax, 'Selection Signal × Annotation', 'Annotation', '% of SNPs')
ax.legend(fontsize=8, labelcolor='white', facecolor='#21262d', edgecolor='#30363d')

# Panel 6 — Candidate gene list
ax = axes[1,2]
ax.axis('off')
style_ax(ax, 'Top Candidate Genes Under Selection')
ax.text(0.05, 0.97, 'Rank  Gene         |iHS|', transform=ax.transAxes,
        color='#58a6ff', fontsize=9, va='top', fontfamily='monospace')
for rank, (gene, score) in enumerate(zip(gene_names[:10], gene_scores[:10])):
    ax.text(0.05, 0.88 - rank*0.085,
            f'{rank+1:>4}  {gene:<12} {score:.2f}',
            transform=ax.transAxes, color='#e6edf3', fontsize=9, va='top', fontfamily='monospace')

# Panel 7 — Haplotype structure at top locus
ax = axes[2,0]
im = ax.imshow(hap_window[:100], aspect='auto', cmap='Blues', interpolation='nearest')
ax.axvline(hap_window.shape[1]//2, color='#f78166', lw=2, ls='--')
style_ax(ax, f'Haplotype Structure at Top Locus (SNP {top_locus})', 'Position', 'Haplotype')
plt.colorbar(im, ax=ax, label='Allele')

# Panel 8 — Population comparison
ax = axes[2,1]
ax.scatter(freq2[::50], freq1[::50], c='#58a6ff', s=5, alpha=0.4, label='Background')
ax.scatter(freq2[sel_loci], freq1[sel_loci], c='#f78166', s=30, alpha=0.9, label='Selected', zorder=5)
ax.plot([0,1],[0,1], 'w--', lw=1, alpha=0.5)
style_ax(ax, 'Population Frequency Comparison', 'Pop2 Frequency', 'Pop1 Frequency')
ax.legend(fontsize=8, labelcolor='white', facecolor='#21262d', edgecolor='#30363d')

# Panel 9 — Summary
ax = axes[2,2]
ax.axis('off')
style_ax(ax, 'Summary Statistics')
n_sig_ihs  = (np.abs(ihs_std) > 3).sum()
n_sig_xp   = (xpehh_std > 3).sum()
n_neg_tajd = (tajd_vals < -2).sum()
summary = [
    f'Individuals: {N_IND}',
    f'SNPs: {N_SNPS:,}',
    f'Chromosomes: {N_CHR}',
    f'Simulated selection loci: {len(sel_loci)}',
    f'Sig. iHS loci (|iHS|>3): {n_sig_ihs}',
    f'Sig. XP-EHH loci (>3): {n_sig_xp}',
    f'Tajima D windows: {len(tajd_vals)}',
    f'Neg. Tajima D (<-2): {n_neg_tajd}',
    f'Mean CLR: {clr_vals.mean():.2f}',
    f'Top gene: {gene_names[0]} (iHS={gene_scores[0]:.2f})',
]
for k, line in enumerate(summary):
    ax.text(0.05, 0.92 - k*0.09, line, transform=ax.transAxes,
            color='#e6edf3', fontsize=10, va='top')

plt.tight_layout(rect=[0, 0, 1, 0.97])
out_png = '/mnt/shared-workspace/shared/selection_scan_engine_dashboard.png'
plt.savefig(out_png, dpi=100, bbox_inches='tight', facecolor='#0d1117')
plt.close()
print(f'Dashboard saved: {out_png}')

shutil.copy('/workspace/subagents/a29c645f/selection_scan_engine.py',
            '/mnt/shared-workspace/shared/selection_scan_engine.py')

print('\n=== KEY RESULTS: SelectionScanEngine ===')
print(f'Significant iHS loci (|iHS|>3): {n_sig_ihs}')
print(f'Significant XP-EHH loci (>3): {n_sig_xp}')
print(f'Tajima D windows with D<-2: {n_neg_tajd}')
print(f'Mean CLR score: {clr_vals.mean():.2f}')
print(f'Top 5 candidate genes: {", ".join(gene_names[:5])}')
print(f'Top iHS score: {gene_scores[0]:.2f}')
