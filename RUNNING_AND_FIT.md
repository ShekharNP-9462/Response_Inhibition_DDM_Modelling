# Running the models and judging fit

## Prerequisites

From the project root (where `run_beests_single_subject.py` and `run_ddm_single_subject.py` live):

```bash
pip install -r requirements.txt
```

You need your Rashi dataset: a folder containing one subfolder per subject (e.g. `EF_Y_01`, `EF_O_01`), each with `trialData*.csv` files. Use the **full path** to a subject folder (e.g. `C:\...\Rashi_Dataset\EF_Y_01`).

---

## How to run the models

### BEESTS (ex-Gaussian go + ex-Gaussian SSRT)

```bash
python run_beests_single_subject.py <SUBJECT_DIR> [OPTIONS]
```

**Example** (save posterior to NetCDF for later PPC):

```bash
python run_beests_single_subject.py "C:\Users\...\Rashi_Dataset\EF_Y_01" --draws 1000 --tune 1000 --chains 2 --output EF_Y_01_beests.nc
```

**Arguments:**

| Argument        | Meaning |
|----------------|--------|
| `subject_dir`  | Path to the subject folder (e.g. `.../Rashi_Dataset/EF_Y_01`). |
| `--draws`      | Number of MCMC draws per chain (default: 1000). |
| `--tune`       | Number of tuning steps per chain (default: 1000). |
| `--chains`     | Number of chains (default: 2). |
| `--n-mc`       | Monte Carlo samples per stop-trial likelihood (default: 500). |
| `--output`     | Optional path to save the trace as NetCDF (e.g. `subject_beests.nc`). |
| `--log-interval` | Print elapsed/ETA every N steps (default: 100). |

The script prints a summary table of the posterior (e.g. `mu_go`, `sigma_go`, `tau_go`, `mu_ssrt`, `sigma_ssrt`, `tau_ssrt`) and, if `--output` is set, saves the full trace.

---

### DDM (drift, boundary, Ter + ex-Gaussian SSRT)

```bash
python run_ddm_single_subject.py <SUBJECT_DIR> [OPTIONS]
```

**Example:**

```bash
python run_ddm_single_subject.py "C:\Users\...\Rashi_Dataset\EF_Y_01" --draws 1000 --tune 1000 --chains 2 --output EF_Y_01_ddm.nc
```

**Arguments:** Same as BEESTS. The summary table shows `v`, `a`, `ter` (drift, boundary, non-decision time) plus `mu_ssrt`, `sigma_ssrt`, `tau_ssrt`.

---

## Judging fit: visual + quantitative

After fitting, run the assessment script to get **visual** checks (PPC, QP plot) and **quantitative** measures (WAIC, LOO, R̂).

**Run assessment** (requires a saved NetCDF trace and the same subject folder):

```bash
python assess_fit_ppc.py <SUBJECT_DIR> <TRACE_NETCDF> --model beests --output-dir figs
python assess_fit_ppc.py <SUBJECT_DIR> <TRACE_NETCDF> --model ddm --output-dir figs
```

**Example:**

```bash
python assess_fit_ppc.py "C:\...\Rashi_Dataset\EF_Y_01" EF_Y_01_beests.nc --model beests --output-dir ppc_figs
python assess_fit_ppc.py "C:\...\Rashi_Dataset\EF_Y_01" EF_Y_01_ddm.nc --model ddm --output-dir ppc_figs
```

### What you get

**1. Visual: PPC (posterior predictive check)**  
- **Go RT**: Histogram of observed go RTs vs many posterior-predictive histograms. Good fit = overlap.  
- **Inhibition curve**: Observed P(inhibit) binned by SSD vs posterior-predictive mean and 94% band. Good fit = observed points inside or close to the band.

**2. Visual: QP plot (quantile–probability)**  
- Observed go RT quantiles (0.1, 0.2, …, 0.9) vs predicted quantiles. Points near the 45° line = good fit; systematic drift indicates misfit in the fast or slow tail.

**3. Quantitative: WAIC / LOO**  
- **WAIC** (Widely Applicable Information Criterion) and **LOO** (leave-one-out): used to **compare** models (e.g. DDM vs BEESTS). Lower elpd_waic / elpd_loo = worse fit; more negative = worse. Reported in the console and in the JSON report.

**4. Quantitative: R̂ (Gelman–Rubin)**  
- MCMC convergence diagnostic. R̂ ≈ 1 (e.g. &lt; 1.01) for all parameters = chains converged. R̂ &gt; 1.01 is flagged. A bar plot of R̂ per parameter is saved.

**Outputs in `--output-dir`:**  
- `{subject_id}_{model}_ppc.png` — PPC (go RT + inhibition curve) + QP plot.  
- `{subject_id}_{model}_rhat.png` — R̂ bar plot.  
- `{subject_id}_{model}_fit_report.json` — WAIC, LOO, R̂ max, and R̂ by parameter.

**Optional:** `--n-waic-draws 300` limits how many posterior draws are used to compute pointwise log-likelihood (faster; default uses all). `--n-mc-ll 200` sets MC samples per stop trial for that step.

---

## Quick checklist

1. Run BEESTS: `python run_beests_single_subject.py <subject_dir> --output subject_beests.nc`
2. Run DDM: `python run_ddm_single_subject.py <subject_dir> --output subject_ddm.nc`
3. Assess BEESTS fit: `python assess_fit_ppc.py <subject_dir> subject_beests.nc --model beests --output-dir ppc_figs`
4. Assess DDM fit: `python assess_fit_ppc.py <subject_dir> subject_ddm.nc --model ddm --output-dir ppc_figs`
5. Inspect PPC plots; if observed and predicted align, the fit is reasonable.
