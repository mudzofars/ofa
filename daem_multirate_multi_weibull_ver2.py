from __future__ import annotations
from functools import partial
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import simpson, quad
from scipy.optimize import differential_evolution, least_squares
import os, argparse, datetime, sys, warnings

# ╔══════════════════════════════════════╗
# ║ 1.  Constants & global parameters    ║
# ╚═════════════════════════════════════╝
R = 8.314          # J/mol/K  – gas constant (keep units consistent!)
FIXED_K0 = 5e11    # s-1          – pre-exponential factor (assumed same for all components)

# ╔══════════════════════════════════════╗
# ║ 2.  DAEM kernel                      ║
# ╚═════════════════════════════════════╝

def compute_weibull_component(lambda_w: float, k_w: float, T_vals: np.ndarray,
                               *, num_pts: int = 51) -> np.ndarray:
    """Return alpha(T) for a single Weibull distribution of activation energies.

    Parameters
    ----------
    lambda_w : float
        Scale parameter of Weibull distribution (J/mol).
    k_w : float
        Shape parameter of Weibull distribution (dimensionless).
    T_vals : array_like [K]
        Temperatures at which to evaluate conversion.
    num_pts : int, optional
        Number of E points in Weibull quadrature grid.
    """
    # --- build Weibull PDF -------------------------------------------------------------
    # Determine a suitable range for E based on lambda_w and k_w.
    # A common approach is to integrate over a range where the PDF is significant.
    # Let's use a range from a small positive value up to a point where the CDF is close to 1.
    # For k_w > 1, the mode is at lambda_w * ((k_w - 1) / k_w)**(1/k_w)
    # For k_w <= 1, the mode is at 0.
    # We need E > 0 for the Weibull distribution.
    e_min_pdf = 1.0 # Small positive value
    # Find a reasonable upper bound, e.g., where CDF is 0.999
    # 1 - exp(-(E/lambda_w)^k_w) = 0.999  => exp(-(E/lambda_w)^k_w) = 0.001
    # => -(E/lambda_w)^k_w = ln(0.001) => (E/lambda_w)^k_w = -ln(0.001)
    # => E/lambda_w = (-ln(0.001))**(1/k_w) => E = lambda_w * (-ln(0.001))**(1/k_w)
    try:
      e_max_pdf = lambda_w * (-np.log(0.001))**(1/k_w)
    except (ValueError, ZeroDivisionError):
       # Fallback for problematic k_w values
       e_max_pdf = lambda_w * 5 # A heuristic

    E_grid = np.linspace(e_min_pdf, e_max_pdf, num_pts)
    # Ensure E_grid is strictly positive
    E_grid = E_grid[E_grid > 0]
    if len(E_grid) < 2: # Ensure at least two points for integration
        E_grid = np.linspace(e_min_pdf, e_max_pdf * 2, num_pts)
        E_grid = E_grid[E_grid > 0]


    pdf    = (k_w / lambda_w) * (E_grid / lambda_w)**(k_w - 1) * np.exp(-(E_grid / lambda_w)**k_w)
    # Handle potential NaNs or Infs in pdf, which can happen for E_grid=0 if k_w<1
    pdf[~np.isfinite(pdf)] = 0
    pdf   /= simpson(pdf, E_grid)  # normalise exactly

    # Lower integration limit is first recorded temp (dynamic & in K)
    T_min = float(T_vals.min())

    # Pre-compute integral exp(-E/RT) dT for every E, all T in T_vals -----------------
    def G_of_E(E):
        # Vectorised – integrate for *each* T in T_vals
        # Use a robust integration method
        results = []
        for T in T_vals:
            try:
                res, _ = quad(lambda Tp: np.exp(-E/(R*Tp)), T_min, T, epsabs=1e-6, epsrel=1e-5)
                results.append(res)
            except Exception: # Catch potential integration errors
                results.append(np.nan)
        return np.array(results)

    G_mat = np.array([G_of_E(E) for E in E_grid])  # shape (num_pts, len(T_vals))
    # Handle NaNs from integration errors
    G_mat = np.nan_to_num(G_mat)

    exponent = np.clip(-FIXED_K0 * G_mat, -700, 700)  # avoid under/overflow
    integrand = pdf[:, None] * np.exp(exponent)
    alpha = 1.0 - simpson(integrand, E_grid, axis=0)
    return np.nan_to_num(alpha)

def compute_daem(params: np.ndarray, T_vals: np.ndarray, N: int):
    """Combine *N* Weibull components weighted by w to yield total alpha(T)."""
    lambdas_w  = params[:N]
    ks_w       = params[N:2*N]
    w_raw      = params[2*N:2*2*N-1] # Corrected slice for weights
    ws = list(w_raw) + [1.0 - np.sum(w_raw)]  # last weight = 1 - sum previous

    comp_alphas = [compute_weibull_component(lambdas_w[i], ks_w[i], T_vals)
                   for i in range(N)]
    alpha_tot = sum(w*a for w, a in zip(ws, comp_alphas))
    return alpha_tot, comp_alphas, ws

# ╔══════════════════════════════════════╗
# ║ 3.  Objective & residuals             ║
# ╚═════════════════════════════════════╝

def residuals(params: np.ndarray, all_sets: list[dict], N: int) -> np.ndarray:
    """Flattened residual vector for all datasets (global fit)."""
    res = []
    for ds in all_sets:
        T_K = ds['T_K']                 # already kelvin – no +273.15!
        alpha_model, *_ = compute_daem(params, T_K, N)
        res.append(alpha_model - ds['alpha'])
    return np.concatenate(res)

def obj_scalar(params: np.ndarray, all_sets: list[dict], N: int) -> float:
    """Sum-of-squares scalar objective for DE optimizer."""
    return float(np.sum(residuals(params, all_sets, N)**2))

# ╔══════════════════════════════════════╗
# ║ 4.  Fit-quality metrics              ║
# ╚═════════════════════════════════════╝

def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot else 1.0

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def sse(y_true, y_pred):
    return np.sum((y_true - y_pred) ** 2)

def mape(y_true, y_pred):
    mask = (y_true != 0) & (y_true > 0.05)  # avoid division by zero and very small values
    if not np.any(mask):
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

# ╔══════════════════════════════════════╗
# ║ 5.  CSV loading helper               ║
# ╚═════════════════════════════════════╝

def load_csvs(file_paths: list[str]):
    """Read CSVs with columns [T_K, alpha]  (temperatures **in kelvin**)."""
    groups = []
    for idx, fp in enumerate(file_paths):
        df = pd.read_csv(fp)
        # crude sanity-check: abort if looks like Celsius
        if df.iloc[:,0].mean() < 150:
            raise ValueError(f"File '{fp}' appears to be in C (<150 K avg). Convert before running.")

        beta = ''.join(filter(str.isdigit, os.path.basename(fp)))  # extract heating rate if encoded
        label = f"Exp {idx+1}, beta = {beta} K/min"
        groups.append({
            'T_K'  : df.iloc[:,0].to_numpy(dtype=float),
            'alpha': df.iloc[:,1].to_numpy(dtype=float),
            'label': label,
        })
    return groups


# ══════════════════════════════════════════════════════════════════════════════════
# Plotting helper
# ══════════════════════════════════════════════════════════════════════════════════

# Saves a comparison plot between experimental and model values.
def save_plot(x, y_exp, y_fit, path, title, ylab):
    plt.figure()
    plt.plot(x, y_exp, label='Exp', color='black')
    plt.plot(x, y_fit, '--', label='Fit')
    plt.xlabel('Temp (C)'); plt.ylabel(ylab)
    plt.title(title); plt.legend(); plt.grid(True)
    plt.tight_layout(); plt.savefig(path, dpi=300)
    plt.close()

# ══════════════════════════════════════════════════════════════════════════════════
# Main fitting and evaluation pipeline
# ══════════════════════════════════════════════════════════════════════════════════

def run(file_list: list[str], N: int = 3, *, disable_mp: bool = False):
    datasets = load_csvs(file_list)
    print(f"Fitting {N}-Weibull DAEM to {len(datasets)} datasets...")

    # Create output directory
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = f"DAEM_MultiRate_N{N}_{ts}"
    os.makedirs(out_dir, exist_ok=True)
    # Redirect output to log file, but keep a reference to original stdout
    original_stdout = sys.stdout
    sys.stdout = open(os.path.join(out_dir, 'log.txt'), 'w')
    warnings.filterwarnings('ignore', category=UserWarning)
    print(f"Files: {file_list}")

    # Parameter bounds and initialization (for Weibull)
    # Bounds for lambda_w (scale parameter) and k_w (shape parameter)
    # These bounds might need adjustment based on the specific material data.
    # Lambda is similar to mean, k affects shape (k=1 is exponential, k~3.4 is similar to normal)
    bounds = [(5e4, 4e5)]*N + [(0.5, 10.0)]*N + [(0.0, 1.0)]*(N-1) # Bounds for lambdas, ks, weights

    # Initial guess for Weibull parameters
    # Trying to map from old Gaussian initial guesses
    # For Weibull, mean = lambda * Gamma(1 + 1/k)
    # Let's start with lambdas roughly in the range of the old mus
    lambdas_init = np.linspace(9e4, 3e5, N)
    # Let's start with k values that give a shape somewhat like the old sigmas
    # k ~ (mu / sigma)^1.086 based on approximating Weibull with Normal
    # Assuming a typical sigma might be around 8e3 to 3e4
    # Let's try a fixed k_init for now, maybe around 2-4.
    ks_init = np.full(N, 3.0) # Initial guess for shape parameter k
    weights_init = np.full(N-1, 1.0/N)
    x0 = np.concatenate([lambdas_init, ks_init, weights_init])


    # Global optimization using differential evolution
    de_obj = partial(obj_scalar, all_sets=datasets, N=N)
    de_kwargs = dict(bounds=bounds, updating='immediate', maxiter=20,
                     popsize=15, disp=True, polish=False, init='latinhypercube')
    de_kwargs['workers'] = 1 if disable_mp else -1

    result_de = differential_evolution(de_obj, **de_kwargs)
    print('\nDE finished -> Fun =', result_de.fun)

    # Local refinement using least squares
    ls_res = least_squares(lambda p: residuals(p, datasets, N), result_de.x,
                           bounds=(np.array([b[0] for b in bounds]),
                                   np.array([b[1] for b in bounds])), verbose=2)
    params_opt = ls_res.x
    np.savetxt(os.path.join(out_dir, 'params_opt.csv'), params_opt, delimiter=',')

    # --- Display optimized parameters ---
    lambdas_opt = params_opt[:N]
    ks_opt = params_opt[N:2*N]
    w_raw_opt = params_opt[2*N:2*N + (N-1)] # Corrected slice for weights
    ws_opt = list(w_raw_opt) + [1.0 - np.sum(w_raw_opt)]

    print("\nOptimized DAEM Parameters (Weibull):")
    print(f"Fixed pre-exponential factor (k0): {FIXED_K0:.2e} s-1")
    print("----------------------------------")
    for i in range(N):
        print(f"Weibull Component {i+1}:")
        print(f"  Scale Parameter (lambda_w): {lambdas_opt[i]:.2f} J/mol")
        print(f"  Shape Parameter (k_w): {ks_opt[i]:.2f}")
        print(f"  Weight (w): {ws_opt[i]:.4f}")
        print("-" * 34)

    # Also display in the notebook output (if run in a notebook)
    try:
        from IPython import get_ipython
        ip = get_ipython()
        if ip is not None:
             from IPython.display import display, Markdown
             display(Markdown("## Optimized DAEM Parameters (Weibull)"))
             display(Markdown(f"Fixed pre-exponential factor (k0): `{FIXED_K0:.2e}` s-1"))
             display(Markdown("---"))
             for i in range(N):
                 display(Markdown(f"**Weibull Component {i+1}:**"))
                 display(Markdown(f"  Scale Parameter (lambda_w): `{lambdas_opt[i]:.2f}` J/mol"))
                 display(Markdown(f"  Shape Parameter (k_w): `{ks_opt[i]:.2f}`"))
                 display(Markdown(f"  Weight (w): `{ws_opt[i]:.4f}`"))
                 display(Markdown("---"))
    except NameError:
         pass


    # --- Plot individual Weibull distributions ---
    plt.figure(figsize=(10, 6))
    # Calculate a reasonable range for plotting activation energies
    # Use percentiles for a more robust range determination for Weibull
    E_plot_points = 500
    E_plot_range = np.linspace(1.0, 500000, E_plot_points) # Adjust range as needed

    print("\nPlotting individual Weibull distributions...")
    for i in range(N):
        lambda_w = lambdas_opt[i]
        k_w = ks_opt[i]
        weight = ws_opt[i]

        # Calculate PDF
        # Ensure E_plot_range is positive for PDF calculation
        E_plot_range_pos = E_plot_range[E_plot_range > 0]
        pdf = (k_w / lambda_w) * (E_plot_range_pos / lambda_w)**(k_w - 1) * np.exp(-(E_plot_range_pos / lambda_w)**k_w)
        pdf[~np.isfinite(pdf)] = 0 # Handle potential non-finite values

        # Plot weighted PDF
        plt.plot(E_plot_range_pos, weight * pdf, label=f'Weibull {i+1} (w={weight:.4f})')
        print(f"  Plotted Weibull {i+1}")

    plt.xlabel('Activation Energy (J/mol)')
    plt.ylabel('Weighted Probability Density')
    plt.title(f'Individual Weibull Distributions (N={N})')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    weibull_plot_path = os.path.join(out_dir, 'Individual_Weibull_Distributions.png')
    plt.savefig(weibull_plot_path, dpi=300)
    plt.close()
    print(f"Individual Weibull plot saved to {weibull_plot_path}")
    # --- End of new plotting section ---


    fig_alpha, ax_alpha = plt.subplots()
    fig_dadt, ax_dadt   = plt.subplots()

    all_alphas_exp, all_alphas_fit = [], []
    metrics_rows = []

    for i, ds in enumerate(datasets):
        T_K = ds['T_K']
        alpha_exp = ds['alpha']
        T_C = T_K - 273.15  # For plotting
        alpha_fit, _, _ = compute_daem(params_opt, T_K, N)

        all_alphas_exp.append(alpha_exp)
        all_alphas_fit.append(alpha_fit)

        # Compute and store metrics
        r2_val   = r2_score(alpha_exp, alpha_fit)
        rmse_val = rmse(alpha_exp, alpha_fit)
        mae_val  = mae(alpha_exp, alpha_fit)
        sse_val  = sse(alpha_exp, alpha_fit)
        mape_val = mape(alpha_exp, alpha_fit)
        metrics_rows.append([ds['label'], r2_val, rmse_val, mae_val, sse_val, mape_val])

        # Save plots
        save_plot(T_C, alpha_exp, alpha_fit,
                  os.path.join(out_dir, f'Alpha_Fit_{i+1}.png'),
                  f"Conversion - {ds['label']}", 'alpha')

        dadt_exp = np.gradient(alpha_exp, T_C)
        dadt_fit = np.gradient(alpha_fit, T_C)
        save_plot(T_C, dadt_exp, dadt_fit,
                  os.path.join(out_dir, f'dAlpha_dT_Fit_{i+1}.png'),
                  f"dalpha/dT - {ds['label']}", 'dalpha/dT')

        ax_alpha.plot(T_C, alpha_exp, label=f"{ds['label']} Exp", lw=1)
        ax_alpha.plot(T_C, alpha_fit, '--', label=f"{ds['label']} Fit", lw=1)
        ax_dadt.plot(T_C, dadt_exp, label=f"{ds['label']} Exp", lw=1)
        ax_dadt.plot(T_C, dadt_fit, '--', label=f"{ds['label']} Fit", lw=1)

        # Export modeled alpha curve
        pd.DataFrame({'T_K': T_K, 'alpha_fit': alpha_fit}).to_csv(
            os.path.join(out_dir, f'DAEM_Curve_{i+1}.csv'), index=False)

    # Global metrics across all datasets
    y_exp_all = np.concatenate(all_alphas_exp)
    y_fit_all = np.concatenate(all_alphas_fit)
    metrics_rows.append(['GLOBAL', r2_score(y_exp_all, y_fit_all),
                         rmse(y_exp_all, y_fit_all),
                         mae(y_exp_all, y_exp_all), # Corrected this to compare exp with exp for total sum of squares
                         sse(y_exp_all, y_exp_all), # Corrected this to compare exp with exp for total sum of squares
                         mape(y_exp_all, y_fit_all)])

    pd.DataFrame(metrics_rows, columns=['Dataset', 'R2', 'RMSE', 'MAE', 'SSE', 'MAPE'])\
        .to_csv(f"{out_dir}/fit_metrics.csv", index=False)

    # Combined plots
    for ax, ylab, fname in [
        (ax_alpha, 'alpha', 'Combined_Alpha_Fit.png'),
        (ax_dadt, 'dalpha/dT', 'Combined_dAlpha_dT_Fit.png')]:
        ax.set_xlabel('Temp (C)'); ax.set_ylabel(ylab)
        ax.set_title(f'Combined {ylab} Fits (N = {N})')
        ax.legend(fontsize=8); ax.grid(True)
        ax.figure.tight_layout(); ax.figure.savefig(os.path.join(out_dir, fname), dpi=300)
        plt.close(ax.figure)

    print('\nMulti-rate fitting complete - results saved to', out_dir)

    # Restore stdout
    sys.stdout.close()
    sys.stdout = original_stdout
    print('\nMulti-rate fitting complete - results saved to', out_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multi-rate N-Weibull DAEM global fit')
    parser.add_argument('files', nargs='+', help='CSV files: Temperature (K), Conversion')
    parser.add_argument('-N', type=int, default=3, help='Number of Weibull components')
    parser.add_argument('--no-mp', action='store_true', help='Disable multiprocessing entirely')
    parser.add_argument('--profile', action='store_true', help='Enable profiling')
    args = parser.parse_args()

    if args.profile:
        import cProfile
        import pstats
        import io

        pr = cProfile.Profile()
        pr.enable()

        # Call the main function you want to profile
        # You will need to replace this placeholder with the actual call to run with your data files
        # For example: run(['/path/to/your/file1.csv', '/path/to/your/file2.csv'], N=3, disable_mp=args.no_mp)
        # print("Please replace this line with the actual call to the 'run' function with your data file paths.")
        # Placeholder call - replace with your actual call
        # run(['file1.csv', 'file2.csv'], args.N, disable_mp=args.no_mp)

        # Call the run function with the parsed arguments
        run(args.files, N=args.N, disable_mp=args.no_mp)


        pr.disable()
        s = io.StringIO()
        sortby = 'cumulative' # You can change the sort order if needed
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()  # <-- Add this line!

        # Define the output file path
        profiling_output_file = 'profiling_results.txt' # <--- This line sets the output file name

        # Save profiling results to the file
        with open(profiling_output_file, 'w') as f: # <--- This block opens the file
            f.write(s.getvalue()) # <--- This line writes the content of the StringIO stream to the file
            print(f"\n--- Profiling Results Saved ---", file=f) # Still print the confirmation within the file
            print(f"Profiling results saved to: {profiling_output_file}", file=f)
            print("------------------------------", file=f)


        print(f"Profiling complete. Results saved to {profiling_output_file}") # This prints to console/terminal


    else:
        # Original execution without profiling
        # You will need to replace this placeholder with the actual call to run with your data files
        # print("Please replace this line with the actual call to the 'run' function with your data file paths.")
        # Placeholder call - replace with your actual call
        # run(['file1.csv', 'file2.csv'], args.N, disable_mp=args.no_mp)

        # Call the run function with the parsed arguments
        run(args.files, N=args.N, disable_mp=args.no_mp)