import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline
import warnings
warnings.filterwarnings('ignore')

# Bearman and Harvey data
# Lift coefficient vs spin factor (S)
cl_data = {
    'v_U': np.array([0.02011326174665183, 0.02429808975566769, 0.028361959589688575, 0.034263037414684464,
                     0.043332711992731095, 0.05234865128383021, 0.06180802080978374, 0.0755282656232066,
                     0.08816041256844004, 0.10231756395769158, 0.12011631702776247, 0.13781723638914023,
                     0.15366220672364808, 0.1723131051189872, 0.19545410207272504, 0.2179257293888039,
                     0.24021326637587756, 0.263267921706261, 0.2836196578675266, 0.3037994607050791]),
    'cl': np.array([0.0808642066609761, 0.08641972419711044, 0.09074076338886916, 0.09753087700694163,
                    0.10617286120040893, 0.11358027285945053, 0.12037033938249803, 0.1290123706709904,
                    0.13703705682348855, 0.14567901746944334, 0.15679012318424945, 0.16790125244656817,
                    0.1802469306958, 0.19382718147945754, 0.20864198125002825, 0.2259259260894503,
                    0.24074074940753357, 0.2555555727256168, 0.2660493803994538, 0.27901235668965474])
}

# Drag coefficient vs spin factor (S)
cd_data = {
    'v_U': np.array([0.01988416952070005, 0.022945446364320162, 0.02632679792300783, 0.031442163981969355,
                     0.03566471185406009, 0.040223297987607626, 0.04695066828545349, 0.0570451766161739,
                     0.069310028887889, 0.08325263584038037, 0.0988609675869052, 0.11739557328800672,
                     0.13392625543378037, 0.15543266877661374, 0.17630661006044027, 0.2022880018045478,
                     0.22684090292901263, 0.26478017765410683, 0.27878767271897237]),
    'cd': np.array([0.26419753337157154, 0.26234568634368927, 0.263580258878115, 0.26234568634368927,
                    0.2629629608371459, 0.26234568634368927, 0.2629629608371459, 0.263580258878115,
                    0.26419753337157154, 0.26481483141254064, 0.2672839529338795, 0.2728395175650389,
                    0.27777778415522913, 0.28580248208148357, 0.2907407251241612, 0.29999998381108517,
                    0.30679013275042644, 0.31666666593080695, 0.3209876580275406])
}

# Define fitting functions
def linear(x, a, b):
    """Linear function: C = a * x + b"""
    return a * x + b

def poly2(x, a, b, c):
    """Quadratic function: C = a*x² + b*x + c"""
    return a * x**2 + b * x + c

def poly3(x, a, b, c, d):
    """Cubic function: C = a*x³ + b*x² + c*x + d"""
    return a * x**3 + b * x**2 + c * x + d

def power_law(x, a, n):
    """Power law function: C = a * x^n"""
    return a * np.power(x, n)

def exponential(x, a, b, c):
    """Exponential saturation: C = a * (1 - exp(-b*x)) + c"""
    return a * (1 - np.exp(-b * x)) + c

def rational(x, a, b, c):
    """Rational function: C = (a*x + b) / (c*x + 1)"""
    return (a * x + b) / (c * x + 1)


# Calculate R² for each fit
def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

# ============================================================
# FIT CL DATA
# ============================================================
print("=" * 70)
print("LIFT COEFFICIENT (CL) FITTING")
print("=" * 70)

# Linear fit for CL
popt_linear, _ = curve_fit(linear, cl_data['v_U'], cl_data['cl'])
cl_pred_linear = linear(cl_data['v_U'], *popt_linear)
r2_cl = r_squared(cl_data['cl'], cl_pred_linear)

print(f"\nLinear fit: CL = {popt_linear[0]:.6f} * S + {popt_linear[1]:.6f}")
print(f"  R² = {r2_cl:.6f}")

# ============================================================
# FIT CD DATA
# ============================================================
print("\n" + "=" * 70)
print("DRAG COEFFICIENT (CD) FITTING")
print("=" * 70)

# Quadratic fit for CD
popt_quad_cd, _ = curve_fit(poly2, cd_data['v_U'], cd_data['cd'])
cd_pred_quad = poly2(cd_data['v_U'], *popt_quad_cd)
r2_cd = r_squared(cd_data['cd'], cd_pred_quad)

print(f"\nQuadratic fit: CD = {popt_quad_cd[0]:.6f} * (S)² + {popt_quad_cd[1]:.6f} * S + {popt_quad_cd[2]:.6f}")
print(f"  R² = {r2_cd:.6f}")

# ============================================================
# PLOTTING
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Smooth curves for plotting
v_u_smooth_cl = np.linspace(cl_data['v_U'].min(), cl_data['v_U'].max(), 300)
v_u_smooth_cd = np.linspace(cd_data['v_U'].min(), cd_data['v_U'].max(), 300)

# ---- Plot 1: CL with Linear Fit ----
ax = axes[0]
ax.scatter(cl_data['v_U'], cl_data['cl'], s=120, alpha=0.7, color='steelblue', 
          edgecolors='navy', linewidth=1.5, label='Bearman & Harvey Data', zorder=5)

cl_fit_smooth = linear(v_u_smooth_cl, *popt_linear)
ax.plot(v_u_smooth_cl, cl_fit_smooth, 'r-', linewidth=3, label='Linear Fit', alpha=0.85)

# Create equation string with proper formatting
eq_cl = f'$C_L = {popt_linear[0]:.4f} \\cdot (S) + {popt_linear[1]:.4f}$\n$R^2 = {r2_cl:.5f}$'
ax.text(0.05, 0.95, eq_cl, transform=ax.transAxes, fontsize=13, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

ax.set_xlabel('Spin Factor (S)', fontsize=13, fontweight='bold')
ax.set_ylabel('Lift Coefficient ($C_L$)', fontsize=13, fontweight='bold')
ax.set_title('Bearman & Harvey: Lift Coefficient (Linear Fit)', fontsize=14, fontweight='bold')
ax.legend(fontsize=11, loc='lower right')
ax.grid(True, alpha=0.35, linestyle='--')
ax.set_axisbelow(True)

# ---- Plot 2: CD with Quadratic Fit ----
ax = axes[1]
ax.scatter(cd_data['v_U'], cd_data['cd'], s=120, alpha=0.7, color='coral', 
          edgecolors='darkred', linewidth=1.5, label='Bearman & Harvey Data', zorder=5)

cd_fit_smooth = poly2(v_u_smooth_cd, *popt_quad_cd)
ax.plot(v_u_smooth_cd, cd_fit_smooth, 'b-', linewidth=3, label='Quadratic Fit', alpha=0.85)

# Create equation string with proper formatting
eq_cd = f'$C_D = {popt_quad_cd[0]:.4f} \\cdot (S)^2 + {popt_quad_cd[1]:.4f} \\cdot (S) + {popt_quad_cd[2]:.4f}$\n$R^2 = {r2_cd:.5f}$'
ax.text(0.05, 0.95, eq_cd, transform=ax.transAxes, fontsize=13, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

ax.set_xlabel('Spin Factor (S)', fontsize=13, fontweight='bold')
ax.set_ylabel('Drag Coefficient ($C_D$)', fontsize=13, fontweight='bold')
ax.set_title('Bearman & Harvey: Drag Coefficient (Quadratic Fit)', fontsize=14, fontweight='bold')
ax.legend(fontsize=11, loc='lower right')
ax.grid(True, alpha=0.35, linestyle='--')
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig('bearman_harvey_selected_fits.png', dpi=300, bbox_inches='tight')
print("\n" + "=" * 70)
print("Figure saved as 'bearman_harvey_selected_fits.png'")
print("=" * 70)
plt.show()
