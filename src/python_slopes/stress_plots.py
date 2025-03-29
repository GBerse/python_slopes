import matplotlib.pyplot as plt
import math
from python_slopes.calculations import SlopeStabilityCalculator


def plot_normal_stresses(calculator: SlopeStabilityCalculator, bishop_fs: float):
    """
    Calculate and plot total and effective normal stresses along the slip surface
    for both Ordinary Method of Slices and Simplified Bishop procedures as scatter plots.
    
    Parameters:
    -----------
    calculator : SlopeStabilityCalculator
        Initialized calculator with slices and analysis performed
    bishop_fs : float
        Factor of safety from Bishop's method (needed for stress calculations)
    """
    #Initialize
    x_coords = []
    total_stress_oms = []
    effective_stress_oms = []
    total_stress_bishop = []
    effective_stress_bishop = []
    
    for slice in calculator.slices:
        x_coords.append(slice.mid_x)
        
        # OMS
        W = slice.weight
        alpha = slice.alpha
        L = slice.base_length
        u = slice.pore_pressure
        
        # Total stress
        N_total_oms = W * math.cos(alpha)
        total_stress_oms.append(N_total_oms / L)
        
        # Eff stresses
        N_effective_oms = N_total_oms - u * L
        effective_stress_oms.append(N_effective_oms / L)
        
        # Bishop's
        N_total_bishop = slice.normal_force(bishop_fs)  # Bishop normal force already accounts for pore pressure
        m_alpha = math.cos(alpha) + math.sin(alpha) * math.tan(math.radians(slice.material["friction_angle"])) / bishop_fs
        
        # Total stress
        N_effective_bishop = N_total_bishop
        uL = u * L
        N_total_bishop = (N_effective_bishop * m_alpha + uL * math.cos(alpha)) / (math.cos(alpha))
        
        total_stress_bishop.append(N_total_bishop / L)
        effective_stress_bishop.append(N_effective_bishop / L)
    
    # make some scatter plots
    fig_total, ax_total = plt.subplots(figsize=(12, 6))
    fig_effective, ax_effective = plt.subplots(figsize=(12, 6))

    ax_total.scatter(x_coords, total_stress_oms, c='blue', label='Ordinary Method of Slices', s=50, alpha=0.7)
    ax_total.scatter(x_coords, total_stress_bishop, c='red', marker='x', label='Simplified Bishop', s=50, alpha=0.7)
    ax_total.set_xlabel('Distance along slip surface (ft)')
    ax_total.set_ylabel('Total Normal Stress (psf)')
    ax_total.set_title('Total Normal Stress Distribution Along Slip Surface')
    ax_total.legend()
    ax_total.grid(True)
    
    ax_effective.scatter(x_coords, effective_stress_oms, c='blue', label='Ordinary Method of Slices', s=50, alpha=0.7)
    ax_effective.scatter(x_coords, effective_stress_bishop, c='red', marker='x', label='Simplified Bishop', s=50, alpha=0.7)
    ax_effective.set_xlabel('Distance along slip surface (ft)')
    ax_effective.set_ylabel('Effective Normal Stress (psf)')
    ax_effective.set_title('Effective Normal Stress Distribution Along Slip Surface')
    ax_effective.legend()
    ax_effective.grid(True)
    
    plt.tight_layout()
    plt.show()