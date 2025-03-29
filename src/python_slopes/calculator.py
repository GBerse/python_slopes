# The MIT License (MIT)
#
# Copyright (c) 2025-2026 Griffin Berse
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import math
import sys
import numpy as np
import scipy
from scipy.interpolate import interp1d
from scipy.optimize import root_scalar
from typing import List, Dict, Tuple, Optional, Callable
import matplotlib.pyplot as plt

class SlopeStabilityCalculator:
    """Class for calculating the factor of safety for a circular slip surface
     
    Parameters
    ----------
    slices:  List[Slice]
        List of slice objects that divide the slope
    piezeometric_line: Optional[interp1d]
        Linear interpolation function that represents a piezometric line in slope
    slip_circle: Optional[Dict[str, Tuple[float, float] or float]]]    
        Dictionary with center (x,y) and radius of the slip circle
    material_properties: List[Dict[str, float]]
        List of soil properties including:

        unit_weight: float
            Unit weight of soil in pcf
        cohesion: float
            Cohesion of soil in psf
        friction_angle: float
            Friction angle of soil in degrees
   
         """

    def __init__(self):
        self.slices = []
        self.piezeometric_line = None
        self.slip_circle = None
        self.material_properties = []
        self.ground_surface = None
        self.use_base_unit_weight = False

    def add_soil_layer(self,
                       unit_weight: float, 
                       cohesion: float, 
                       friction_angle: float,
                       top_elevation: float,
                       bottom_elevation: float):
        """Adds each soil layer and its respective properties"""
        self.material_properties.append({
            "unit_weight" : unit_weight,  #pcf
            "cohesion" : cohesion,        #psf
            "friction_angle" : friction_angle,   #degrees
            "top": top_elevation,
            "bottom": bottom_elevation
        })

    def add_piezeometric_line(self,x_cord,y_cord):
        """Creates a piezometric line based on coordinates; is a linear interpolation function"""
        self.piezeometric_line = interp1d(
            x_cord, y_cord,
            kind='linear',
            fill_value='extrapolate'
        )

    def add_slip_circle(self, center, radius):
        """Adds a slip circle to the slope"""
        self.slip_circle = {
            "center" : center,
            "radius" : radius
        }
    
    def set_ground_surface(self, surface_func):
        """Set a function that defines the ground surface"""
        self.ground_surface = surface_func

    def create_slices(self, 
                      num_slices: int = 10,
                      x_range: Tuple[float, float] = None,
                      slice_widths: Optional[List[float]] = None):
        """Creates slices and allows custom slice widths
        Parameters
        ---------- 
        num_slices: int
            Number of slices to create
        x_range: Tuple[float, float]
            Range of x values to create slices
        slice_widths: Optional[List[float]]
            List of custom slice widths
        """
        self.slices = []

        if x_range is None:
            #use the intersection of the circle and ground surface
            intersects = self.slip_circle_ground_surface_intersections()
            if len(intersects) >= 2:
                x_range = (intersects[0], intersects[-1])
            else:
                raise ValueError('Beep Boop, problem in calculating the intersection')
            
        if slice_widths:
            #create custom slice widths
            x_values = [x_range[0]]
            for width in slice_widths:
                self.slices.append(Slice(x, x+width, self))
                x+=width
        else:
            #create evenly spaced slices
            slice_width = (x_range[1] - x_range[0]) / num_slices
            for i in range(num_slices):
                left_x = x_range[0] + i * slice_width
                right_x = left_x + slice_width
                self.slices.append(Slice(left_x, right_x, self))
    
    def analyze(self,
                methods: List[str] = ['oms','bishop'],
                bishop_tol: float = 0.01,
                max_iter: int = 100,
                use_base_unit_weight: bool = False) ->Dict[str, float]:
        """factor of safety calcs with specified methods

            Parameters:
            -----------
            method: List[str]
                List of methods to use ('ordinary','bishop')
            bishop_tol: float
                Tolerance for Bishop Method
            max_iter: int
                Maximum number of iterations for Bishop Method
            
            Resulkts/Returns:
            -----------------
            Dict[str,float]
                Dictionary of factor of safety values for each method
        """
        self.use_base_unit_weight = use_base_unit_weight
        self._validate_slices()
        results = {}
    
        for method in methods:
            if method.lower() == 'oms':
                results['oms'] = self.OrdinaryMethodSlices()
            elif method.lower() == 'bishop':
                results['bishop'] = self.BishopMethod(tol=bishop_tol,max_iter=max_iter)
            else:
                raise ValueError(f"Unknown/not implemented method: {method}")
            
        return results

    def _validate_slices(self):
        """Makes sure slices are valid (i.e., make physical sense)"""
        if not self.slices:
            raise ValueError("No slices have been created")

        for i, slice in enumerate(self.slices):
            if not (0 < slice.width < float('inf')):
                raise ValueError(
                    f"Slice {i} has invalid width {slice.width}. "
                    f"Must be positive and finite (left_x={slice.left_x}, right_x={slice.right_x})"
                )
            if slice.height < 0:
                raise ValueError(
                    f"Slice {i} has negative height {slice.height}. "
                    f"Top ({slice.top_y_mid}) is below base ({slice.base_y_mid})"
            )
            if slice.material is None:
                raise ValueError(f"Slice {i} has no material assigned")
            if slice.base_length <= 0:
                raise ValueError(
                    f"Slice {i} has invalid base length {slice.base_length}. "
                    f"Left base ({slice.base_y_left}) and right base ({slice.base_y_right}) may coincide"
                )
            if abs(slice.alpha) > math.pi/2:
                raise ValueError(f"Slice {i} has unrealistic alpha value (over 90degrees!)")
            
    def OrdinaryMethodSlices(self) -> float:
        """Calculates factor of safety using Ordinary Method of Slices (Fellanius Method)"""
        resisting = driving = 0.0 #initializing
        for slice in self.slices:
             W = slice.weight
             alpha = slice.alpha
             L = slice.base_length
             u = slice.pore_pressure
             N = W * math.cos(alpha) - u * L

             resisting += (slice.material["cohesion"] * L + N * math.tan(math.radians(slice.material["friction_angle"])))
             driving += W * math.sin(alpha)
        
        return resisting / driving if driving != 0 else float('inf')
    
    def BishopMethod(self, tol:float, max_iter: int = 100) -> float:
        """Simplified Bishop Procedure"""
        fs = self.OrdinaryMethodSlices()
        for _ in range(max_iter):
            resisting = driving = 0.0
            for slice in self.slices:
                 N = slice.normal_force(fs)
                 resisting += (slice.material["cohesion"] * slice.base_length + N * math.tan(math.radians(slice.material["friction_angle"])))
                 driving += slice.driving_force()
            new_fs = resisting / driving if driving != 0 else float('inf')
            if abs(new_fs - fs) < tol:
                return new_fs
            fs = new_fs
        return fs

    def slip_circle_ground_surface_intersections(self, xtol=1e-4):
        """
        Returns the x and y coordinates of intersection using Brent's Method (1973) of root finding
        This is used for creating the slice range
        """
        cx, cy = self.slip_circle['center']
        r = self.slip_circle['radius']
        intersects = []

        def delta_height(x):
            """diff between circle and ground in x"""
            circle_y = cy - math.sqrt(max(0,r**2 - (x-cx)**2))
            return circle_y - self.ground_surface(x)
        
        # Using IVT to find sign changes (help optimise our search)
        x_min, x_max = cx -r, cx +r
        x_vals = np.linspace(x_min,x_max,10)
        signs = np.sign([delta_height(x) for x in x_vals])
        sign_changes = np.where(np.diff(signs))[0]

        #Using Brent's Method
        for i in sign_changes:
            try:
                sol = root_scalar(
                    delta_height,
                    bracket = [x_vals[i], x_vals[i+1]],
                    xtol = xtol
                )
                if sol.converged:
                    intersects.append(sol.root)
            except ValueError:
                continue
        return sorted(list(set(intersects)))

    def plot_analysis(self):
        fig, ax = plt.subplots(figsize=(12,8))

        #Ground Surface plot
        x_vals = [s.left_x for s in self.slices] + [self.slices[-1].right_x]
        y_vals = [self.ground_surface(x) for x in x_vals]
        ax.plot(x_vals,y_vals,'b-',label="Ground Surf")

        #Slip Circle
        if self.slip_circle:
            cx, cy = self.slip_circle['center']
            r = self.slip_circle['radius']
            circle = plt.Circle((cx,cy),r, color='r', fill=False, linestyle='--', label = 'Slip Surface')
            ax.add_patch(circle)

        #Slice depiction
        for i, slice in enumerate(self.slices):
            ax.plot([slice.left_x, slice.left_x],[self.ground_surface(slice.left_x), slice.base_y_left], 'k-', alpha = 0.35)
            ax.plot([slice.left_x,slice.right_x],[slice.base_y_left, slice.base_y_right], 'g-', lw=1.5)

        #Plotting Slip Surface and Ground Surface Intersection Points
        intersects = self.slip_circle_ground_surface_intersections()
        for i in intersects:
            y=self.ground_surface(i)
            ax.plot(i,y,'ro',markersize=8)

        for i, layer in enumerate(self.material_properties):
            # Create layer boundaries
            layer_top = np.full_like(x_vals, layer['top'])
            layer_bottom = np.full_like(x_vals, layer['bottom'])
            
            # Fill between layer boundaries
            ax.fill_between(x_vals, layer_bottom, layer_top, 
                        alpha=0.3, label=f"Layer {i+1} (γ={layer['unit_weight']} pcf)")
            
            # Add layer labels
            mid_x = np.mean(x_vals)
            ax.text(mid_x, (layer['top'] + layer['bottom']) / 2, 
                f"c={layer['cohesion']} psf\nϕ={layer['friction_angle']}°",
                ha='center', va='center', bbox=dict(facecolor='white', alpha=0.7))

        ax.set_aspect('equal')
        ax.set_xlabel("Distance [ft]")
        ax.set_ylabel("Elevation [ft]")
        ax.set_title('Slope Geometry')
        ax.legend()
        ax.grid(True)
        plt.show() 


class Slice:
    def __init__(self,left_x: float,right_x: float, calculator: SlopeStabilityCalculator) -> None:
        self._left_x = left_x
        self._right_x = right_x
        self.calculator = calculator
        self._base_y_left : None
        self._base_y_right : None
        self._top_y_mid : None
        self.material : None
        self.pore_pressure : None

    # ----------------------
    # General Geometry of the Slice
    # ----------------------

    @property
    def left_x(self):
        return self._left_x

    @property
    def right_x(self):
        return self._right_x

    @property
    def width(self):
         return self.right_x - self.left_x
    
    @property
    def mid_x(self):
         return self.left_x + (self.right_x - self.left_x) / 2
    
    # ----------------------
    # Geometry of Slice Base
    # ----------------------

    @property
    def base_y_left(self)-> float:
        if not hasattr(self, "_base_y_left"):
            self.calculate_base_geometry()
        return self._base_y_left
    
    @property
    def base_y_right(self)-> float:
        if not hasattr(self, "_base_y_right"):
            self.calculate_base_geometry()
        return self._base_y_right
    
    def calculate_base_geometry(self):
        """Where does the slice intersect the slip surface"""
        cx, cy =self.calculator.slip_circle["center"]
        r = self.calculator.slip_circle["radius"]

        #left side
        y_left = cy - math.sqrt(max(0, r**2 - (self.left_x - cx)**2))
        #Right side
        y_right = cy - math.sqrt(max(0, r**2 - (self.right_x - cx)**2))

        self._base_y_left = y_left
        self._base_y_right = y_right

    @property
    def base_y_mid(self)-> float:
        return (self.base_y_left + self.base_y_right) / 2
    
    @property
    def base_length(self)-> float:
        dx = self.right_x - self.left_x
        dy = self.base_y_right - self.base_y_left
        return math.sqrt(dx**2 + dy**2)
    
    @property
    def alpha(self)-> float:
        dx = self.right_x - self.left_x
        dy = self.base_y_right - self.base_y_left
        return math.atan2(-dy,dx)

    # ----------------------
    # Geometry of ground surface
    # ----------------------

    @property
    def top_y_mid(self)-> float:
        if not hasattr(self, "_top_y_mid"):
            self._top_y_mid = self.calculator.ground_surface(self.mid_x)
        return self._top_y_mid
    
    @property
    def height(self)-> float:
        return self.top_y_mid - self.base_y_mid
    
    # ----------------------
    # Soil Properties
    # ----------------------

    @property
    def material(self)-> Dict:
        if not hasattr(self, '_material'):
              self._determine_material()
        return self._material

    def _determine_material(self):
        mid_depth = self.base_y_mid  # using base elevation to determine material
        
        # Find the first layer where mid_depth is between top and bottom
        for layer in self.calculator.material_properties:
            if 'top' in layer and 'bottom' in layer:
                if layer['bottom'] <= mid_depth <= layer['top']:
                    self._material = layer
                    return
        
        # Default to the last layer if no match found
        self._material = self.calculator.material_properties[-1]


    # ----------------------
    # Pore Pressure(s)
    # ----------------------

    @property
    def pore_pressure(self)-> float:
        if not hasattr(self, "_pore_pressure"):
            self.calc_pore_pressure()
        return self._pore_pressure
    
    def calc_pore_pressure(self):
        if self.calculator.piezeometric_line is None:
            self._pore_pressure = 0.0
            return
        
       #Now considering the presence of a piezometric line
        piez_height = self.calculator.piezeometric_line(self.mid_x)

        #water height above base midpoint
        water_height = max(0, piez_height - self.base_y_mid)
        self._pore_pressure = water_height * 62.4 #pcf

    # ----------------------
    # Forces (used in F calculations)
    # ----------------------
    @property
    def weight(self) -> float:
        """Calculate total weight using either base material or all layers"""
        width = self.width
        slice_top = self.top_y_mid
        slice_bottom = self.base_y_mid
        height = slice_top - slice_bottom

        if self.calculator.use_base_unit_weight:
            # Use only the base material's unit weight
            return self.material["unit_weight"] * height * width
        
        # Original implementation using all layers
        total_weight = 0.0
        for layer in sorted(self.calculator.material_properties, 
                        key=lambda x: x.get('top', float('-inf')), 
                        reverse=True):
            # Skip layers completely above or below our slice
            if (layer.get('top', float('inf')) <= slice_bottom or 
                layer.get('bottom', float('-inf')) >= slice_top):
                continue
                
            # Calculate overlapping region
            layer_top = min(slice_top, layer.get('top', float('inf')))
            layer_bottom = max(slice_bottom, layer.get('bottom', float('-inf')))
            layer_height = layer_top - layer_bottom
            
            if layer_height > 0:
                total_weight += layer["unit_weight"] * layer_height * width

        return total_weight

    def normal_force(self, fs: float = 1.0) -> float:
        """Normal force for Bishop Method"""
        phi_rad = math.radians(self.material["friction_angle"])
        m_alpha = math.cos(self.alpha) + math.sin(self.alpha) * math.tan(phi_rad) / fs
        return (self.weight - self.pore_pressure * self.base_length * math.cos(self.alpha)) / m_alpha
    
    def resisting_force(self, fs: float = 1.0)-> float:
        N = self.normal_force(fs)
        return (self.material['cohesion'] * self.base_length +
                N * math.tan(math.radians(self.material['friction_angle'])))
    
    def driving_force(self)->float:
        return self.weight * math.sin(self.alpha)
    


# Initialize calculator
calculator = SlopeStabilityCalculator()
calculator.add_soil_layer(unit_weight=125, cohesion=600, friction_angle=25,bottom_elevation= 20,top_elevation=60)
calculator.add_soil_layer(unit_weight=110, cohesion=0, friction_angle=30,bottom_elevation= 0,top_elevation=20)
x_piez = [0.0, 50.0, 90.0, 130.0, 180.0]
y_piez = [40.0, 35.0, 30.0, 20.0, 20.0]
calculator.add_piezeometric_line(x_piez, y_piez)
slip_center = (100.0, 80.0)
slip_radius = 78.0
calculator.add_slip_circle(center=slip_center, radius=slip_radius)
def ground_surface(x):
    if x <= 50: return 60.0
    elif x <= 130: return 60 - (40/80) * (x - 50)
    else: return 20.0
calculator.set_ground_surface(ground_surface)
calculator.create_slices(num_slices=100)
results = calculator.analyze(methods=['oms', 'bishop'], use_base_unit_weight=False)
print("Factors of Safety:", results)
for i, s in enumerate(calculator.slices[:3]):
    print(f"Slice {i}: Weight={s.weight}, Pore Pressure={s.pore_pressure}, Alpha={s.alpha}")


calculator.plot_analysis()