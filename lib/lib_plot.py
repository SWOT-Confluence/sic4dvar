import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D

DPI = 300

def compute_pdf(data, bins=20):
    out, bin_edges = pd.qcut(data, bins, retbins=True, duplicates='drop')
    pdf, bin_edges = np.histogram(data, bins=bin_edges, density=False)
    
    return pdf, bin_edges
    
def compute_cdf(data, bins=20):
    pdf, bin_edges = compute_pdf(data, bins)
    cdf = np.cumsum(pdf) / np.sum(pdf)

    return cdf, bin_edges

def radar_factory(num_vars, frame='circle'):
    """Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle' | 'polygon'}
        Shape of frame surrounding axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)
    
    class RadarAxes(PolarAxes):
        
        name = 'radar'
        
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')
        
        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)
        
        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)
        
        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.concatenate((x, [x[0]]))
                y = np.concatenate((y, [y[0]]))
                line.set_data(x, y)
        
        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)
        
        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError("unknown value for 'frame': %s" % frame)
        
        def draw(self, renderer):
            """ Draw. If frame is polygon, make gridlines polygon-shaped """
            if frame == 'polygon':
                gridlines = self.yaxis.get_gridlines()
                for gl in gridlines:
                    gl.get_path()._interpolation_steps = num_vars
            super().draw(renderer)
        
        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)
                
                return {'polar': spine}
            else:
                raise ValueError("unknown value for 'frame': %s" % frame)
    
    register_projection(RadarAxes)
    return theta


def main():
    # nrmse = [1.4070276599927, 0.366644703706346, 0.712732517737094, 0.879451087463557, 1.2783133291189, 6.0662682419847,
    # 1.30739332124767, 1.13808240535342, 1.21371078523566, 0.459622260332796, 9.195253204065, 2.1498158280966, 1.31713215970313,
    # 1.26767019172246, 2.05781945998904, 0.315131365373192, 2.85495062736148, 0.626182636530603, 0.55982809457064, 0.702126264732896,
    # 0.454709200252366, 9.31291610185166, 0.907635144751162, 1.43970133448933, 9.92017516459473, 0.578005936946969, 0.62689438554856,
    # 0.361567326357788, 1.61817702589417, 0.231684163058234, 0.635650324596971, 0.352990362428902, 0.626848395371685, 0.547599733809922,
    # 0.509172302759967, 0.696207136154895, 0.505602015311265, 0.468920786497396, 0.348869002540406, 0.441055523941531, 1.17852868668364,
    # 0.89634950415783, 0.971624504543823, 0.999489503649545, 0.978806071189704, 1.03448595297404, 1.00799272905899, 0.616982935869336,
    # 0.733653121144741, 0.933436664637157, 0.849564643273146, 0.896979260759504, 0.62735618696073, 2, 7.06515456064, 0.752462035180555,
    # 0.649747754012518, 0.570532305543382, 0.722005158953771, 1.0709343047604, 1.26230757755509, 0.822251456893947, 0.791639558051904,
    # 0.723270032078105, 0.590009540243446, 0.214763713657475, 0.218873822774722, 0.298261205362828, 0.241535277113693, 0.799078289292124,
    # 4.92627300830022, 3.7754189552518, 3.00113207491804, 1.99655668389135, 0.646217014984416, 2.74149802279532, 1.1635854517533,
    # 5.13195938831419, 0.899830428178329, 1, 1.961782694247, 1.81656147725857, 0.508610444646104, 2.0281062191761, 1.2590638196491,
    # 1.6025743038284, 11, 7.67695309555, 0.987869902419652, 4.110747002748, 1.03524622857754, 3.8646702147316, 7.99618290456638,
    # 3.9430923026453, 2.2614663826443, 1.47612964696634, 2.67530167707327, 255, 9.1870117793, 1800, 0.191777791, 1.99786800413993,
    # 1.89335996151184, 3.91379105679, 0.926835259927687, 0.79051251036513, 0.26085354201987, 0.262465281324961, 0.261018746249666,
    # 0.476137054436081, 0.32266821520761, 3, 6.64283985682, 7.10523616107239, 4.35904089622733, 0.923038136277615, 0.4760006539264,
    # 2.12093792568087, 3.06180302553964, 3, 6.11901319027, 2.32138839043827, 0.648384167620287, 0.969126780873015, 1.23455112958611,
    # 0.437634332215873, 0.663655630038384, 0.484319039285061, 0.553460938028559, 7.04863629411516, 0.810618776640137, 2.00194922379098,
    # 1.5614937574541, 1.80729367402754, 0.347730420819964, 0.628645968150048, 1.1581130972982, 2.14189209830291, 0.618251255187153,
    # 0.341923191605141, 0.351277525331619, 0.418286574575824, 0.32565533470778, 0.602221199674181, 1.32726328275794, 0.863085001111412,
    # 1.61319046698867, 0.963580490859026, 1.52654510563994, 0.907734399095894, 0.585175114975599, 0.564134838539565, 0.736782649213974,
    # 0.543052033708234, 0.539663347334009, 0.440083276353404, 0.724450488941779, 0.490401508078588, 1.63286247006698, 1.17619763023728,
    # 0.665326745463655, 0.696647076269848, 0.668389713041184, 0.711343972471826, 1.1860070439367, 0.66708815193047, 0.495411536478338]
    from scipy.io import loadmat
    
    data = loadmat('/home/ccazals/SIC4DVar/Test_SOS_filtering/verif_CDF_POM/results_algo5_pom.mat')  # Charger le fichier
    
    
    for d in data['results_RMSE']:
        cdf, bin_edges = compute_cdf(d, bins=50)
        plt.plot(bin_edges, np.concatenate(([0], cdf)))
        plt.xlim((0, 1.4))
    
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()

