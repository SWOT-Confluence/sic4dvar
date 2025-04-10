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
    return (pdf, bin_edges)

def compute_cdf(data, bins=20):
    pdf, bin_edges = compute_pdf(data, bins)
    cdf = np.cumsum(pdf) / np.sum(pdf)
    return (cdf, bin_edges)

def radar_factory(num_vars, frame='circle'):
    theta = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)

    class RadarAxes(PolarAxes):
        name = 'radar'

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            return super().fill(*args, closed=closed, **kwargs)

        def plot(self, *args, **kwargs):
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            if x[0] != x[-1]:
                x = np.concatenate((x, [x[0]]))
                y = np.concatenate((y, [y[0]]))
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars, radius=0.5, edgecolor='k')
            else:
                raise ValueError("unknown value for 'frame': %s" % frame)

        def draw(self, renderer):
            if frame == 'polygon':
                gridlines = self.yaxis.get_gridlines()
                for gl in gridlines:
                    gl.get_path()._interpolation_steps = num_vars
            super().draw(renderer)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                spine = Spine(axes=self, spine_type='circle', path=Path.unit_regular_polygon(num_vars))
                spine.set_transform(Affine2D().scale(0.5).translate(0.5, 0.5) + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("unknown value for 'frame': %s" % frame)
    register_projection(RadarAxes)
    return theta

def main():
    from scipy.io import loadmat
    data = loadmat('/home/ccazals/SIC4DVar/Test_SOS_filtering/verif_CDF_POM/results_algo5_pom.mat')
    for d in data['results_RMSE']:
        cdf, bin_edges = compute_cdf(d, bins=50)
        plt.plot(bin_edges, np.concatenate(([0], cdf)))
        plt.xlim((0, 1.4))
    plt.grid()
    plt.show()
if __name__ == '__main__':
    main()