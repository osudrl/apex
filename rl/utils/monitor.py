from bokeh.layouts import gridplot
from bokeh.plotting import curdoc, figure
from bokeh.models import ColumnDataSource
from functools import partial
from itertools import cycle
from tornado import gen


class Monitor():
    def __init__(self):
        self.doc = curdoc()
        self.sources = {}

        self.color = cycle(['#348ABD', '#7A68A6', '#A60628', '#467821',
                            '#CF4457', '#188487', '#E24A33'])

        self.grid = [[None, None, None, None]]
        self.cur_row = 0
        self.cur_col = 0

        self.layout = gridplot([[]], sizing_mode="scale_width")

        self.doc.add_root(self.layout)

        self.xtick = {}

    def add_plot(self, name, xlabel=None, ylabel=None):
        self.sources[name] = ColumnDataSource(data=dict(x=[], y=[]))

        if xlabel is None:
            xlabel = "Iteration"
        if ylabel is None:
            ylabel = name

        self.xtick[name] = 0

        p = figure(
            title=name,
            x_axis_label=xlabel,
            y_axis_label=ylabel,
            sizing_mode="scale_width"
        )

        l = p.line(
            x='x',
            y='y',
            line_width=2,
            line_color=next(self.color),
            source=self.sources[name]
        )

        if self.cur_col < 4:
            self.grid[self.cur_row][self.cur_col] = p
            self.cur_col += 1

            self.doc.remove_root(self.layout)
            self.layout = gridplot(self.grid, sizing_mode="scale_width")
            self.doc.add_root(self.layout)
        else:
            self.grid.append([p, None, None, None])
            self.cur_row += 1
            self.cur_col = 1

            self.doc.remove_root(self.layout)
            self.layout = gridplot(self.grid, sizing_mode="scale_width")
            self.doc.add_root(self.layout)

    @gen.coroutine
    def _update(self, name, x, y):
        self.sources[name].stream(dict(x=[x], y=[y]))

    def update(self, name, y, x=None):
        if x is None:
            x = self.xtick[name]
            self.xtick[name] += 1

        self.doc.add_next_tick_callback(
            partial(self._update, name=name, x=x, y=y)
        )
