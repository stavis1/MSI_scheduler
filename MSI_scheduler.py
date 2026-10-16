#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 11:50:22 2026

@author: stavis
"""

import os
import sys
import uuid
import geojson
import numpy as np
import matplotlib.path as mplPath
from shapely.ops import split
from shapely import LineString, Polygon
from scipy.optimize import minimize

def make_valid_feature_collection(regions):
    features = []
    for region in regions:
        for i, polygon in enumerate(region.gjs_polygons):
            feature = geojson.Feature(id = uuid.uuid4().hex,
                                      geometry = polygon,
                                      properties={'objectType':'annotation',
                                                  'name':f'{region.name}_{i}'})
            features.append(feature)
    feature_collection = geojson.FeatureCollection(features)
    return feature_collection

class Region:
    def __init__(self, 
                 pts, 
                 name, 
                 parent_name = None,
                 area = None):
        self.parent_name = parent_name if parent_name is not None else name
        self.name = name
        self.pts = pts
        self.x = [p[0] for sublist in pts for p in sublist]
        self.xmin = min(self.x)
        self.xmax = max(self.x)
        self.y = [p[1] for sublist in pts for p in sublist]
        self.ymin = min(self.y)
        self.ymax = max(self.y)
        self.paths = [mplPath.Path(p, closed = True) for p in pts]
        self.gjs_polygons = [geojson.Polygon([c]) for c in pts]
        self.area = self.poly_area() if area is None else area
    
    def poly_area(self):
        area = 0
        for coords in self.pts:
            x = [p[0] for p in coords]
            y = [p[1] for p in coords] 
            area += 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))
        return area
    
    def __eq__(self, value):
        return (v.item() for p in self.pts for v in p) == (v.item() for p in value.pts for v in p)

def get_boundary(feature):
    if feature['geometry']['type'] == 'Polygon':
        coords = feature['geometry']['coordinates']
    else:
        coords = [c for sublist in feature['geometry']['coordinates'] for c in sublist]
    polys = [Region([c], feature['properties']['name']) for c in coords]
    boundary = max(polys, key = lambda p: p.area)
    return boundary

def cut_region(region, xcut):
    polys = [Polygon(c) for c in region.pts]
    cut = LineString([(xcut-0.1, region.ymin-1),
                      (xcut, region.ymax+1)])
    peices = [g for p in polys for g in split(p, cut).geoms]
    coords = [list(p.boundary.coords) for p in peices]
    areas = [p.area for p in peices]
    left = [i for i,c in enumerate(coords) if any(p[0] < xcut-0.1 for p in c)]
    right = [i for i,c in enumerate(coords) if any(p[0] > xcut for p in c)]
    lregion = Region([coords[i] for i in left], 
                     f'{region.parent_name}_{uuid.uuid4().hex}',
                     region.parent_name,
                     np.sum([areas[i] for i in left]))
    rregion = Region([coords[i] for i in right],
                     f'{region.parent_name}_{uuid.uuid4().hex}',
                     region.parent_name,
                     np.sum([areas[i] for i in right]))
    return (lregion, rregion)

def slices(xdeltas, region):
    xcuts = [region.xmin + np.sum(xdeltas[:i]) for i in range(1, len(xdeltas)+1)]
    new_regions = []
    for xcut in xcuts:
        lregion, region = cut_region(region, xcut)
        new_regions.append(lregion)
    new_regions.append(region)
    return new_regions

def loss(xdeltas, region):
    regions = slices(xdeltas, region)
    return np.std([r.area for r in regions])

def eq_area_slice(region, args):
    rrange = region.xmax - region.xmin
    constraint = {'type':'eq',
                  'fun':lambda x: np.sum(x) < rrange}
    
    init_width = (region.xmax - region.xmin)/args.slices_per_sample
    x_init = np.array([init_width]*(args.slices_per_sample-1))
    result = minimize(loss, 
                      x_init, 
                      args = (region,),
                      constraints = constraint,
                      method = 'COBYLA')
    xdeltas = result.x
    opt_regions = slices(xdeltas, region)
    return opt_regions

def split_tiny(region, n):
    rng = np.random.default_rng(1)
    xdeltas = [2]*(int((region.xmax - region.xmin)/2)-1)
    new_regions = slices(xdeltas, region)[:-1]
    idxs = rng.choice(range(len(new_regions)), n, replace = False)
    return [new_regions[i] for i in idxs]

def parse_geojson(args):
    with open(args.input, 'r') as gjs:
        regions = geojson.load(gjs)
    regions = regions['features']
    regions = {r['properties']['name']:r for r in regions}
    return regions

def get_samples(regions, args):
    samples = []
    for sample in args.samples:
        sample = regions[sample]
        sample = get_boundary(sample)
        sample = eq_area_slice(sample, args)
        samples.append(sample)
    return samples

def get_blank(regions, args):
    if args.blank != 'None':
        blank = regions[args.blank]
        blank = get_boundary(blank)
        blank = split_tiny(blank, 2)
        return blank
    else:
        return None

def get_QC(regions, args, samples):
    if args.QC != 'None':
        QC = regions[args.QC]
        QC = get_boundary(QC)
        QC = split_tiny(QC, len([r for s in samples for r in s])+2)
        return QC
    else:
        return None

def scheduler(samples, QC, blank):
    rng = np.random.default_rng(1)
    blocks = []
    while any(bool(s) for s in samples):
        block = []
        for sample in samples:
            if sample:
                i = rng.choice(range(len(sample)))
                block.append(sample.pop(i))
        block_idxs = rng.choice(range(len(block)), len(block), replace = False)
        block = [block[i] for i in block_idxs]
        blocks.append(block)
    schedule = [s for b in blocks for s in b]
    
    if QC is not None:
        tmp = [QC.pop()]
        for s in schedule:
            tmp.append(s)
            tmp.append(QC.pop())
        schedule = tmp
    
    if blank is not None:
        schedule = [blank.pop()]+schedule+[blank.pop()]
    return schedule

def main(args):
    regions = parse_geojson(args)
    samples = get_samples(regions, args)
    blank = get_blank(regions, args)
    QC = get_QC(regions, args, samples)
    schedule = scheduler(samples, QC, blank)
    fc = make_valid_feature_collection(schedule)
    with open(args.output, 'w') as gjs:
        geojson.dump(fc, gjs, indent = 2)

fields = {'input':('The name of the geojson file generated by QuPath.', None),
          'output':('The name of the geojson file to write.', 'regions.geojson'),
          'samples':('A space separated list of region names for samples.', None),
          'blank': ('The region name of the blank patch.', 'None'),
          'QC': ('The region name of the pooled QC.', 'None'),
          'slices_per_sample':('The number of slices each sample is split into.', 10)}
if len(sys.argv) > 1:
    import argparse
    parser = argparse.ArgumentParser()
    for field, (help, default) in fields.items():
        args = {'name_or_flags':(f'--{field}',),
                'type':type(default) if default is not None else str,
                'requred':field in ('input', 'samples'),
                'help':help}
        if default is not None:
            args['default'] = default
        parser.add_argument(**args)
    args = parser.parse_args()
    main(args)
    
else:
    import tkinter as tk
    fields['wd'] = ('the folder with your geojson file', os.getcwd())
    
    class Args:
        def __init__(self, app):
            self.input = app.input.get()
            self.output = app.output.get()
            self.samples = [s.get() for s in app.samples]
            self.blank = app.blank.get()
            self.QC = app.QC.get()
            self.slices_per_sample = app.slices_per_sample.get()
    
    class App(tk.Frame):
        def __init__(self, master):
            super().__init__(master)
            self.master = master
            self.grid()
            self.font = ('Helvetica', 15)
            
            row = 0
            for field, (help, default) in fields.items():
                if field != 'samples':
                    self.__dict__[field] = tk.StringVar() if type(default) != int else tk.IntVar()
                    label = tk.Label(self,
                                     text=help,
                                     font = self.font)
                    label.grid(row = row, column = 0)
                    entry = tk.Entry(self,
                                     textvariable=self.__dict__[field],
                                     font = self.font)
                    entry.grid(row = row, column = 1)
                    if default is not None:
                        self.__dict__[field].set(default)
                    row += 1

            #run scheduler
            self.run_scheduler_button = tk.Button(self, 
                                               text = 'run scheduler',
                                               command = self.run_scheduler,
                                               font = self.font)
            self.run_scheduler_button.grid(row = row, 
                                           column = 0)
                        
            #samples
            self.samples = []
            self.sample_rows = []
            self.add_sample_button = tk.Button(self, 
                                               text = 'add sample',
                                               command = self.add_sample,
                                               font = self.font)
            self.add_sample_button.grid(row = 0, 
                                        column = 2)
            self.add_sample()
            self.remove_sample_button = tk.Button(self, 
                                                  text = 'remove sample',
                                                  command = self.remove_sample,
                                                  font = self.font)
            self.remove_sample_button.grid(row = 0, 
                                           column = 3)
            self.samples_label = tk.Label(self,
                                    text = 'Samples',
                                    font = self.font)
            self.samples_label.grid(row = 1, 
                                    column = 2)
        
        def add_sample(self):
            value = tk.StringVar()
            self.samples.append(value)
            row = len(self.samples)
            sample = tk.Entry(self,
                            textvariable=value,
                            font = self.font)
            sample.grid(row = row,
                        column = 3)
            self.sample_rows.append(sample)
        
        def remove_sample(self):
            if self.samples:
                self.samples.pop()
                row = self.sample_rows.pop()
                row.destroy()
        
        def run_scheduler(self):
            os.chdir(self.wd.get())
            args = Args(self)
            main(args)
            self.master.destroy()

    root = tk.Tk()
    root.tk.call('tk', 'scaling', 2.0)
    myapp = App(root)
    myapp.mainloop()

