#!/usr/bin/env python3

import os
import sys
import argparse
import pandas as pd
import lxml.etree as ET

def read_xml(file):
    return ET.parse(file, ET.XMLParser(remove_blank_text=True))

space=10
color='1 0.9 0 0.7'


def add_waypoints(input_file, output_file, waypoints_file):

    try:
        # create trajectory data frame
        traj_df = pd.read_csv(waypoints_file, header=None, usecols=[0, 1], names=['X', 'Y'])

        # read xml file
        tree = read_xml(input_file)

    except TypeError:
        if not input_file:
            print('No XML file provided...\n')
        else:
            print(str(input_file) + ' not found. Check XML file path.')
        sys.exit(0)

    # get root of xml tree
    root = tree.getroot()

    # get worldbody subelement from root
    worldbody = root.find('worldbody')

    for idx, pos in enumerate(traj_df.values[20::int(space)], start=1):
        # create a waypoint subelement
        ET.SubElement(worldbody, 'geom', {'name': 'waypoint{}'.format(idx),
                                          'pos': '{} {} 1.01 '.format(pos[0], pos[1]),
                                          'size': '0.03 0.03 0.03',
                                          'type': 'sphere',
                                          'contype': '0',
                                          'rgba': color})

    # add to root
    tree.write(output_file, encoding='utf-8', pretty_print=True, xml_declaration=True)


if __name__ == "__main__":
    add_waypoints("default")
