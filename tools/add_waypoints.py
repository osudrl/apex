#!/usr/bin/env python3

import sys
import argparse
import pandas as pd
import lxml.etree as ET


def get_args():
    parser = argparse.ArgumentParser(description='Add waypoints to any cassie environment')
    parser.add_argument('-i', action='store', dest='input',
                        help='XML file to modify')
    parser.add_argument('-o', action='store', dest='output',
                        help='Output file to write modified XML file', default='cassie_waypoint.xml')
    parser.add_argument('-w', action='store', dest='waypoints',
                        help='File containing waypoints to add to XML file')
    parser.add_argument('-c', action='store', dest='color',
                        help='RGBA values for waypoints color (i.e. 1 0.9 0 0.7)', default='1 0.9 0 0.7')
    parser.add_argument('-s', action='store', dest='space',
                        help='How far points are spaced out (default=10)', default=10)

    return parser


def read_xml(file):
    return ET.parse(file, ET.XMLParser(remove_blank_text=True))


def main():
    parser = get_args()
    args = parser.parse_args()

    try:
        # create trajectory data frame
        traj_df = pd.read_csv(args.waypoints, header=None, usecols=[0, 1], names=['X', 'Y'])

        # read xml file
        tree = read_xml(args.input)
    except ValueError:
        if not args.waypoints:
            print('No trajectory file provided...\n')
            parser.print_help()
        else:
            print(str(args.waypoints) + ' not found. Check trajectory file path.')
        sys.exit(0)

    except TypeError:
        if not args.input:
            print('No XML file provided...\n')
            parser.print_help()
        else:
            print(str(args.input) + ' not found. Check XML file path.')
        sys.exit(0)

    # get root of xml tree
    root = tree.getroot()

    # get worldbody subelement from root
    worldbody = root.find('worldbody')

    for idx, pos in enumerate(traj_df.values[20::int(args.space)], start=1):
        # create a waypoint subelement
        ET.SubElement(worldbody, 'geom', {'name': 'waypoint{}'.format(idx),
                                          'pos': '{} {} 1.01 '.format(pos[0], pos[1]),
                                          'size': '0.03 0.03 0.03',
                                          'type': 'sphere',
                                          'contype': '0',
                                          'rgba': args.color})

    # add to root
    tree.write(args.output, encoding='utf-8', pretty_print=True, xml_declaration=True)


if __name__ == "__main__":
    main()
