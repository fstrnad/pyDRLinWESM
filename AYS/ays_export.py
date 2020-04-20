#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
"""
adapted from https://github.com/timkittel/ays-model/
"""

from __future__ import generators, print_function, division

import AYS.ays_general as ays_general
from pyviability import libviability as lv

import numpy as np


import os
import argparse, argcomplete

if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description="Export an AWS - TSM file to text.",
    )
    parser.add_argument("input_file", metavar="input-file",
                        help="file with the tsm data")
    parser.add_argument("txt_file", metavar="txt-file", nargs="?", default="",
                        help="output text file")
    parser.add_argument("-f", "--force", action="store_true",
                        help="overwrite text file if already existing")

    # use argcomplete auto-completion
    argcomplete.autocomplete(parser)

    args = parser.parse_args()

    if args.txt_file and (not args.force) :
        if os.path.isfile(args.txt_file):
            parser.error("'{}' exists already, use '--force' option to overwrite".format(args.txt_file))

    if args.txt_file == args.input_file:
        parser.error("'txt-file' and 'output-file' should be different from each other, not both '{}'".format(args.input_file))

    header, data = ays_general.load_result_file(args.input_file)

    header_txt = "#"*80 + "\n"
    header_txt += ays_general.recursive_dict2string(header)
    header_txt += "#"*80 + "\n"

    for region in lv.REGIONS:
        header_txt += "{} = {:>2d}\n".format(region, getattr(lv, region))
    header_txt += "#"*80

    states = data["states"]

    print(header_txt)

    if args.txt_file:
        print("saving to {!r} ... ".format(args.txt_file), end="", flush=True)
        np.savetxt(args.txt_file, states, fmt="%i", header=header_txt, comments="")
        print("done")
