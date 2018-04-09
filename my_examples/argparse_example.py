# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 16:29:17 2018

@author: zdu
"""

import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--square", help="display a square of a given number",
                    type=int, default = 10)
#args = parser.parse_args()

def main(args):
	args = parser.parse_args(args[1:])		
#	args = parser.parse_args
	print(args.square**2)

if __name__ == '__main__':
	main(sys.argv)