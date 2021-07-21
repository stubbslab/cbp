from solarcell_dataset import SolarCellRun
import os
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Extract CBP photodiode charges and solar cell charges from a solar cell run.')
    parser.add_argument('datapath', type=str, help='Path to CBP data run', nargs=1)
    parser.add_argument('-o', '--output-file', default="./solarcell.npy",
                        help="Laser flux and wavelength.")
    parser.add_argument('-t', '--tag', help="String tag to select specific file names.", default="")
    parser.add_argument('-s', '--show', action='store_true', help="Make a bunch of control plots")
    parser.add_argument('-b', '--nbursts', help="Number of laser bursts (default: 5)", default=5)
    args = parser.parse_args()

    run = SolarCellRun(directory_path=args.datapath[0], tag=args.tag, nbursts=int(args.nbursts))
    if os.path.isfile(args.output_file):
        print(f"File {args.output_file} already exists. I load it instead of re-analyzing everything.")
        run.load_from_file(args.output_file)
    else:
        run.load()
        run.solarcell_characterization()
        run.save(args.output_file)
    if args.show:
        run.plot_summary()
