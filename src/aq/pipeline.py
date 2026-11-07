"""Full, portable pipeline: python -m src.aq.pipeline [--full | --quick]."""
import argparse
import subprocess
import sys
import time

from src.common.config import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="download the public data for the period before modeling")
    parser.add_argument("--quick", action="store_true", help="demo: train the production models, forecast and draw a map; keeps the versioned test metrics")
    args = parser.parse_args()
    if args.quick and args.full:
        parser.error("--quick uses the sample; run --full separately")
    commands = []
    if args.full:
        commands.extend((f"src.aq.data.{name}", []) for name in
                        ("openaq", "era5", "goes_aod", "goes_fire", "s5p_no2", "geo", "static"))
    commands.append(("src.aq.build_dataset", [] if args.full else ["--from-sample"]))
    if args.quick:
        print("Quick demo: the versioned test CSVs will not be recomputed.", flush=True)
        commands.extend([("src.aq.train", ["--fit-only"]),
                         ("src.aq.cli", ["--municipio", "Sorocaba", "--date", "2021-08-23", "--horizon", "1"]),
                         ("src.aq.maps", ["--date", "2022-10-11"])])
    else:
        commands.extend((f"src.aq.{name}", []) for name in ("benchmark", "train", "evaluate"))
        commands.extend(("src.aq.maps", ["--date", date]) for date in ("2022-10-11", "2022-11-16", "2021-08-24"))
        commands.append(("src.aq.maps", ["--date", "2021-08-23", "--horizon", "1"]))
    started = time.monotonic()
    for module, arguments in commands:
        print(f"Running {module} {' '.join(arguments)}", flush=True)
        subprocess.run([sys.executable, "-m", module, *arguments], cwd=ROOT, check=True)
    print(f"Pipeline finished in {(time.monotonic() - started) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
