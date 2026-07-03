"""JASMIN command-line interface.

  jasmin collect [--days N] [--offline]      run all collectors
  jasmin build-dataset                       validate + clean + engineer + label
  jasmin train                               train & register a model version
  jasmin predict SYMBOL [SYMBOL...]          predict with explanation
  jasmin cycle [--offline]                   full collect->train->predict cycle
  jasmin models                              list registry versions
  jasmin approve VERSION / rollback          manage which model is live
  jasmin serve [--port P]                    start the FastAPI service
  jasmin daemon [--interval-hours H]         continuous-learning loop
"""

from __future__ import annotations

import argparse
import json
import sys

from jasmin.config import PipelineConfig, ensure_dirs


def _print(obj) -> None:
    print(json.dumps(obj, indent=2, default=str))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="jasmin", description="JASMIN AI pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    p_collect = sub.add_parser("collect", help="run all data collectors")
    p_collect.add_argument("--days", type=int, default=400)
    p_collect.add_argument("--offline", action="store_true",
                           help="force synthetic sources (no network)")

    sub.add_parser("build-dataset", help="build the master dataset from raw data")
    sub.add_parser("train", help="train and register a new model version")

    p_predict = sub.add_parser("predict", help="predict one or more symbols")
    p_predict.add_argument("symbols", nargs="+")

    p_cycle = sub.add_parser("cycle", help="run the full pipeline end to end")
    p_cycle.add_argument("--days", type=int, default=400)
    p_cycle.add_argument("--offline", action="store_true")

    sub.add_parser("models", help="list model registry versions")
    p_approve = sub.add_parser("approve", help="approve a model version")
    p_approve.add_argument("version")
    sub.add_parser("rollback", help="revert to the previous approved model")

    p_serve = sub.add_parser("serve", help="start the API server")
    p_serve.add_argument("--host", default="0.0.0.0")
    p_serve.add_argument("--port", type=int, default=8000)

    p_daemon = sub.add_parser("daemon", help="run the continuous-learning daemon")
    p_daemon.add_argument("--interval-hours", type=float, default=24)
    p_daemon.add_argument("--offline", action="store_true")

    args = parser.parse_args(argv)
    config = PipelineConfig()
    ensure_dirs()

    if args.command == "collect":
        from jasmin.pipeline import run_collectors
        run_collectors(config, days=args.days, offline=args.offline)
    elif args.command == "build-dataset":
        from jasmin.dataset import build_master_dataset
        master = build_master_dataset(config)
        _print({"rows": len(master), "columns": len(master.columns),
                "symbols": sorted(master["symbol"].unique().tolist())})
    elif args.command == "train":
        from jasmin.models.train import train_models
        _print(train_models(config))
    elif args.command == "predict":
        from jasmin.prediction import predict
        for symbol in args.symbols:
            _print(predict(symbol.upper(), config=config).to_dict())
    elif args.command == "cycle":
        from jasmin.pipeline import run_cycle
        result = run_cycle(config, days=args.days, offline=args.offline)
        _print(result["training"])
        for pred in result["predictions"]:
            print(f"{pred['symbol']}: {pred['direction']} "
                  f"(p_up={pred['probability_up']}, move={pred['expected_move_pct']}%, "
                  f"confidence={pred['confidence']['score']})")
    elif args.command == "models":
        from jasmin.models.registry import ModelRegistry
        _print(ModelRegistry().list_versions())
    elif args.command == "approve":
        from jasmin.models.registry import ModelRegistry
        ModelRegistry().set_approval(args.version, True)
        print(f"approved {args.version}")
    elif args.command == "rollback":
        from jasmin.models.registry import ModelRegistry
        print(f"live model is now {ModelRegistry().rollback()}")
    elif args.command == "serve":
        import uvicorn
        uvicorn.run("jasmin.api.app:app", host=args.host, port=args.port)
    elif args.command == "daemon":
        from jasmin.scheduler.daemon import run_daemon
        run_daemon(interval_hours=args.interval_hours, offline=args.offline, config=config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
