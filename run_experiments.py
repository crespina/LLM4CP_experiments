from app.experiments.Experiment1 import Experiment1
from app.experiments.Experiment2 import Experiment2
from configuration import config_parser


def main():
    # Parse command line arguments
    parser = config_parser()
    args = parser.parse_args()

    # Run Experiment 1
    print("Running Experiment 1...")
    exp1 = Experiment1(args=args)
    exp1.run()

    # Run Experiment 2
    print("\nRunning Experiment 2...")
    exp2 = Experiment2(args=args)
    exp2.run()


if __name__ == "__main__":
    main()
