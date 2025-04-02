from _experiments.generate_descriptions import DescriptionGenerator
from configuration import config_parser

if __name__ == "__main__":
    parser = config_parser()
    args = parser.parse_args()

    descriptor = DescriptionGenerator(args)
    descriptor.run()    
