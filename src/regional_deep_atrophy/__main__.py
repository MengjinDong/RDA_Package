import argparse
from regional_deep_atrophy.train import RDATrainLauncher
from regional_deep_atrophy.test import RDATestLauncher
# from regional_deep_atrophy.PAIIR import PAIIRLauncher

# Create a parser
parse = argparse.ArgumentParser(
    prog="regional_deep_atrophy", description="Regional-Deep-Atrophy: a Longitudinal Package for Brain Progression Estimation and Heatmap Prediction")

# Add subparsers for the main commands
sub = parse.add_subparsers(dest='command', help='sub-command help', required=True)

# Add the Regional Deep Atrophy subparser commands
c_model_train = RDATrainLauncher(
    sub.add_parser('run_training', help='Regional Deep Atrophy model training'))

c_model_test = RDATestLauncher(
    sub.add_parser('run_test', help='Regional Deep Atrophy model testing for the whole dataset'))
    
# c_PAIIR = PAIIRLauncher(
#     sub.add_parser('PAIIR', help='Predicted-to-Actual Interscan Interval Ratio'))

# Parse the arguments
args = parse.parse_args()
args.func(args)