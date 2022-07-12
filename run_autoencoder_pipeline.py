import argparse
import os
import pandas as pd

from tods import generate_dataset, load_pipeline, evaluate_pipeline

default_data_path = 'datasets/anomaly/raw_data/yahoo_sub_5.csv'
parser = argparse.ArgumentParser(description='Arguments for running predefined pipelin.')
parser.add_argument('--table_path', type=str, default=default_data_path,
                    help='Input the path of the input data table')
parser.add_argument('--target_index', type=int, default=6,
                    help='Index of the ground truth (for evaluation)')
parser.add_argument('--metric',type=str, default='ALL',
                    help='Evaluation Metric (F1, F1_MACRO)')
parser.add_argument('--pipeline_path', 
                    default='pipelines/autoencoder_pipeline.json',
                    help='Input the path of the pre-built pipeline description')




args, unknown = parser.parse_known_args()
table_path = args.table_path 
target_index = args.target_index # what column is the target
pipeline_path = args.pipeline_path
metric = args.metric # F1 on both label 0 and 1

# Read data and generate dataset
df = pd.read_csv(table_path)
print("df.shape = ",df.shape)
dataset = generate_dataset(df, target_index)

# Load the default pipeline
pipeline = load_pipeline(pipeline_path)

# Run the pipeline
pipeline_result = evaluate_pipeline(dataset, pipeline, metric)
print(pipeline_result)
#raise pipeline_result.error[0]