#!/usr/bin/env python

import os
import pandas as pd 
import numpy as np
import logging
import deepsvr_utils as dp
import joblib
import gzip
import pdb
import vaex
import urllib.request
from tqdm import tqdm
import time

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV

## The script was modified to:
# 1. Allow automatic detection of VCF or VCF.GZ files
# 2. Save timing statistics for each major block of the pipeline
# 3. Obtain class 1 probability scores from the classifier
# 4. Save features before and after hard filtering steps
# 5. Save a final output with the class 1 (real mutations) probability scores, prediction, and features 
# of all SNVs including those filtered out by hard filters.
# The Hard-Filtered SNVs with assigned with score of 0 and prediction of 0 (artifact).
# 6. The VCF filtering block is retained but commented out as it is not needed for performance evaluation.
# 7. Added feature table order list to ensure consistent column ordering, to avoid any bugs resulting from edge cases.

## The deepsvr_utils.py file was also modified to:
# 1. Iron out issues with parsing the bam_readcount output.
# 2. BAM readcount does not provide avg_distance_to_effective_5p_end, therefore, this was filled in with 0s during parsing.
# 3. Discovered that missing avg_distance_to_effective_5p_end causes issues with column ordering unless explicitly handled during parsing.
# 4. The original deepsvr_utils.py file uses regular expressions to parse the bam_readcount output. 
# 5. This causes downstream issues when parsing, likely due to changes in the bam_readcount output format.
# does not parse fields in the expected order. and tries to cast non-numeric strings such as nucleotide bases to numeric types.
#. This modified version parses the bam_readcount more gracefully by splitting lines on tabs and accessing fields by their index positions.

# --- MODIFICATION: Context Manager for Timing ---
class PerformanceTimer:
    def __init__(self, name, storage_list):
        self.name = name
        self.storage_list = storage_list
        self.start_time = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.perf_counter()
        elapsed_time = end_time - self.start_time
        self.storage_list.append({
            'block': self.name,
            'time_seconds': elapsed_time
        })
# ---------------------------------------------


BASE = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

feature_table_order = [
	'tumor_ref_count', 
	'tumor_ref_avg_mapping_quality', 
	'tumor_ref_avg_basequality', 
	'tumor_ref_avg_se_mapping_quality', 
	'tumor_ref_num_plus_strand', 
	'tumor_ref_num_minus_strand', 
	'tumor_ref_avg_pos_as_fraction', 
	'tumor_ref_avg_num_mismaches_as_fraction', 
	'tumor_ref_avg_sum_mismatch_qualities', 
	'tumor_ref_num_q2_containing_reads', 
	'tumor_ref_avg_distance_to_q2_start_in_q2_reads', 
	'tumor_ref_avg_clipped_length', 
	'tumor_ref_avg_distance_to_effective_3p_end', 
	'tumor_ref_avg_distance_to_effective_5p_end', 
	'tumor_var_count', 
	'tumor_var_avg_mapping_quality', 
	'tumor_var_avg_basequality', 
	'tumor_var_avg_se_mapping_quality', 
	'tumor_var_num_plus_strand', 
	'tumor_var_num_minus_strand', 
	'tumor_var_avg_pos_as_fraction', 
	'tumor_var_avg_num_mismaches_as_fraction', 
	'tumor_var_avg_sum_mismatch_qualities', 
	'tumor_var_num_q2_containing_reads', 
	'tumor_var_avg_distance_to_q2_start_in_q2_reads', 
	'tumor_var_avg_clipped_length', 
	'tumor_var_avg_distance_to_effective_3p_end', 
	'tumor_var_avg_distance_to_effective_5p_end', 
	'tumor_other_bases_count', 
	'tumor_depth', 
	'tumor_VAF'
]

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def idx_to_snv(feature_table: pd.DataFrame) -> pd.DataFrame:
    pattern = r'~(?P<chrom>[^:]+):(?P<pos>\d+)-\d+(?P<ref>[A-Za-z]+)>(?P<alt>[A-Za-z]+)'
    # feature_table = feature_table.copy()
    snvs = feature_table.index.to_series().str.extract(pattern)
    snvs['pos'] = pd.to_numeric(snvs['pos'])
    df = pd.concat([snvs, feature_table], axis=1)
    df = df.reset_index(drop=True)
    return df

    
# def read_variants(path, snv=True):
#     df = pd.read_csv(path,
#             comment="#",
#             sep="\t",
#             header=None
#         ).iloc[:, [0, 1, 3, 4]]
    
#     df.columns = ["chrom", "pos", "ref", "alt"]

#     if snv:
#         is_snv_mask = (df['ref'].str.len() == 1) & (df['alt'].str.len() == 1)
#         df = df[is_snv_mask]

#     return df


def download_url(url, output_path):
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def filter(ref, vcf, bam, outdir, prefix, retrain, grid_search, cores, seed, loglevel):

    # Initialize timing storage
    timing_records = []


    with PerformanceTimer("Total Execution", timing_records):


        logging.basicConfig(level=loglevel,
                            format='%(asctime)s (%(relativeCreated)d ms) -> %(levelname)s: %(message)s',
                            datefmt='%I:%M:%S %p')

        logger = logging.getLogger()
        logger.info('Running FFPolish prediction')

        if not prefix:
            prefix = os.path.basename(os.path.join(outdir, bam.split('.')[0]))

        # --- Model Loading / Training Block ---
        with PerformanceTimer("Model Setup", timing_records):

            if retrain:
                if not os.path.exists(os.path.join(BASE, 'orig_train.hdf5')):
                    logger.info('Downloading original training data')
                    download_url('https://www.bcgsc.ca/downloads/morinlab/FFPolish/training_data.hdf5', 
                                os.path.join(BASE, 'orig_train.hdf5'))
                
                orig_train_df = vaex.open(os.path.join(BASE, 'orig_train.hdf5'))
                train_features = orig_train_df.get_column_names(regex='tumor')
                new_train_df = vaex.open(retrain)

                orig_train_df = orig_train_df.to_pandas_df(train_features + ['real'])
                new_train_df = new_train_df.to_pandas_df(train_features + ['real'])

                clf = LogisticRegression(penalty='l2', random_state=seed, solver='saga', max_iter=10000, 
                                        class_weight='balanced', C=0.001)
                scaler = MinMaxScaler()

                logger.info('Concatenate old and new training feature matrices')
                train_df = pd.concat([orig_train_df, new_train_df], ignore_index=True)
                X = train_df[train_features]
                y = train_df['real']

                logger.info('Scaling training feature matrix')
                X = scaler.fit_transform(X)

                logger.info('Training model')
                if grid_search:
                    logger.info('Training using grid search')
                    param = {'C': [1e-3, 1e-2, 1e-1, 1]}
                    metrics = ['accuracy', 'precision', 'recall', 'roc_auc', 'f1']
                    gs = GridSearchCV(clf, param, n_jobs=cores, scoring=metrics, cv=10, refit='f1')
                    gs.fit(X, y)
                    clf = gs.best_estimator_
                else:
                    logger.info('Training using previous optimized parameters')
                    clf.fit(X, y)

            else:
                clf = joblib.load(os.path.join(BASE, 'models', 'trained.clf'))
                scaler = joblib.load(os.path.join(BASE, 'models', 'trained.scaler'))

                # MODIFICATION: Newer Scikit-Learn requires the clip attribute
                if not hasattr(scaler, 'clip'):
                    scaler.clip = False

                os.makedirs(outdir, exist_ok=True)

        # --------------------------------------

        # --- Data Preparation Block ---
        # Includes VCF conversion, Feature Extraction (PrepareData), and Hard Filtering
        with PerformanceTimer("Data Preparation", timing_records):

            logger.info('Converting VCF to bed file')
            bed_file_path = os.path.join(outdir, prefix) + '.bed'
            if vcf.endswith(".gz"):
                os.system('zcat {} | grep PASS | vcf2bed | cut -f 1-3,6-7 > {}'.format(vcf, bed_file_path))
            else:
                os.system('cat {} | grep PASS | vcf2bed | cut -f 1-3,6-7 > {}'.format(vcf, bed_file_path))
            
            logger.info('Extracting features and preparing data')
            prep_data = dp.PrepareData(prefix, bam, bed_file_path, ref, outdir)

            df = prep_data.training_data[feature_table_order]

            features_prefilter =  idx_to_snv(df)
            features_prefilter.to_csv(f"{outdir}/{prefix}.features-prefilter.tsv", sep="\t", index=False)

            logger.info('Hard-filtering variants')
            df = df[df.tumor_VAF > 0.05]
            df = df[df.tumor_depth > 10]
            df = df[df.tumor_var_num_minus_strand + df.tumor_var_num_plus_strand > 4]

            idx_to_snv(df).to_csv(f"{outdir}/{prefix}.features-postfilter.tsv", sep="\t", index=False)

            # df = df.drop(['ref', 'var'], axis=1)

            # # --- Modification: Sanitize Data ---
            # # Replace infinite values with NaN, then fill all NaNs with 0.
            # # This prevents the "Input contains NaN, infinity" error in sklearn.
            # logger.info('Sanitizing features (removing NaNs/Infs)')
            # df = df.replace([np.inf, -np.inf], np.nan)
            # df = df.fillna(0)
            # # --- Modification END ---

        # --------------------------------

        # --- Inference Block ---
        # Includes Scaling and Prediction
        with PerformanceTimer("Scaling and Inference", timing_records):  
            logger.info('Scaling features')
            df_scaled = scaler.transform(df)

            logger.info('Obtaining predictions and probability scores')
            preds = clf.predict(df_scaled)
            # Get class 1 probability scores i.e probability if real mutations - Moyukh
            scores = clf.predict_proba(df_scaled)[:, 1]

        # --------------------------------

        with PerformanceTimer("Data Post-processing and writing", timing_records):            
            df['ffpolish_pred'] = preds
            df['score'] = scores
            df = idx_to_snv(df)

            logger.info('Obtaining SNVs from VCF')
            # snvs = read_variants(vcf)

            ## Get back SNVs dropped by Hard Filters
            ## Assign the scores to 0 as FFPolish is essentially calling them artifacts
            logger.info('Setting hard filtered variants scores to 0')
            df_all_snv = features_prefilter.merge(
                df[["chrom", "pos", "ref", "alt", "ffpolish_pred", "score"]], 
                on=["chrom", "pos", "ref", "alt"], 
                how="left"
            )
            df_all_snv.ffpolish_pred = df_all_snv.ffpolish_pred.fillna(0)
            df_all_snv.score = df_all_snv.score.fillna(0)

            # df_real = df[df.preds == 1]
            # df_arti = df[df.preds != 1]

            logger.info('Writing results to file')
            df_all_snv.to_csv(f"{outdir}/{prefix}.ffpolish.tsv", sep="\t", index=False)


        # --- VCF Filtering Block ---
        # with PerformanceTimer("VCF Filtering and writing", timing_records):
        #     logger.info('Filtering VCF')
        #     kept_vars = set(df.index.str.replace(prefix + '~', ''))

        #     vcf_file_path = os.path.join(outdir, prefix + '_filtered.vcf')
        #     with open(vcf_file_path, 'w') as f_out:
        #         with gzip.open(vcf, 'rt') as f_in:
        #             for line in f_in:
        #                 if not line.startswith('#'):
        #                     split = line.split('\t')
        #                     var = '{}:{}-{}{}>{}'.format(split[0], split[1], split[1], split[3], split[4])
        #                     if var in kept_vars:
        #                         f_out.write(line)
        #                 else:
        #                     f_out.write(line)

        # --------------------------------

    # --- Modification: Save Timing Data ---
    # Convert the list of dicts to a DataFrame
    df_timing = pd.DataFrame(timing_records)
    df_timing["time_minutes"] = round(df_timing["time_seconds"] / 60.0, 4)

    # Save to CSV
    timing_output_path = os.path.join(outdir, f"{prefix}.timing_stats.tsv")
    df_timing.to_csv(timing_output_path, sep="\t", index=False)

    logger.info(f'Timing statistics saved to {timing_output_path}')  


