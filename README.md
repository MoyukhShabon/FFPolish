# FFPolish 
Filters Artifacts From Formalin-Fixed Paraffin-Embedded (FFPE) Variants

## Installation 
### Conda Installation
Ensure that you have the `conda-forge` and `bioconda` channels in the correct priority order.

```
conda config --show channels
channels:
	- conda-forge
	- bioconda
	- defaults
```

If the above command doesn't have the correct output, run:
```
conda config --add channels defaults
conda config --add channels bioconda
conda config --add channels conda-forge
```

To create a new environment named `ffpolish` with FFPolish installed, run:
```
conda create -n ffpolish -c matnguyen ffpolish
```

Activate the environment 
```
source activate ffpolish
```

And run FFPolish, producing the help output
```
ffpolish -h
```

## Running FFPolish
### Filtering VCF
You can filter artifacts from a VCF using the pre-trained model. 

#### Input Requirements
* A reference genome in FASTA format
* A bgzipped VCF file of FFPE variants
* A BAM file of the FFPE tumour

#### Command 
The available options are:
* `-o`/`--outdir` - the output directory (default: current directory)
* `-p`/`--prefix` - the output prefix (default: basename of the BAM)

FFPolish can be run with the following command and outputs a filtered vcf `out_filtered.vcf`:
```
ffpolish filter -o outfolder -p out reference.fa vcf.gz tumour.bam
```

### Retraining Model With New Data
We recommend that if you have ground truth data (paired FFPE and fresh frozen tumours), you should create your own dataset to augment the included training set at least with a partial subset of your data.

#### Input Requirements
* A reference genome in FASTA format
* A bgzipped VCF file of FFPE variants
* A BAM file of the FFPE tumour
* A tab-delimited file of true variants
* Output directory

Tab delimited format:
| Column |    Definition    |
|:------:|:----------------:|
| chr    | Chromosome       |
| start  | Start position   |
| end    | End position     |
| ref    | Reference allele |
| alt    | Alternate allele |

#### Command
The available options are:
* `-p`/`--prefix` - the output prefix (default: basename of the BAM)

```
ffpolish extract -p out reference.fa vcf.gz tumour.bam labels.tsv outdir
```

## Custom Installation Procedure

The installation procedure laid out by the author did not work as conda could not resolve the dependency versions provided by the author. 

Dependencies were therefore manually solved via trial and error. Working set of dependencies can be installed using conda via the following command:

```
conda create -n ffpolish -c conda-forge -c bioconda python=3.8 convert_zero_one_based pandas numpy scikit-learn tqdm pysam vaex bam-readcount bedops "python-utils<3.8.0"
```

Then install ffpolish manually by navigate to the `src` directory and running:

```bash
ln -s cli.py ffpolish
chmod +x ffpolish
echo "export PATH=\$PATH:$(dirname $(readlink -f ffpolish))" >> ~/.bashrc
source ~/.bashrc
```

Then run FFPolish from the conda environment previously created:
		```bash
			conda activate ffpolish
		```

## Source Code modifications

The `filter.py` script was modified to:
1. Allow automatic detection of VCF or VCF.GZ files. Only VCF.GZ files were supported previously.
2. Save timing statistics for each major block of the pipeline (Data Preparation (Pre-Processing and Feature Extraction), Model Inference, Post Processing as well as Total Execution).
3. Obtain class 1 probability scores from the classifier. This is used to better assess performance instead of the internal hard score cutoff.
4. Save features before and after hard filtering steps. FFPolish performs hard filtering before inference (Tumor Depth > 10, Tumor VAF > 0.05, Reads Supporting Alt Allele > 4)
5. Save a final output with the class 1 (real mutations) probability scores, prediction, and features of all SNVs including those filtered out by hard filters. The Hard-Filtered SNVs were assigned with score of 0 and prediction of 0 (artifact).
6. The original code outputs a filtered VCF. This VCF filtering block is retained but commented out as it is not required for performance evaluation.
7. Added feature table order list to ensure consistent column ordering, to avoid any bugs resulting from edge cases. Refer to modifications made to `deepsvr_utils.py` for details.
8. Older versions of scikit-learn did not have the clip attribute for `MinMaxScaler` and did not clip features but newer versions expects it to be explicitly stated. Therefore, the clip attribute was explicitly added to the trained scaler.
	```python
	if not hasattr(scaler, 'clip'):
		scaler.clip = False
	```

The `deepsvr_utils.py` script was also modified to tron out issues with parsing the **bam-readcount** output:

1. The original deepsvr_utils.py file uses regular expressions to parse the **bam-readcount** output. This causes downstream issues when parsing leading to an error, likely due to changes in the **bam-readcount** output format. Aa a result, the original script does not parse fields in the expected order and and tries to cast non-numeric strings such as nucleotide bases to numeric types causing the pipeline to fail. This modified version parses the **bam-readcount** output more gracefully by splitting lines on tabs and accessing fields by their index positions.

2. **bam-readcount** also does not provide `avg_distance_to_effective_5p_end` but FFPolish expects it. It was discovered that FFPolish's author requested this feature to be added to in the BAM ReadCount repository but this does not exist as of current date. So, my assumption is that this is a placeholder feature. Therefore, `avg_distance_to_effective_5p_end` was filled in with 0s during parsing. As it was discovered that missing `avg_distance_to_effective_5p_end` messes up column orders of the final feature table leading to flawed inference, unless explicitly handled during parsing.
