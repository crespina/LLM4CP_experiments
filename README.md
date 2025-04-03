# CP-Model-Zoo: Experiments

This repository contains the code and data to fully reproduce the results presented in the paper
"CP-Model-Zoo: A Natural Language Query System for Constraint Programming Models".

## Project Structure

### Data

#### Input Data

- `data/input/csplib`: Contains CP problems from CSPLib with MiniZinc implementations and specifications
- `data/input/csplib_descriptions_obfuscated`: Problem descriptions with problem names removed
- `data/input/csplib_models_concat`: Compiled MiniZinc implementations for each problem
- `data/input/minizinc_source_codes`: MiniZinc example files in both `.mzn` and `.txt` formats
- `data/input/merged_mzn_source_codes`: Final database merging CSPLib and MiniZinc example implementations

#### Output Data

- `data/output/generated_descriptions`: Contains generated problem descriptions at three expertise levels:
    - `beginner.txt`: Simplified problem descriptions
    - `medium.txt`: Intermediate-level problem descriptions
    - `expert.txt`: Technical problem descriptions
    - `source_code.txt`: Original source code

#### Results

- `data/results/exp1`: Leave-one-out experiment results with MRR metrics
- `data/results/exp2`: CSPLib experiment results with MRR metrics

#### Vector Databases

- `data/vector_dbs/code_as_text`: Vector store indices for different combinations of expertise levels

## Setup

### Create a Virtual Environment

```bash
# Create a virtual environment
python -m venv venv

# Activate virtual environment
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

### Installation

```bash
# Install the required packages
pip install -r requirements.txt
```

## Usage

### API Keys Setup

To use this system, you'll need a Groq API key:

1. Generate an API key from [Groq](https://console.groq.com/)
2. Create a `.env` file in the `app/assets/env` folder with the following content:
   ```
   GROQ_API_KEY=your_groq_api_key
   ```

Alternatively, you can pass your API key directly as a command-line argument with the `--groq_api_key` parameter when
running the scripts.

### Indexing Process

To create or recreate the vector embedding databases (indices), run the indexing script:

```bash
python run_indexing.py
```

This script performs two main operations:

1. Generates problem descriptions at different expertise levels
2. Creates vector stores from the generated descriptions

The indices will be saved in the `./data/vector_dbs/code_as_text/` directory with separate subdirectories for each
expertise level.

>[!TIP]
> To add a new MiniZinc model into the database, all you need to do is to create a <the name of the problem>.txt file, containing the MiniZinc implementation into the data/input/merged_mzn_source_code folder. 
> Once this is done, simply rerun the indexing as stated above. 
> 
> This will create a new set of vector stores in the data/vector_dbs folder.

### Experiments

To run the experiments described in the paper, execute the experiments script:

```bash
python run_experiments.py
```

This script automatically performs two main experiments:

1. **Leave-One-Out Experiment**: Evaluates the system's ability to retrieve relevant models when given a query derived
   from a held-out model.
2. **CSPLib Experiment**: Evaluates retrieval performance on the CSPLib problem collection.

Results will be saved in the `data/results/exp1` and `data/results/exp2` directories, including Mean Reciprocal Rank (
MRR) metrics and detailed retrieval analyses.

### Inference CLI

For debugging and interactive exploration of the system, use the Inference Tool:

```bash
python run_inference.py --storage_dir ./data/vector_dbs/code_as_text/medium
```

The `--storage_dir` parameter specifies which embedding database (Index) you want to query:

- Beginner level: `--storage_dir ./data/vector_dbs/code_as_text/beginner`
- Medium level: `--storage_dir ./data/vector_dbs/code_as_text/medium`
- Expert level: `--storage_dir ./data/vector_dbs/code_as_text/expert`
- Combined levels: `--storage_dir ./data/vector_dbs/code_as_text/beginnermediumexpert`

Once running, you can enter questions about constraint programming algorithms and problems. The tool will display ranked
results based on relevance to your query. Type 'quit' to exit the program.
