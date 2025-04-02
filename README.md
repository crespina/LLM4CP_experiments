_experiments
 ---- final_exp.py
app
 ---- data_processing
    ---- data_loaders.py
    ---- indexing.py
 ---- inference
    ---- inference.py
 ---- utils
    ---- app_utils.py
    ---- CONSTANTS.py
    ---- data_utils.py
data
 ---- input
    ---- csplib
            This folder contains a folder for every CP problems listed in the CSPLib catalog that contains a MiniZinc implementation. In each of these folders are all the MiniZinc implementations (one or more) of said problem as well as a specifications.md file, containing a human-written descriptions of the problem.
    ---- csplib_descriptions_obfuscated
            This folder contains a txt file for every problem in the csplib folder. This txt file is the same as the initial specifications.md file, except that it has been obfuscated (i.e. all mentions of the name of the problem have been erased).
    ---- csplib_models_concat
            This folder contains a txt file for every problem in the csplib folder. This txt file compiles all MiniZinc implementation(s) of the problem into a single txt file.
    ---- minizinc_source_codes
        ---- mzn
                This folder contains all the (relevant) mzn files from the MiniZinc_examples repository.
        ---- txt
                This folder contains all the mzn files from the Minzinc_examples repository that have been saved into txt files.
    ---- merged_mzn_source_codes
            This folder contains the merge of the txt files from the minizinc_source_codes folder and the csplib_models_concat folder. It is the final database used for the experiment. Nb: when a problem had a MiniZinc implementation in both of the folders, both of those were merged into a a single txt files.
 ---- output
    ---- generated_descriptions
            This folder contains a subfolder for every problem in the merged_mzn_source_codes folder. In these subfolders, they are always four files : 
                ---- beginner.txt : the generated beginner-level description of the problem
                ---- medium.txt : the generated intermediate-level description of the problem
                ---- expert.txt : the generated expert-level description of the problem
                ---- source_code.txt : the source code of the problem (from the merged_mzn_source_codes folder)
 ---- results
    ---- exp1/no_rerank : This folder contains the result of the first experiment (leave-one-out).
                The txt files in this folder are all formatted as index_xxx_level_yyy with xxx=the level(s) contained in the index in addition to the source code, and yyy=the level of the description left out and used as query. Inside of these files, there are 67 lines of text (because there are 67 problems in the database). Each line is formatted the same : the first word is the name of the problem that is tested and the five next words are the problems output by the system in its ranking.
                The exp1.txt file contains the final result in terms of MRR (i.e. the MRR is computed using all the results from the other files).
    ---- exp2/no_rerank : This folder contains the results of the second experiment (CSPLib)
                The txt files in this folder are all formatted as index_xxx with xxx=the level(s) contained in the index in addition to the source code. Inside of these files, there are 36 lines of text (because there are 36 problems that have a CSPlib description). Each line is formatted the same : the first word is the name of the problem that is tested and the five next words are the problems output by the system in its ranking.
                The exp2.txt file contains the final result in terms of MRR (i.e. the MRR is computed using all the results from the other files).
 ---- vector_dbs/code_as_text : 
    This folder contains several subfolders (beginner, beginnerexpert,  beginnermedium, beginnermediumexpert, code, expert, medium, mediumexpert), each containing a vector store index (saved using the persist function). The name of the subfolder represent what level of documents are contained in the index. 
configuration.py
requirements.txt
run_indexing.py
run_inference.py