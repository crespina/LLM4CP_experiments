data
 ---- input
    ---- csplib
            This folder contains a folder for every CP problems listed in the CSPLib catalog that contains a MiniZinc implementation. In each of these folders are all the MiniZinc implementations (one or more) of said problem as well as a specifications.md file, containing a human-written descriptions of the problem
    ---- csplib_descriptions_obfuscated
            This folder contains a txt file for every problem in the csplib folder. This txt file is the same as the initial specifications.md file, except that it has been obfuscated (i.e. all mentions of the name of the problem have been erased).
    ---- csplib_models_concat
            This folder contains a txt file for every problem in the csplib folder. This txt file compiles all MiniZinc implementation(s) of the problem into a single txt file.
    ---- minizinc_source_codes
        ---- mzn
                This folder contains all the mzn files from the 
        ---- txt
 ---- output
    ---- generated_descriptions
 ---- results
    ---- exp1
    ---- exp2
 ---- vector_dbs
    ---- code_as_text
        ---- beginner
        ---- beginnerexpert
        ---- beginnermedium
        ---- beginnermediumexpert
        ---- code
        ---- expert
        ---- medium
        ---- mediumexpert