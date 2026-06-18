import json
import cadquery as cq
import os

# Your input file
jsonl_file = "/home/jacob/CADBench/tested_models/cadrecode/bench0.json"

with open(jsonl_file, "r") as f:
    for line in f:
        data = json.loads(line)
        file_id = data["file_id"]
        code_str = data["generated"]
        
        # Create a dictionary to act as the local namespace for exec()
        local_vars = {"cq": cq}
        
        try:
            # Execute the string. This populates local_vars with w0, r, etc.
            exec(code_str, {"cq": cq}, local_vars)
            
            # Identify the CadQuery object in the local variables.
            # Usually, the 'result' is the last assigned variable (like 'r' in your example)
            # or we can look for any Workplane object.
            result = None
            for var_name in reversed(list(local_vars.keys())):
                if isinstance(local_vars[var_name], cq.Workplane):
                    result = local_vars[var_name]
                    break
            
            if result:
                output_filename = f"{file_id}.step"
                # Export to STEP
                cq.exporters.export(result, output_filename)
                print(f"Successfully exported {output_filename}")
            else:
                print(f"Warning: No Workplane object found in code for {file_id}")
                
        except Exception as e:
            print(f"Error processing {file_id}: {e}")