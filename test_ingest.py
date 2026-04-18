import sys
from pathlib import Path
sys.path.insert(0, str(Path("/home/all_might/Documents/ProjectCodeForPurpose/Talk_to_Data")))
from main_pipeline import AIQuerySystem

system = AIQuerySystem()

# create test txt
with open("test_file.txt", "w") as f:
    f.write("This is a test unstructured text.")

res = system.upload_file("test_file.txt", original_file_name="test_file.txt")
print(res)

with open("test_file.csv", "w") as f:
    f.write("id,name\n1,test\n")

res2 = system.upload_file("test_file.csv", original_file_name="test_file.csv")
print(res2)
