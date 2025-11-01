import os
import json

music_path = r"/home/jochen/Music/Samples"

paths = []
new_annotations = {}

for root, dirs, files in os.walk(music_path):
    for file in files:
        paths.append(os.path.join(root, file))

cwd = os.getcwd()
AAI_folder =os.path.dirname(cwd)

annotations = os.path.join(AAI_folder, "annotations.json")

with open(annotations, "r") as f:
    annotations = json.load(f)

original_length = len(annotations)
for key, value in annotations.items():
    key = os.sep+key
    for path in paths:
        if key in path:
            if path not in new_annotations:
                new_annotations[path] = value
                break
            print(f"skipped: {key}")

new_length = len(new_annotations)





with open("new_annotations.json", "w") as f:
    json.dump(new_annotations, f, indent=4)




