#!/bin/bash
main_folder="/Users/hassan/Downloads/Caltech/"
tmp_dir="/Users/hassan/Downloads/Caltech/tmp"
output_dir="/Users/hassan/Downloads/Caltech/derivatives/"

mkdir $output_dir
# Loop through all subject folders
for subject_folder in "${main_folder}"/*; do
    if [ ! -d "$subject_folder" ]; then
        continue
    fi
    subject_name=$(basename "$subject_folder")
    session1_path="${subject_folder}/session_1"
    anat_1_path="${session1_path}/anat_1"

    # Loop through all anat_1 folders
    for anat_folder_path in "${anat_1_path}"/*; do
      [ -e "$anat_folder_path" ] || continue
       if test -f "${output_dir}$(basename anat_folder_path)"; then
           echo "$(basename anat_folder_path) exists."
           continue
       fi
        # Execute the commands
        eval "rm -dr ${tmp_dir}"
        mkdir $tmp_dir
        tmp_image="${tmp_dir}/sub"
        3drefit -deoblique "${anat_folder_path}"
        3dresample -orient RPI -inset ${anat_folder_path} -prefix ${tmp_image}
        3dSkullStrip -input "${tmp_image}+orig" -prefix "${output_dir}${subject_name}"
        3dAFNItoNIFTI -prefix "${output_dir}${subject_name}" "${output_dir}${subject_name}+orig"
    done
done
rm "${output_dir}*orig*"