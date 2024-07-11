#!/bin/bash

# Define the list of datasets
datasets=(
    "allisonramsey"
    "brohn"
    "centurycommunities"
    "harrisdoyle"
    "homeplans"
    "lennar"
    "lgi"
    "markstewart"
    "nvhomes"
    "pulte"
    "yourarborhome"
)

# Define the list of model checkpoints
checkpoints=(
    "0004999"
    "0009999"
    "0014999"
    "0019999"
    "0024999"
    "0029999"
    "0034999"
)

# Loop through each dataset
for datasetname in "${datasets[@]}"
do
    # Loop through each checkpoint
    for checkpoint in "${checkpoints[@]}"
    do
        # Define paths
        input_dir="~/dataset/samples_0624/${datasetname}"
        output_dir="../output_experiment_two/output/${checkpoint}/${datasetname}"
        config_file="../configs/coco/panoptic-segmentation/maskdino_higharc_brochure_R50_bs16_50ep_3s_dowsample1_2048_e2.yaml"
        weights_file="../output_experiment_two/model_${checkpoint}.pth"

        # Create output directory if it doesn't exist
        mkdir -p "${output_dir}"

        # Run the command
        python demo_1.py --config-file "${config_file}" --input "${input_dir}" --output "${output_dir}" --opts MODEL.WEIGHTS "${weights_file}"

        # Optionally, you can add a newline or separator for clarity
        echo "Finished processing ${datasetname} with model ${checkpoint}"
        echo "---------------------------------------"
    done
done
