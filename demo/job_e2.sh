#!/bin/bash

# Define the list of datasets
datasets=(
    "test"
    # "allisonramsey"
    # "brohn"
    # "centurycommunities"
    # "harrisdoyle"
    # "homeplans"
    # "lennar"
    # "lgi"
    # "markstewart"
    # "nvhomes"
    # "pulte"
    # "yourarborhome"
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

/home/ubuntu/dataset/

# Loop through each dataset
for datasetname in "${datasets[@]}"
do
    # Loop through each checkpoint
    for checkpoint in "${checkpoints[@]}"
    do
        # Define paths
        input_dir="~/dataset/expriment_three_1/${datasetname}"
        output_dir="../output/test_inf/${checkpoint}/${datasetname}"
        config_file="../configs/coco/panoptic-segmentation/maskdino_higharc_brochure_R50_bs16_50ep_3s_dowsample1_2048_e2.yaml"
        weights_file="../output/model_${checkpoint}.pth"

        # Create output directory if it doesn't exist
        mkdir -p "${output_dir}"

        # Run the command
        python demo_1_e2.py --config-file "${config_file}" --input "${input_dir}" --output "${output_dir}" --opts MODEL.WEIGHTS "${weights_file}"

        # Optionally, you can add a newline or separator for clarity
        echo "Finished processing ${datasetname} with model ${checkpoint}"
        echo "---------------------------------------"
    done
done