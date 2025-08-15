#!/bin/bash

# Ensure script exits on error
set -e

# Check if gdown is installed, install if not
if ! command -v gdown &> /dev/null
then
    echo "gdown not installed, installing..."
    pip install gdown
fi

# Create data folder if it doesn't exist
mkdir -p data

# Define Google Drive file ID and output filename
FILE_ID="1FQMyKLLd6hP9dw9rfZn1EZOWNvKaDsqw"
OUTPUT_FILE="temp.zip"

# Download file
 echo "Downloading file from Google Drive..."
gdown --id $FILE_ID -O $OUTPUT_FILE

# Extract file to data folder
echo "Extracting file to data folder..."
unzip -q $OUTPUT_FILE -d data

# Clean up temporary file
echo "Cleaning up temporary file..."
rm -f $OUTPUT_FILE

echo "Download and extraction completed! Files have been saved to the data folder."