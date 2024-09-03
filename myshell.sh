#!/bin/bash

# Specify the directory (change this to the desired directory)
DIR="/Users/robertopozzi/temp/robipozzi-ai/coursera/01_SupervisedMachineLearning/Week2/images"
DIR_TO="/Users/robertopozzi/dev/robipozzi-ai/coursera/01_SupervisedMachineLearning/Week2/images"

# Check if the directory exists
if [ -d "$DIR" ]; then
  # Cycle over the files in the directory
  for file in "$DIR"/*; do
    # Check if it's a file (not a directory)
    if [ -f "$file" ]; then
      # Print the file name
      fileIn="$(basename "$file")"
      echo $fileIn
      cp $DIR/$fileIn $DIR_TO/$fileIn

    fi
  done
else
  echo "Directory $DIR does not exist."
fi
