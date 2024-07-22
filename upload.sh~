#!/bin/bash

# Define the repository URL
REPO_URL="git@github.com:vladimirovertheworld/attractors.git"

# Function to handle errors
handle_error() {
    echo "Error: $1"
    exit 1
}

# Check if the repository URL is not empty
if [ -z "$REPO_URL" ]; then
    handle_error "REPO_URL is empty"
fi

# Initialize a new git repository
git init || handle_error "Failed to initialize git repository"

# Add all files in the current directory
git add . || handle_error "Failed to add files to the repository"

# Commit the files with a message
git commit -m "Initial commit" || handle_error "Failed to commit files"

# Add the remote repository
git remote add origin $REPO_URL || handle_error "Failed to add remote repository"

# Push the files to the remote repository
git push -u origin master || handle_error "Failed to push files to remote repository"

echo "Repository has been successfully pushed to $REPO_URL"
