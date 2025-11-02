#!/bin/bash
# Script to push code to GitHub
# Replace YOUR_USERNAME and REPO_NAME with your GitHub details

echo "==================================="
echo "Pushing to GitHub"
echo "==================================="

# Get GitHub username and repo name
read -p "Enter your GitHub username: " GITHUB_USERNAME
read -p "Enter your repository name (default: qdrant-pdf-tutorial): " REPO_NAME
REPO_NAME=${REPO_NAME:-qdrant-pdf-tutorial}

# Add remote
echo ""
echo "Adding GitHub remote..."
git remote add origin git@github.com:${GITHUB_USERNAME}/${REPO_NAME}.git 2>/dev/null || \
git remote set-url origin git@github.com:${GITHUB_USERNAME}/${REPO_NAME}.git

# Test SSH connection
echo ""
echo "Testing GitHub SSH connection..."
ssh -T git@github.com

# Push to GitHub
echo ""
echo "Pushing to GitHub..."
git push -u origin main

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Successfully pushed to GitHub!"
    echo "   Repository: https://github.com/${GITHUB_USERNAME}/${REPO_NAME}"
else
    echo ""
    echo "❌ Push failed. Make sure:"
    echo "   1. Your SSH key is added to GitHub"
    echo "   2. The repository exists on GitHub"
    echo "   3. You have write access to the repository"
fi

