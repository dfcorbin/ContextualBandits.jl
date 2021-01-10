# Format code before deployment

echo "Formatting julia code..."
julia -e 'using JuliaFormatter; format(".")'

# Restage changes
git add $(git diff --name-only --cached)
git commit -m "$1"
git push
