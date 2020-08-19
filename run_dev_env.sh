docker run --rm -p 8888:8888 \
  --ipc=host \
  --volume="$PWD:/workspace" \
  savta_depth_dev jupyter lab \
    --ip=0.0.0.0 \
    --port=8888 \
    --allow-root \
    --no-browser \
    --NotebookApp.token='' \
    --NotebookApp.password=''
