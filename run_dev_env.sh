docker run -d \
    -p 8080:8080 \
    --name "ml-workspace" -v "${PWD}:/workspace" \
    --env AUTHENTICATE_VIA_JUPYTER="dagshub_savta" \
    --shm-size 2G \
    --restart always \
    mltooling/ml-workspace-minimal:latest
