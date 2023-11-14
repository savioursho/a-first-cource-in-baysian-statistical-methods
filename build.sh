#!/bin/sh

jupyter-book build \
    --path-output jupyterbook \
    --config jupyterbook/_config.yml \
    --toc jupyterbook/_toc.yml \
    .