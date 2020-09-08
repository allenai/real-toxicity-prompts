#!/bin/bash

echo "Building KenLM for $1..."
lmplz --order 3 --text "$1" --arpa "$1.arpa"
build_binary "$1.arpa" "$1.binary"
