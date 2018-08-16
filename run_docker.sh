#!/usr/bin/env bash
docker run --name carnd-capstone -p 4567:4567 -v $PWD:/capstone -v /tmp/log:/root/.ros/ --rm -it capstone
