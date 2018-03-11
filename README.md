# train-neural-network
Automatically detect changes in the standard format data CSV and convert it to a data vector.
Use this data vector to train a neural network.

## Docker usage
Build a docker image with

`docker build -t train-neural-network .`

Then run the container with

`docker run train-neural-network`

Find the process ID with

`docker ps -a`

And copy the output folder from the container to the host

`docker cp <process_id>:/output/ src/output/`

Cleanup the container and image using

`docker rm $(docker ps -a -q) && docker rmi $(docker images -q)`
