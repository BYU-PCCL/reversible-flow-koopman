IMAGE_TAG := robert:v1
CONTAINER_NAME := robert-pottorff-reversible-flow-koopman
TF := $(shell tempfile)

.SILENT: train, tensorboard

tensorboard:
	tensorboard --logdir=./logs --port=8080

clean:
	rm -rf logs checkpoints

.EXPORT_ALL_VARIABLES:
train:
	python3 main.py $(args)

profile:
	kernprof -l -o /dev/null main.py $(args) | less 
	
docker-build:
	docker build --no-cache -t ${IMAGE_TAG} -f ./Dockerfile .

docker-run:
	ssh remote@rainbow.cs.byu.edu

experiment:
	CUDA_VISIBLE_DEVICES=0 main.py --model.num_layers_per_block=15 --epochs=40 --model.flow.safescaler=SigmoidShiftScaler &
	CUDA_VISIBLE_DEVICES=1 main.py --model.num_layers_per_block=15 --epochs=40 --model.flow.safescaler=AdditiveOnlyShiftScaler &
	CUDA_VISIBLE_DEVICES=2 main.py --model.num_layers_per_block=15 --epochs=40 --model.flow.safescaler=GlowShift &
	CUDA_VISIBLE_DEVICES=3 main.py --model.num_layers_per_block=15 --epochs=40 --model.flow.safescaler=SigmoidShiftScaler --optimizer.lr=1e-3 &
	CUDA_VISIBLE_DEVICES=4 main.py --model.num_layers_per_block=15 --epochs=40 --model.flow.safescaler=AdditiveOnlyShiftScaler --optimizer.lr=1e-3 &
	CUDA_VISIBLE_DEVICES=5 main.py --model.num_layers_per_block=15 --epochs=40 --model.flow.safescaler=GlowShift --optimizer.lr=1e-3 &
	CUDA_VISIBLE_DEVICES=6 main.py --model.num_layers_per_block=15 --epochs=40 --model.flow.safescaler=GlowShift --optimizer.lr=1e-3 --max_grad_norm=50 &
	CUDA_VISIBLE_DEVICES=7 main.py --model.num_layers_per_block=15 --epochs=40 --model.flow.safescaler=SigmoidShiftScaler --optimizer.lr=1e-3 --max_grad_norm=1000 &

docker-run-local:
	# nvidia-smi test if gpus are being used
	nvidia-docker run --workdir="${PWD}"  --rm --name=${CONTAINER_NAME} -v /mnt:/mnt -v /home:/home -p 9999:9999 -p 8181:8080 --hostname $(shell hostname) -t -i ${IMAGE_TAG} bash -i