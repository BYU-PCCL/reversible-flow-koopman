IMAGE_TAG := robert:v1
CONTAINER_NAME := robert-pottorff-reversible-flow-koopman-2
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
	kernprof -l -o /dev/null main.py $(args)
	
docker-build:
	docker build --no-cache -t ${IMAGE_TAG} -f ./Dockerfile .

docker-run:
	ssh remote@rainbow.cs.byu.edu

test:
	python3 main.py --

experiment:
	# CUDA_VISIBLE_DEVICES=0 python main.py --epochs=500 --name=512-d10 --model.flow.flow.f.hidden=512 --model.max_hidden_dim=256 --train_dataset.sequence_length=25 --model.flow.permute=ReversePermutation &
	# CUDA_VISIBLE_DEVICES=1 python main.py --epochs=500 --name=256-d20 --model.flow.flow.f.hidden=256 --model.flow.num_layers_per_block=20 --model.max_hidden_dim=256 --train_dataset.sequence_length=25 --model.flow.permute=ReversePermutation &
	# CUDA_VISIBLE_DEVICES=2 python main.py --epochs=500 --name=512-d10-checkpoint --model.flow.checkpoint_gradients=True --model.flow.flow.f.hidden=512 --model.max_hidden_dim=256 --train_dataset.sequence_length=25 --model.flow.permute=ReversePermutation &
	# CUDA_VISIBLE_DEVICES=3 python main.py --epochs=500 --name=256-d20-checkpoint --model.flow.checkpoint_gradients=True --model.flow.flow.f.hidden=256 --model.flow.num_layers_per_block=20 --model.max_hidden_dim=256 --train_dataset.sequence_length=25 --model.flow.permute=ReversePermutation
	
	CUDA_VISIBLE_DEVICES=0 python main.py --epochs=5000 --batch_size=15 --name=512-d10-checkpoint --model.flow.checkpoint_gradients=True --model.flow.flow.f.hidden=512 --model.max_hidden_dim=256 --train_dataset.sequence_length=25 --model.flow.permute=ReversePermutation &
	CUDA_VISIBLE_DEVICES=1 python main.py --epochs=5000 --batch_size=15 --name=256-d20-checkpoint --model.flow.checkpoint_gradients=True --model.flow.flow.f.hidden=256 --model.flow.num_layers_per_block=20 --model.max_hidden_dim=256 --train_dataset.sequence_length=25 --model.flow.permute=ReversePermutation &
	CUDA_VISIBLE_DEVICES=2 python main.py --epochs=5000 --batch_size=15 --name=64-d80-checkpoint --model.flow.checkpoint_gradients=True --model.flow.flow.f.hidden=64 --model.flow.num_layers_per_block=80 --model.max_hidden_dim=256 --train_dataset.sequence_length=25 --model.flow.permute=ReversePermutation &
	CUDA_VISIBLE_DEVICES=4 python main.py --epochs=5000 --batch_size=15 --name=512-d10-checkpoint-64 --model.flow.checkpoint_gradients=True --model.flow.flow.f.hidden=512 --model.max_hidden_dim=256 --train_dataset.sequence_length=25 --train_dataset.width=64 --train_dataset.height=64 --model.flow.permute=ReversePermutation &
	CUDA_VISIBLE_DEVICES=5 python main.py --epochs=5000 --batch_size=15 --name=256-d20-checkpoint-64 --model.flow.checkpoint_gradients=True --model.flow.flow.f.hidden=256 --model.flow.num_layers_per_block=20 --model.max_hidden_dim=256 --train_dataset.width=64 --train_dataset.height=64 --train_dataset.sequence_length=25 --model.flow.permute=ReversePermutation &
	CUDA_VISIBLE_DEVICES=6 python main.py --epochs=5000 --batch_size=15 --name=512-d10-checkpoint-64-vel6 --model.flow.checkpoint_gradients=True --model.flow.flow.f.hidden=512 --model.max_hidden_dim=256 --train_dataset.sequence_length=25 --train_dataset.width=64 --train_dataset.height=64 --train_dataset.velocity=6 --model.flow.permute=ReversePermutation &
	CUDA_VISIBLE_DEVICES=7 python main.py --epochs=5000 --batch_size=15 --name=256-d20-checkpoint-64-vel6 --model.flow.checkpoint_gradients=True --model.flow.flow.f.hidden=256 --model.flow.num_layers_per_block=20 --model.max_hidden_dim=256 --train_dataset.width=64 --train_dataset.height=64 --train_dataset.velocity=6 --train_dataset.sequence_length=25 --model.flow.permute=ReversePermutation

	# CUDA_VISIBLE_DEVICES=4 python main.py --epochs=5000 --name=512-d10-checkpoint-128 --model.flow.checkpoint_gradients=True --model.flow.flow.f.hidden=512 --model.max_hidden_dim=256 --train_dataset.sequence_length=25 --train_dataset.width=128 --train_dataset.height=128 --model.flow.permute=ReversePermutation &
	# CUDA_VISIBLE_DEVICES=5 python main.py --epochs=5000 --name=256-d20-checkpoint-128 --model.flow.checkpoint_gradients=True --model.flow.flow.f.hidden=256 --model.flow.num_layers_per_block=20 --model.max_hidden_dim=256 --train_dataset.width=128 --train_dataset.height=128 --train_dataset.sequence_length=25 --model.flow.permute=ReversePermutation

	# let's try 
		# 128x128 bouncing mnist
		# strided convolutions in the flow block
		# deeper layers in the flow block



	# CUDA_VISIBLE_DEVICES=9 python main.py --epochs=500 --name=128-d40 --model.flow.flow.f.hidden=128 --model.flow.num_layers_per_block=40 --model.max_hidden_dim=256 --train_dataset.sequence_length=25 --model.flow.permute=ReversePermutation &
	# CUDA_VISIBLE_DEVICES=10 python main.py --epochs=500 --name=64-d40  --model.flow.flow.f.hidden=64  --model.flow.num_layers_per_block=40 --model.max_hidden_dim=256 --train_dataset.sequence_length=25 --model.flow.permute=ReversePermutation &

	# CUDA_VISIBLE_DEVICES=1 python main.py --name=invertible     --model.max_hidden_dim=256 --train_dataset.sequence_length=25 --model.flow.permute=Invertible1x1Conv &
	# CUDA_VISIBLE_DEVICES=2 python main.py --name=orthogonal     --model.max_hidden_dim=256 --train_dataset.sequence_length=25 --model.flow.permute=Orthogonal1x1Conv &
	# CUDA_VISIBLE_DEVICES=3 python main.py --name=glowloss                                  --train_dataset.sequence_length=2  --model=GlowPrediction &
	# CUDA_VISIBLE_DEVICES=4 python main.py --name=realjordan     --model.max_hidden_dim=256 --train_dataset.sequence_length=25 --model.A=StableRealJordanForm --resume=realjordan-01-17-071955 --optimizer.lr=0.00001 &
	# CUDA_VISIBLE_DEVICES=5 python main.py --name=svd            --model.max_hidden_dim=256 --train_dataset.sequence_length=25 --model.A=StableSVD

	# CUDA_VISIBLE_DEVICES=0 python3 main.py --name=exp1 --model.num_layers_per_block=15 --epochs=40 --model.flow.safescaler=SigmoidShiftScaler &
	# CUDA_VISIBLE_DEVICES=1 python3 main.py --name=exp1 --model.num_layers_per_block=15 --epochs=40 --model.flow.safescaler=AdditiveOnlyShiftScaler &
	# CUDA_VISIBLE_DEVICES=2 python3 main.py --name=exp1 --model.num_layers_per_block=15 --epochs=40 --model.flow.safescaler=GlowShift &
	# CUDA_VISIBLE_DEVICES=3 python3 main.py --name=exp1 --model.num_layers_per_block=15 --epochs=40 --model.flow.safescaler=SigmoidShiftScaler --optimizer.lr=1e-3 &
	# CUDA_VISIBLE_DEVICES=4 python3 main.py --name=exp1 --model.num_layers_per_block=15 --epochs=40 --model.flow.safescaler=AdditiveOnlyShiftScaler --optimizer.lr=1e-3 &
	# CUDA_VISIBLE_DEVICES=5 python3 main.py --name=exp1 --model.num_layers_per_block=15 --epochs=40 --model.flow.safescaler=GlowShift --optimizer.lr=1e-3 &
	# CUDA_VISIBLE_DEVICES=6 python3 main.py --name=exp1 --model.num_layers_per_block=15 --epochs=40 --model.flow.safescaler=GlowShift --optimizer.lr=1e-3 --max_grad_norm=50 &
	# CUDA_VISIBLE_DEVICES=7 python3 main.py --name=exp1 --model.num_layers_per_block=15 --epochs=40 --model.flow.safescaler=SigmoidShiftScaler --optimizer.lr=1e-3 --max_grad_norm=1000 &

	# CUDA_VISIBLE_DEVICES=0 python3 main.py --name=exp3 --model.num_layers_per_block=15 --epochs=40 --model.flow.safescaler=SigmoidShiftScaler &
	# CUDA_VISIBLE_DEVICES=1 python3 main.py --name=exp3 --model.num_layers_per_block=15 --epochs=40 --model.flow.safescaler=SigmoidShiftScaler --scheduler=TwoStage --scheduler.after=1e-5 &

	# CUDA_VISIBLE_DEVICES=0 python3 main.py --name=exp3 --model.num_layers_per_block=15 --epochs=40 --model.flow.safescaler=SigmoidShiftScaler --scheduler=TwoStage --scheduler.after=2e-4 &
	# CUDA_VISIBLE_DEVICES=1 python3 main.py --name=exp3 --model.num_layers_per_block=15 --epochs=40 --model.flow.safescaler=SigmoidShiftScaler --scheduler=TwoStage --scheduler.after=1e-5 &

docker-run-local:
	# nvidia-smi test if gpus are being used
	nvidia-docker run --workdir="${PWD}"  --rm --name=${CONTAINER_NAME} -v /mnt:/mnt -v /home:/home --hostname $(shell hostname) -t -i ${IMAGE_TAG} bash -i