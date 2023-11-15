#!/bin/bash

module purge

RESOLUTION=256
MAGNIFICATION=20x
MODALITY=heightmap
MODEL=ResNet50
PRETRAINED=pretrained
FROZEN=unfrozen
EPOCHS=10
BATCH_SIZE=200
START_LR=0.01
SEED=1234
VOTE=vote
declare -a ResolutionArray=(256 512 865)
declare -a MagnificationArray=(5x 10x 20x)
declare -a ModalityArray=(texture heightmap)
declare -a pretrainedArray=(pretrained not_pretrained)
declare -a frozenArray=(frozen unfrozen)
declare -a voteArray=(vote no_vote)

cd ..

for RESOLUTION in "${ResolutionArray[@]}"; do
	for MAGNIFICATION in "${MagnificationArray[@]}"; do
		for MODALITY in "${ModalityArray[@]}"; do
			for PRETRAINED in "${pretrainedArray[@]}"; do
				for FROZEN in "${frozenArray[@]}"; do
					for VOTE in "${voteArray[@]}"; do
						if [ "$PRETRAINED" == "pretrained" ];
						then
							EPOCHS=10
						else
							EPOCHS=20
						fi
						echo $RESOLUTION $MAGNIFICATION $MODALITY $PRETRAINED $FROZEN $EPOCHS $BATCH_SIZE $START_LR $SEED $VOTE
					done
				done
			done
		done
	done
done

