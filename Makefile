.PHONY: dataset
dataset: ## download dataset from kaggle API (takes about 20 min)
	mkdir -p ~/.kaggle
	cp kaggle.json ~/.kaggle/
	chmod 600 ~/.kaggle/kaggle.json
	kaggle datasets download -d selfishgene/youtube-faces-with-facial-keypoints
	mkdir data
	unzip youtube-faces-with-facial-keypoints.zip -d data
	rm youtube-faces-with-facial-keypoints.zip