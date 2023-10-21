# Celebrity Lookalike Finder

This project builds a face recognition feature that takes as input the picture of someone and outputs the celebrity that looks the most like them.

It is based on the [YouTube Faces with Keypoints Dataset](https://www.kaggle.com/datasets/selfishgene/youtube-faces-with-facial-keypoints) from Kaggle.

# Demo
![demo](/img/celebrity_lookalike_demo.gif)

# How to run

The best way to run the app is to use the following Google Colab notebook: [Celebrity Lookalike Colab Notebook](https://colab.research.google.com/drive/1_CK7_WmrGgtoHEQsTQoBiTwlSeS_dKN-?usp=sharing)

We suggest to choose a High-RAM runtime (otherwise the notebook will crash when building the vector embeddings) and a GPU to speed up the embedding computations.

You can also run the app using your local runtime by following the steps below:
- Clone the repository
- Install the requirements in requirements.txt
- Get your Kaggle API credentials: To use the Kaggle API, sign up for a Kaggle account at https://www.kaggle.com. Then go to the 'Account' tab of your user profile and select 'Create API Token'. This will trigger the download of <code>kaggle.json</code>, a file containing your API credentials
- Place the <code>kaggle.json</code> file in the root of the repository
- Download the *Youtube Faces with Keypoints dataset* by running the command <code>make dataset</code>
- Build the ANN Index for the embeddings of all the faces in the dataset by running <code>python main.py</code>
- Launch the app by running <code>streamlit run app.py</code>
