# gesture-classification
by Maximilian Huber (huber.maxi@northeastern.edu)

The easiest way to verify the figures of the report and ensure the models run is by running the demo.ipynb (it will load and run all the saved models from the model_weights folder). The Random Forest model weights are too big to store on GitHub, so I compressed them to a zip file and stored them in a Google Drive. If you download the zip file, unpack it, and move the two files into the model_weights folder, then everything should work.

The RF model weights can be found here: https://drive.google.com/file/d/1c_6HTnHywXFMvcOkWY4PJC8a2mNHzFXr/view?usp=sharing

The remaining notebooks are the ones I did the project work in. Phase 0 (phase_0.ipynb) contains all the data preprocessing steps and a few data explorations/analyses. Phase 1 (phase_1.ipynb) contains all the Random Forest and Support Vector Machine training and results. Phase 2 (phase_2.ipynb) contains all the Deep Neural Network definition, training, and results. The train_helpers.py and dataloader.py file contain some helper functions for visualizing results and loading data which I wrote to keep the notebook code clean.

This project taught me a lot about EMG data. I hope you like it. Thank you.
