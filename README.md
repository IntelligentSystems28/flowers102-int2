# flowers102-int2

The model itself can be found inside the model_flower.py, in addition to its state saving and loading, in which it also stores the criterion, optimizer and scheduler, and allows loading them too.

The model_train.py and model_test.py contain the functions to train and test the model respectively.

To train the model, use the main_train.py file. This will save the FlowerModel in the model_flower.py file, and then train the model.
<br>The variables inside the main.py class at the top can be altered to change how the model is trained.
<ul>
    <li>tr_batchsize - The number of images to train the model on for each step of an epoch.</li>
    <li>val_test_batchsize - The number of images to validate and test the model on.</li>
    <li>epochs - The number of epochs to train the model on.</li>
    <li>validate_steps - The number of steps between each validation during training. Only used if validate_stepped in the training function is set to True.</li>
    <li>load_model - If a model state dict should be loaded over the FlowerModel to be trained instead.</li>
    <li>model_name - The name of the file that the model should be saved to.</li>
</ul>


After training the model, it will save both the model and its dict, in addition to the criterion, optimizer and learning rate scheduler. The model should be loaded through the '{name}.pt' file, and the state through the '{name}-state.pt' file, mapped to the 'model' key inside the dictionary it loads.

You can also load through the functions contained in the model_flower.py file, giving the file location and model as input, or the Cross Entropy Loss criterion, Adam optimizer and/or StepLR scheduler to have their state dicts loaded too.
<br><br>

Testing the model will be done after the model has been trained through the main_train.py, but can also be done through loading the state dict file whilst running the main_test.py.


For testing a single image, you can change the image_loc path at the start of the main_predict.py and then run the python file. It will output the predicted flower and its likelihood, in addition to the next 4 predicted flowers and their likelihoods.