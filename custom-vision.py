
# coding: utf-8

from azure.cognitiveservices.vision.customvision.training import training_api
from azure.cognitiveservices.vision.customvision.training.models import ImageUrlCreateEntry

# Replace with a valid key
training_key = "YOUR_TRAINING_KEY"
prediction_key = "YOR_PREDICTION_KEY"

trainer = training_api.TrainingApi(training_key)

# Create a new project
print ("Creating project...")
project = trainer.create_project("Custom Vision")

# Make two tags in the new project
hardshell_tag = trainer.create_tag(project.id, "hardshell jackets")
insulated_tag = trainer.create_tag(project.id, "insulated jackets")

# In[9]:

#upload insulated hardshell jackets to custom vision
import os
hardshell_dir = "/home/maysam.mokarian/notebooks/files/gear_images/hardshell_jackets"

for image in os.listdir(os.fsencode(hardshell_dir)):
   with open(hardshell_dir + "/" + os.fsdecode(image), mode="rb") as img_data: 
       trainer.create_images_from_data(project.id, img_data, [ hardshell_tag.id ])

#upload insulated jackets to custom vision
import os
insulated_dir = "/home/maysam.mokarian/notebooks/files/gear_images/insulated_jackets"

for image in os.listdir(os.fsencode(insulated_dir)):
   with open(insulated_dir + "/" + os.fsdecode(image), mode="rb") as img_data: 
       trainer.create_images_from_data(project.id, img_data, [ insulated_tag.id ])

import time

print ("Training...")
iteration = trainer.train_project(project.id)
while (iteration.status != "Completed"):
    iteration = trainer.get_iteration(project.id, iteration.id)
    print ("Training status: " + iteration.status)
    time.sleep(1)

# The iteration is now trained. Make it the default project endpoint
trainer.update_iteration(project.id, iteration.id, is_default=True)
print ("Done!")

# In[12]:

from azure.cognitiveservices.vision.customvision.prediction import prediction_endpoint
from azure.cognitiveservices.vision.customvision.prediction.prediction_endpoint import models

# Now there is a trained endpoint that can be used to make a prediction

predictor = prediction_endpoint.PredictionEndpoint(prediction_key)

# Alternatively, if the images were on disk in a folder called Images alongside the sample.py, then
# they can be added by using the following.
#
# Open the sample image and get back the prediction results.
with open("/home/maysam.mokarian/notebooks/files/test/jacket.jpeg", mode="rb") as test_data:
    results = predictor.predict_image(project.id, test_data, iteration.id)

# Display the results.
for prediction in results.predictions:
    print ("\t" + prediction.tag_name + ": {0:.2f}%".format(prediction.probability * 100))

