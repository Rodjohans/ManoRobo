Follow the steps described in the report in part 3.

- Install the softwares and connect : Myoware -> the Adafruit Feather -> USB Adapter -> PC.

- From the Arduino software, open DataCollection.ino and televerse the code in the Feather.
You can see the myoware working from the Serial monitor (Tools). Close the software.

- With Anaconda open Spyder. From Spyder open ConstructionDataBase.py and verify the port number.
You don't need to add the name of the port following its number. Use this code to create your own database.
Be careful to retract the fingers only one time by experiment. To cancel, just press enter when the program stops
Close the csv file by entering 'dataBase.close()' command in the console.

- With Python open NeuralNetwork.py and run it. You can now edit the neural network and use it to forward the project.
