# import keras
# import numpy as np
# import librosa

# import pyaudio
# import wave
# import speech_recognition as sr

# import librosa
# import librosa.display
# import numpy as np
# import matplotlib.pyplot as plt

# from sklearn.preprocessing import LabelEncoder

# RECORD_SECONDS = 5

# p = pyaudio.PyAudio()
# stream = p.open(format=pyaudio.paInt16,channels=2,rate=44100, input=True, frames_per_buffer=1024)

# print("**** recording")

# frames = []

# for i in range(0, int(44100 / 1024 * RECORD_SECONDS)):
#     data = stream.read(1024)
#     frames.append(data)

# print("**** done recording")

# stream.stop_stream()
# stream.close()
# p.terminate()

# wf = wave.open('output.wav', 'wb')
# wf.setnchannels(2)
# wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
# wf.setframerate(44100)
# wf.writeframes(b''.join(frames))
# wf.close()

# harvard = sr.AudioFile('output.wav')
# r = sr.Recognizer()
# with harvard as source:
#       audio = r.record(source)
#       text = r.recognize_google(audio)
# try:
#     print("You said: " +text)
# except sr.UnknownValueError:
#     print("Google Speech Recognition could not understand audio")
# except sr.RequestError as e:
#     print("Could not request results from Google Speech Recognition service; {0}".format(e))
    
    

# data, sampling_rate = librosa.load('output.wav')

# class livePredictions:
#     """
#     Main class of the application.
#     """
    
#     def __init__(self, path, file):
#         """
#         Init method is used to initialize the main parameters.
#         """
#         self.path = path
#         self.file = file

#     def load_model(self):
#         """
#         Method to load the chosen model.
#         :param path: path to your h5 model.
#         :return: summary of the model with the .summary() function.
#         """
#         self.loaded_model = keras.models.load_model(self.path)
#         return self.loaded_model.summary()

#     def makepredictions(self):
#         """
#         Method to process the files and create your features.
#         """
        
#         # Define the variable key
#         key = 0
#         global predictions
#         data, sampling_rate = librosa.load(self.file)
#         mfccs = np.mean(librosa.feature.mfcc(y=data, sr=sampling_rate, n_mfcc=40).T, axis=0)
#         x = np.expand_dims(mfccs, axis=1)
#         x = np.expand_dims(x, axis=0)
#         predictions = self.loaded_model.predict_step(x)
#         #predictions_int = predictions.astype(np.int)
#         print (predictions)

#         # Fix the error by using the any() function
        
#         print("Prediction is", " ", self.convertclasstoemotion(predictions))

#     @staticmethod
#     def convertclasstoemotion(pred):
#         """
#         Method to convert the predictions (int) into human readable strings.
#         """
        
#         label_conversion = {'0': 'neutral',
#                             '1': 'calm',
#                             '2': 'happy',
#                             '3': 'sad',
#                             '4': 'angry',
#                             '5': 'fearful',
#                             '6': 'disgust',
#                             '7': 'surprised'}

#         for key, value in label_conversion.items():
#             if any(int(key) == p for p in predictions[0]):
#                 label = value
           
#         return label

# # Replace path and file with the path of your model and of the file

# # from the RAVDESS dataset you want to use for the prediction

# pred = livePredictions(path='/audio analyser/Speech-Emotion-Recognition-using-ML-and-DL/Deep Learning/SER_model.h5',file='/audio analyser/Speech-Emotion-Recognition-using-ML-and-DL/examples/10-16-07-29-82-30-63.wav')

# pred.load_model()
# pred.makepredictions()