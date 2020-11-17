import speech_recognition as sr
recognizer = sr.Recognizer()
audio_file_ = sr.AudioFile("E:/K8.wav")
i = 0
result = ''
#3230 is lenght of audio in seconds
while i < 3230:
    with audio_file_ as source:
	# audio recognized in 5 seconds blocks as the free quota is 10MB
        audio_file = recognizer.record(source, duration = 5.0, offset = i)
        try:
            result += recognizer.recognize_google(audio_data=audio_file)
            print(i)
            i += 5
        except Exception as e:
            i += 5
            print('error')
