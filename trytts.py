from src.tts.Synthesizer import Synthesizer as Syn

syn = Syn()

wav, sample_rate = syn.synth("You son of bitch!")
print(wav)
print(sample_rate)