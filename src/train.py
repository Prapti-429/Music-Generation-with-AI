import numpy as np
from tensorflow.keras.utils import to_categorical
from data_loader import load_midi_files, preprocess_midi, prepare_sequences
from model import create_model
from music21 import stream
from midiutil import MIDIFile

# Parameters
sequence_length = 100
directory = 'data/'  # Folder where your MIDI files are stored

# Load and preprocess MIDI data
midi_streams = load_midi_files(directory)
notes = preprocess_midi(midi_streams)
sequences, targets, note_to_int, pitch_names = prepare_sequences(notes, sequence_length)

# Reshape input and normalize
n_vocab = len(pitch_names)
sequences = np.reshape(sequences, (len(sequences), sequence_length, 1))
sequences = sequences / float(n_vocab)
targets = to_categorical(targets)

# Create and train the model
model = create_model(sequence_length, n_vocab)
model.fit(sequences, targets, epochs=100, batch_size=64)

# Save the model
model.save('models/music_generation_rnn.h5')

# Function to generate new music
def generate_notes(model, int_to_note, sequence_length, n_vocab, num_notes=500):
    """Generates new notes from the trained model."""
    start = np.random.randint(0, len(sequences) - 1)
    input_sequence = sequences[start].tolist()

    generated_notes = []

    for _ in range(num_notes):
        prediction_input = np.reshape(input_sequence, (1, len(input_sequence), 1))
        prediction_input = prediction_input / float(n_vocab)

        prediction = model.predict(prediction_input, verbose=0)
        idx = np.argmax(prediction)
        note = int_to_note[idx]

        generated_notes.append(note)

        input_sequence.append([idx])
        input_sequence = input_sequence[1:]

    return generated_notes

int_to_note = {num: note for note, num in note_to_int.items()}
generated_notes = generate_notes(model, int_to_note, sequence_length, n_vocab)

# Function to create MIDI file from generated notes
def create_midi(generated_notes, output_file='output.mid'):
    """Creates a MIDI file from the generated notes."""
    midi_file = MIDIFile(1)
    midi_file.addTempo(0, 0, 120)

    time = 0
    for note in generated_notes:
        if note != 'rest':
            midi_file.addNote(0, 0, stream.note.Note(note).pitch.midi, time, 1, 100)
        time += 1

    with open(output_file, 'wb') as f:
        midi_file.writeFile(f)

# Save generated music as MIDI
create_midi(generated_notes)
