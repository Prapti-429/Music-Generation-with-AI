This script handles loading and preprocessing MIDI files using the `music21` library.

```python
import os
import music21 as m21
import numpy as np

def load_midi_files(directory):
    """Loads and processes MIDI files from the specified directory."""
    midi_streams = []
    for file in os.listdir(directory):
        if file.endswith(".mid"):
            midi_streams.append(m21.converter.parse(os.path.join(directory, file)))
    return midi_streams

def preprocess_midi(streams):
    """Converts notes from MIDI streams into a list of note names."""
    notes = []
    for stream in streams:
        parts = m21.instrument.partitionByInstrument(stream)
        notes_to_parse = parts.parts[0].recurse() if parts else stream.flat.notes
        for element in notes_to_parse:
            if isinstance(element, m21.note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, m21.note.Rest):
                notes.append('rest')
    return notes

def prepare_sequences(notes, sequence_length):
    """Prepares the input sequences and target notes."""
    pitch_names = sorted(set(notes))
    note_to_int = {note: num for num, note in enumerate(pitch_names)}

    sequences = []
    targets = []

    for i in range(len(notes) - sequence_length):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        sequences.append([note_to_int[note] for note in sequence_in])
        targets.append(note_to_int[sequence_out])

    return np.array(sequences), np.array(targets), note_to_int, pitch_names
