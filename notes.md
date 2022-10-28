# 2022-10-19

I'd like to come up with a better algorithm for cutting segments of the midi-like repr; there must be a simpler way. (Which would make it more easily debuggable/extendable/etc.)

Currently the way it works is:

1. first make a list of note_on and note_off events
2. sort this list by time, putting note_off before note_on
3. convert this list to a list of time_shift, note_on, and note_off tokens

Then to segment:
... TODO

Start times can only be at eligible onsets
End times can only be at eligible releases
