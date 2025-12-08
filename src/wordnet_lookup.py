from nltk.corpus import wordnet as wn

# Define the POS and offset
pos = wn.ADJ# Example: 'n' for noun, 'v' for verb, 'a' for adjective, 'r' for adverb
offset = 635456 # Example offset for a noun synset

# Retrieve the synset
synset = wn.synset_from_pos_and_offset(pos, offset)

# Print information about the synset
print(f"Synset: {synset}")
print(f"Definition: {synset.definition()}")
print(f"Lemmas: {synset.lemma_names()}")
